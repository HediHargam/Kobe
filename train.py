seed = 42
import numpy as np
np.random.seed(seed)
import random
random.seed(seed)
import pandas as pd
from logzero import logger
import itertools
import matplotlib.pyplot as plt
#from imblearn.over_sampling import RandomOverSampler, SMOTE
#from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import pickle

grid_search = False

df = pd.read_csv('data.csv')


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=90)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


class KobeModel():

    def __init__(self, target: str = 'shot_made_flag'):

        """
        Class to create dataset for training/testing of Rig State machine learning algorithm.

        It will create X and y, the features and target Rig State based on concatened csv files of tagged datasets
        Possibility to tune the dataset with normalization, balance, temporality, encoded target


        Return X, y features and target Rig State
        """
        self.cols_to_dummies = ['shot_type', 'combined_shot_type', 'shot_zone_area',
                                'shot_zone_basic', 'shot_zone_range', 'action_type', 'opponent']
        self.to_norm = []
        self.past_features = []
        self.target = target

    @classmethod
    def add_temporality(self, df: pd.DataFrame, features: list, win: int = 20) -> pd.DataFrame:
        """
        Add features variation based on a rolling window mean

        Parameters
        ----------
        df: Dataframe
        past_mnemonics: list of mnemonics to add temporal feature
        Returns
        -------
        df: pd.Dataframe with temporal features
        """
        for FEAT in features:
            df[FEAT + '_5'] = (df[FEAT] - df[FEAT].rolling(window=int(win), min_periods=0).mean()).bfill()
        return df

    @classmethod
    def shift_values(self, df: pd.DataFrame, features: list):

        for FEAT in features:
            for i in range(1, 5):
                df[FEAT + '_shift' + str(i)] = df[FEAT].shift(i).bfill()
        return df


    @classmethod
    def normalize_features(self, df, to_norm: list) -> pd.DataFrame:
        """
        Normalize features

        Parameters
        ----------
        df: pd.DataFrame
        features: list of feature to normalize
        Returns
        -------
        df: pd.Dataframe with normalized features
        """
        for FEAT in to_norm:
            df[FEAT] = (df[FEAT] - df[FEAT].mean()) / df[FEAT].std()
        return df

    @classmethod
    def balance_dataset(self, X: pd.DataFrame, y: pd.Series, OVERSAMPLING=True) -> pd.DataFrame:

        """
        Balance the dataset
        Parameters
        ----------
        X: pd.DataFrame of features
        y: pd.Series of target
        OVERSAMPLING: if True the strategy is Oversampling else it's Undersampling
        Returns
        -------
        X: pd.Dataframe of resampled features
        y: pd.Series of resampled target
        """
        logger.info('BALANCING DATASET')
        sample = RandomOverSampler(sampling_strategy='minority') if OVERSAMPLING else RandomUnderSampler(
                sampling_strategy='majority')
        X, y = sample.fit_resample(X, y)

        return X, y

    def transform(self, df: pd.DataFrame,
                        NORM=False,
                        TEMPO=False,
                        BALANCE=False) -> pd.DataFrame:
        """
        Create/Modify the dataset according to a feature engineering
        Parameters
        ----------
        X: pd.DataFrame
        past_mnemonics: list of mnemonics to add temporal feature
        feature: list of feature to correct and fill
        NORM: boolean if True, apply normalisation to feature
        TEMPO: boolean if True, apply temporality to past_mnemonics
        Returns
        -------
        X: pd.Dataframe with feature engineering
        y: the target
        """
        tempo = df.copy()
        logger.info('TRANSFORMING DATASET')

        if 'matchup' in df.columns:
            df['home'] = df['matchup'].str.contains('vs').astype('int')  # 1 if home play else 0

        if 'action_type' in df.columns:
            action_types = df['action_type'].value_counts().index.values[:15]  # 96% of action type are in the 15 first (of 57) actions
            df.loc[~df['action_type'].isin(action_types), 'action_type'] = 'Other'

        for col in self.cols_to_dummies:
            if col in df.columns:
                df = pd.concat([df, pd.get_dummies(df[col], prefix=col)], axis=1)
                df.drop(columns=col, inplace=True,  errors='ignore')

        if 'game_date' in df.columns:
            df['game_date'] = pd.to_datetime(df['game_date'])
            df['exp_year'] = df['game_date'].dt.year - df['game_date'].dt.year.min()
            df['game_month'] = df['game_date'].dt.month

        df.drop(columns='game_date', inplace=True, errors='ignore')

        df = self.normalize_features(df, to_norm=self.to_norm) if NORM else df

        df = self.add_temporality(df, features=self.past_features) if TEMPO else df
        df = self.shift_values(df, features=self.past_features) if TEMPO else df

        df.drop(columns=['game_event_id', 'game_id', 'team_name', 'team_id', 'matchup', 'shot_id', 'season', 'loc_x', 'loc_y'],
                inplace=True,  errors='ignore') #Drop usless/redondant features

        df.dropna(inplace=True)

        X = df.drop(columns=self.target, errors='ignore')

        X = X[['minutes_remaining', 'shot_distance', 'lat', 'lon']] #for app purpose

        y = df[self.target] if self.target in df.columns else 0

        if BALANCE:
            X, y = self.balance_dataset(X, y)


        return X, y


if __name__ == '__main__':

    ### Model training --> Transform and fit

    X, y = KobeModel().transform(df)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed, shuffle=True)

    Rforest = RandomForestClassifier(n_estimators=100, random_state=seed, min_samples_split=20, criterion='gini')

    logger.info('Training Model')
    Rforest.fit(X_train, y_train)


    if grid_search:

        param_grid = {
                         'n_estimators': [150, 200],
                         'min_samples_split': [2, 10, 20],
                         'max_depth' : [None, 10 , 20 , 50],
                         'criterion' : ['gini', 'entropy']
                     }


        grid_clf = GridSearchCV(Rforest, param_grid)
        grid_clf.fit(X_train, y_train)
        grid_clf.best_params_


    #### Evaluate the Model #### (Better in jupyter notebook)

    logger.info('Evaluating Model Results')

    #for name,score in zip(X_test.columns, Rforest.feature_importances_):
      #print(name,score*100)


    eval_y = Rforest.predict(X_test)
    print({'TEST SCORE': Rforest.score(X_test, y_test), 'TRAIN SCORE': Rforest.score(X_train, y_train)})


    conf_mx = confusion_matrix(y_test, eval_y)
    plot_confusion_matrix(conf_mx,target_names = ['0', '1'],
                              title='Confusion matrix',
                              cmap=None,
                              normalize=False)

    print(classification_report(y_test, eval_y))



    #### Save The final Model ####


    #pickle.dump(Rforest, open('RF_Kobe.pkl', 'wb'))