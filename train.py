seed = 42
import numpy as np
np.random.seed(seed)
import random
random.seed(seed)
import pandas as pd
from logzero import logger
#from imblearn.over_sampling import RandomOverSampler, SMOTE
#from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
import pickle

grid_search = False

df = pd.read_csv('data.csv')

df.dropna(inplace=True)


class KobeModel():

    def __init__(self, target: str = 'shot_made_flag'):

        """
        Class to create dataset for training/testing of machine learning algorithm for Kobe Bryant shot prediction.

        It will create X and y, the features and target
        Possibility to tune the dataset with normalization, balance, or temporality


        Return X, y features and target shot result
        """
        self.cols_to_dummies = ['shot_type', 'shot_zone_area',
                                'shot_zone_basic', 'shot_zone_range']
        self.col_to_label = ['opponent', 'action_type', 'combined_shot_type']
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
        features: list of mnemonics to add temporal feature
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
                        TEMPO=False) -> pd.DataFrame:
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
        logger.info('TRANSFORMING DATASET')

        if 'matchup' in df.columns:
            df['home'] = df['matchup'].str.contains('vs').astype('int')  # 1 if home play else 0

        if 'action_type' in df.columns:
            action_types = df['action_type'].value_counts().index.values[:30]  # 96% of action type are in the 15 first (of 57) actions
            df.loc[~df['action_type'].isin(action_types), 'action_type'] = 'Other'

        for col in self.cols_to_dummies:
            if col in df.columns:
                df = pd.concat([df, pd.get_dummies(df[col], prefix=col)], axis=1)
                df.drop(columns=col, inplace=True,  errors='ignore')

        le = LabelEncoder()
        for col in self.col_to_label:
            if col in df.columns:
                df[col] = le.fit_transform(df[col])

        if 'game_date' in df.columns:
            df['game_date'] = pd.to_datetime(df['game_date'])
            df['exp_year'] = df['game_date'].dt.year - df['game_date'].dt.year.min() #number of year of career
            df['game_month'] = df['game_date'].dt.month
            df.drop(columns='game_date', inplace=True, errors='ignore')

        df = self.normalize_features(df, to_norm=self.to_norm) if NORM else df
        df = self.add_temporality(df, features=self.past_features) if TEMPO else df
        df = self.shift_values(df, features=self.past_features) if TEMPO else df

        df.drop(columns=['game_event_id', 'game_id', 'team_name', 'team_id', 'matchup', 'shot_id', 'season',
                         'loc_x', 'loc_y', 'playoffs'],
                inplace=True,  errors='ignore') #Drop usless/redondant features


        return df


    def fit_transform(self, df, BALANCE=False):

        df = self.transform(df)

        X = df.drop(columns=self.target, errors='ignore')

        X = X[['minutes_remaining', 'shot_distance', 'lat', 'lon']] #for app purpose

        y = df[self.target] if self.target in df.columns else 0

        if BALANCE:
            X, y = self.balance_dataset(X, y)

        return X, y



if __name__ == '__main__':

    ### Model training --> Transform and fit

    X, y = KobeModel().fit_transform(df,
                                     BALANCE=False)

    X_train, X_test, y_train, y_test = train_test_split(X, y
                                                        ,test_size=0.3,
                                                        random_state=seed,
                                                        shuffle=True)

    Rforest = RandomForestClassifier(n_estimators=400,
                                     random_state=seed,
                                     min_samples_split=20,
                                     criterion='gini')

    lgbm = LGBMClassifier(objective='binary',
                          max_depth=10,
                          num_iterations=1000,
                          learning_rate=0.01)


    logger.info('Training Model')
    Rforest.fit(X_train, y_train)
    lgbm.fit(X_train, y_train)


    if grid_search:

        param_grid = {
                         'n_estimators': [150, 200],
                         'min_samples_split': [2, 10, 20],
                         'max_depth' : [None, 10 , 20 , 50],
                         'criterion' : ['gini', 'entropy']
                     }


        grid_clf = GridSearchCV(Rforest,
                                param_grid
                                , cv=4)
        grid_clf.fit(X_train, y_train)
        grid_clf.best_params_


    #### Evaluate the Model #### (Better in jupyter notebook)

    logger.info('Evaluating Model Results')

    for name,score in zip(X_test.columns, Rforest.feature_importances_):
      print(name,score*100)

    eval_y = ((Rforest.predict_proba(X_test)[:,1] + lgbm.predict_proba(X_test)[:,1])/2).round() #Use two models to incease robustness

    print({'TEST SCORE': Rforest.score(X_test, y_test), 'TRAIN SCORE': Rforest.score(X_train, y_train)})

    print(confusion_matrix(y_test, eval_y))

    print(classification_report(y_test, eval_y))

    print(roc_auc_score(y_test, eval_y))


    #### Save The final Model ####

   #pickle.dump(Rforest, open('RF_Kobe.pkl', 'wb'))
   #pickle.dump(Rforest, open('LGBM_Kobe.pkl', 'wb'))



