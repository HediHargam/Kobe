from flask import Flask, request, render_template
import pickle
from train import KobeModel
import pandas as pd

app = Flask(__name__)
Rforest = pickle.load(open('RF_Kobe.pkl', 'rb'))
lgbm = pickle.load(open('LGBM_Kobe.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    data = [["minutes_remaining", "shot_distance", "lat", "lon"], int_features]
    column_names = data.pop(0)
    final_features = pd.DataFrame(data, columns=column_names)
    final_features = KobeModel().transform(final_features)
    output = ((Rforest.predict_proba(final_features)[:,1] + lgbm.predict_proba(final_features)[:,1])/2).round()

    txt = 'score this shot' if output == 1 else 'miss this shot'

    return render_template('index.html', prediction_text='Kobe Bryant should ' + txt)


if __name__ == "__main__":
    app.run(debug=True)
