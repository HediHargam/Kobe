from flask import Flask, request, render_template
import pickle
from train import KobeModel
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('RF_Kobe.pkl', 'rb'))

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
    final_features, y = KobeModel().transform(final_features)
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    txt = 'score this shot' if output == 1 else 'miss this shot'

    return render_template('index.html', prediction_text='Kobe Bryant should ' + txt)


if __name__ == "__main__":
    app.run(debug=True)
