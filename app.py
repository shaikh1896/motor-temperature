import numpy as np
import pandas as pd
from flask import jsonify
from flask import Flask, request, render_template
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        input_features = [float(x) for x in request.form.values()]
        features_value = [np.array(input_features)]
    
    features_name = ['u_q', 'coolant', 'stator_winding', 'u_d', 'stator_tooth',
       'motor_speed', 'i_d', 'i_q', 'stator_yoke', 'ambient', 'torque']
    
    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)
    
    if output<0:
        return render_template('index.html',prediction_texts="Sorry we cannot predict temperature")
    else:
        return render_template('index.html',prediction_text="the temperature of motor is {}".format(output))
        


if __name__=="__main__":
    app.run(debug=True)
