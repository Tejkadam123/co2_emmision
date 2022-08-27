from flask import Flask,request,render_template,jsonify
import numpy as np
import pandas as pd
import pickle


with open('model.pkl','rb') as f:
    model = pickle.load(f)

with open('normality_transform.pkl','rb') as f:
    norm = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def inedx():
    return render_template('result.html')

@app.route('/Vehicle_data',methods=['POST'])
def predict():
    Engine_Size = float(request.form['ES'])
    Cylinders =  float(request.form['cyl'])
    Fuel_Type =  float(request.form['FT'])
    Fuel_Consumption_City =  float(request.form['FC_C'])
    Fuel_Consumption_Hwy =  float(request.form['FC_H'])
    Fuel_Consumption_Comb =  float(request.form['FC_comb'])
    Fuel_Consumption_Comb = float(request.form['FC_comb_mpg'])

    data=np.array([[Engine_Size,Cylinders,Fuel_Type,Fuel_Consumption_City,Fuel_Consumption_Hwy,Fuel_Consumption_Comb,Fuel_Consumption_Comb]])

    transform_data = norm.transform(data)
    co2_emission = model.predict(transform_data)

    return (f'Co2 Emission occured by behicle = {co2_emission}')


if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8080)

