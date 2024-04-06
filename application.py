from flask import Flask,jsonify,request,render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
application = Flask(__name__)
app=application
ridge=pickle.load(open('models/ridge.pkl','rb'))
scaler=pickle.load(open('models/scaler.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        Temperature=float(request.form.get('Temperature'))
        Rh=float(request.form.get('Rh'))
        Ws=float(request.form.get('Ws'))
        Rain=float(request.form.get('Rain'))
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC'))
        ISI=float(request.form.get('ISI'))
        Classes=float(request.form.get('Classes'))
        Region=float(request.form.get('Region'))
        scaled_data=scaler.transform([[Temperature,Rh,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge.predict(scaled_data)
        return render_template('home.html',result=result[0])
    else:
        return render_template('home.html')
if __name__=="__main__":
    app.run(host="0.0.0.0")
