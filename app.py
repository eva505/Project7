from distutils.log import debug
from flask import Flask, request
import os, json, sys
import urllib.request
import joblib
import pandas as pd
import numpy as np
import shap

local = False
git_branch = 'dev0'

app = Flask(__name__)
@app.route('/')
def index():
    return "<h1>Home Credit Score App</h1>"

LOCAL_PATH = os.getcwd()  + '//Data//'
REMOTE_PATH = 'https://raw.githubusercontent.com/eva505/Project7/'+git_branch+'/Data/'
MODELNAME = 'LogRegr0'
CLIENTDATA = 'data_processed_min.csv'
SHAPNAME = 'LogRegr0_SHAP'

if local :
    DATAPATH = LOCAL_PATH
else :
    DATAPATH = REMOTE_PATH
DATA_URL = DATAPATH + CLIENTDATA
MODEL_URL = DATAPATH + MODELNAME
SHAP_URL = DATAPATH + SHAPNAME

#load data
df = pd.read_csv(DATA_URL, sep=',').drop(columns='Unnamed: 0').sort_values(by='SK_ID_CURR')
client_ids = df['SK_ID_CURR']
client_ids_json = client_ids.to_json(orient='records')
df = df.drop(columns=['SK_ID_CURR', 'TARGET'])
#load estimator and shap explainer
if local :
    estimator = joblib.load(MODEL_URL)
    with open(SHAP_URL, "rb") as e:
        explainer = shap.Explainer.load(e)
else :
    estimator = joblib.load(urllib.request.urlopen(MODEL_URL))
    with urllib.request.urlopen(SHAP_URL) as e:
        explainer = shap.Explainer.load(e)


#send client ids
@app.route('/client_ids', methods=['POST'])
def return_client_ids(client_ids=client_ids):
    client_ids = client_ids.to_json(orient='records')
    return client_ids

#send client data
@app.route('/client_data', methods=['POST'])
def return_client_data(df=df):
    client_id = json.loads(request.data)["client_id"]
    try:
        client_id = int(client_id)
    except:
        client_id = None
    client_data = df[client_ids == client_id]
    return json.dumps({'data' : client_data.to_json()})

#send client default risk
@app.route('/prediction', methods=['POST'])
def return_prediction(estimator=estimator):
    #utiliser la fonction présedent
    #client_data = pd.read_json(json.loads(request.data)["client_data"])
    client_id = json.loads(request.data)["client_id"]
    client_data = df[client_ids == int(client_id)]
    if len(client_data) :
        y_pred = estimator.predict_proba(client_data)[:, 1][0]
    else :
        y_pred = None
    return json.dumps({'pred' : y_pred})

#send feature importance
@app.route('/shapvalues', methods=['POST'])
def return_shapvalues(explainer=explainer):
    #client_data = pd.read_json(json.loads(request.data)["client_data"])
    client_id = json.loads(request.data)["client_id"]
    client_data = df[client_ids == int(client_id)]
    if len(client_data) :
        shap_values = explainer(client_data, max_evals=1500)[0]
        shap_data = pd.DataFrame(np.array([abs(shap_values.values), shap_values.values, shap_values.data.round(3)]).T, 
                                 index=shap_values.feature_names, 
                                 columns=["SHAP_Strength","SHAP", "Data"]).sort_values(by="SHAP_Strength", ascending=False)
    else :
        shap_data = None
    return json.dumps({'SHAP_data' : shap_data.to_json()})

if __name__ == '__main__':
    app.run(debug=True)