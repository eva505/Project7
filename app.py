from distutils.log import debug
from flask import Flask, request
import os, json, sys
import urllib.request
import joblib
import pandas as pd
import numpy as np
import shap

#where to get data from - local or GitHub, if GitHub then choose the correct branch
local = False
git_branch = 'main'

app = Flask(__name__)
@app.route('/')
def index():
    return "<h1>Home Credit Score App</h1>"

LOCAL_PATH = os.getcwd()  + '//Data//'
REMOTE_PATH = 'https://raw.githubusercontent.com/eva505/Project7/'+git_branch+'/Data/'
MODELNAME = 'LogRegr0'
if local:
    CLIENTDATA = 'data_processed_min.csv'
else :
    CLIENTDATA = 'data_processed_min_min.csv'
FEATURES = 'features.csv'
SHAPNAME = 'LogRegr0_SHAP'
SHAPVALUESNAME = 'LogRegr0_SHAPvalues'

if local :
    DATAPATH = LOCAL_PATH
else :
    DATAPATH = REMOTE_PATH
DATA_URL = DATAPATH + CLIENTDATA
FEATURE_URL = DATAPATH + FEATURES
MODEL_URL = DATAPATH + MODELNAME
SHAP_URL = DATAPATH + SHAPNAME
SHAPVALUES_URL = DATAPATH + SHAPVALUESNAME

#load data
df = pd.read_csv(DATA_URL, sep=',').drop(columns='Unnamed: 0').sort_values(by='SK_ID_CURR')
client_ids = df['SK_ID_CURR']
client_ids_json = client_ids.to_json(orient='records')
#load features and filter the data to only include these features
features = pd.read_csv(FEATURE_URL, sep=',').drop(columns='Unnamed: 0')['features'].values
df = df.filter(items=features)
max_evals_explainer = 2*2*len(features)
#load estimator and shap explainer
if local :
    estimator = joblib.load(MODEL_URL)
    with open(SHAP_URL, "rb") as e:
        explainer = shap.Explainer.load(e)
else :
    estimator = joblib.load(urllib.request.urlopen(MODEL_URL))
    #with urllib.request.urlopen(SHAP_URL) as e:
        #explainer = shap.Explainer.load(e)
    explainer = joblib.load(urllib.request.urlopen(SHAPVALUES_URL))


def client_shap_data(client_id) :
    client_data = df[client_ids == int(client_id)]
    if len(client_data) :
        if local :
            shap_value = explainer(client_data, max_evals=max_evals_explainer)[0]
        else :
            shap_value = explainer[client_ids[client_ids == int(client_id)].index.values[0]]   
        shap_data = pd.DataFrame(np.array([abs(shap_value.values), shap_value.values, shap_value.data.round(3)]).T, 
                                  index=shap_value.feature_names, 
                                  columns=["SHAP_Strength","SHAP", "Data"])
        shap_data = shap_data.sort_values(by="SHAP_Strength", ascending=False)
    else :
        shap_data = None
    return shap_data


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
def return_shapvalues():
    #client_data = pd.read_json(json.loads(request.data)["client_data"])
    client_id = json.loads(request.data)["client_id"]
    shap_data = client_shap_data(client_id)
    return json.dumps({'SHAP_data' : shap_data.to_json()})

if __name__ == '__main__':
    app.run(debug=True)