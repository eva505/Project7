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
git_branch = 'dev2'

app = Flask(__name__)
@app.route('/')
def index():
    return "<h1>Home Credit Score App</h1>"

LOCAL_PATH = os.getcwd()  + '//Data//'
REMOTE_PATH = 'https://raw.githubusercontent.com/eva505/Project7/'+git_branch+'/Data/'
MODELNAME = 'LGBM'
if local:
    CLIENTDATA = 'data_processed_min.csv'
    SHAPVALUESNAME = 'LGBM_SHAPvalues_min'
else :
    CLIENTDATA = 'data_processed_min_min.csv'
    SHAPVALUESNAME = 'LGBM_SHAPvalues_min_min'

FEATURES = 'features.csv'
SHAPNAME = 'LGBM_SHAP'

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
client_defaulted = df['TARGET']
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

#HELPER FUNCTIONS

def client_shap_data(client_id) :
    client_data = df[client_ids == int(client_id)]
    if len(client_data) :
        if local :
            shap_value = explainer(client_data, max_evals=max_evals_explainer)[0]
            shap_value.values = shap_value.values[:,1]
            shap_value.base_values = shap_value.base_values[1]
        else :
            shap_value = explainer[client_ids[client_ids == int(client_id)].index.values[0]]
            shap_value.value = shap_value.value[:,1]
            shap_value.base_values = shap_value.base_values[1]
        shap_data = pd.DataFrame(np.array([abs(shap_value.values), shap_value.values, shap_value.data.round(3)]).T, 
                                  index=shap_value.feature_names, 
                                  columns=["SHAP_Strength","SHAP", "Data"])
        shap_data = shap_data.sort_values(by="SHAP_Strength", ascending=False)
    else :
        shap_data = None
    return shap_data

#functions for producing the data filter on the features
def filter_feature(key):
    d = {'gender' : 'APPL_CODE_GENDER',
         'defaulted' : 'TARGET',
         'income' : 'APPL_AMT_INCOME_TOTAL'}
    return d[key]

def filter_function(comp, value, ref=None):
    if ref == None :
        d = {'eq' : lambda x : x == value,
             'range' : lambda x : (x <= x.mean()*(1+value/200)) & (x >= x.mean()*(1-value/200))}
    else :
        d = {'eq' : lambda x : x == value,
             'range' : lambda x : (x <= ref*(1+value/200)) & (x >= ref*(1-value/200))}
    return d[comp]

def create_select(series, func_type, value, client_index):
    ref = series[series.index == client_index].values[0]
    select = filter_function(func_type, value, ref)(series)
    return select

def create_mask(filter_dict, client_id):
    select = pd.Series([True]*len(df), name='Filter')
    try:
        client_index = df[client_ids == client_id].index.values[0]
    except:
        client_index = None
    for filter in filter_dict.values():
        feature_type = filter['type']
        if feature_type == 'defaulted':
            series = client_defaulted
        else :
            series = df[filter_feature(feature_type)]
        func_type = filter['func']
        value = filter['value']
        select = select & create_select(series, func_type, value, client_index)
    return select


#APP ROUTES

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

#send filter
@app.route('/filter', methods=['POST'])
def return_filter():
    filter_dict = json.loads(request.data)["filter_dict"]
    client_id = int(json.loads(request.data)["client_id"])
    select = create_mask(filter_dict, client_id)
    return json.dumps({'feature_filter' : select.to_json(orient='records')})

#send feature data
@app.route('/features', methods=['POST'])
def return_feature(df=df):
    feature = json.loads(request.data)["feature"]
    data = df[feature]
    return json.dumps({'feature_data' : data.to_json(orient='records')})



if __name__ == '__main__':
    app.run(debug=True)