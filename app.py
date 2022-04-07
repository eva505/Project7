from flask import Flask, request
import urllib.request
import joblib, os, json
import pandas as pd
import numpy as np
import shap

#where to get data from - local or GitHub, if GitHub then choose the correct branch
local = True
git_branch = 'dev1'

app = Flask(__name__)
@app.route('/')
def index():
    return "<h1>Home Credit Score App</h1>"

LOCAL_PATH = os.getcwd()  + '//Data//'
REMOTE_PATH = 'https://raw.githubusercontent.com/eva505/Project7/'+git_branch+'/Data/'
MODELNAME = 'LogRegr0'
CLIENTDATA = 'data_processed_min.csv'
FEATURES = 'features.csv'

if local :
    DATAPATH = LOCAL_PATH
else :
    DATAPATH = REMOTE_PATH
DATA_URL = DATAPATH + CLIENTDATA
FEATURE_URL = DATAPATH + FEATURES
MODEL_URL = DATAPATH + MODELNAME

#load data
df = pd.read_csv(DATA_URL, sep=',').drop(columns='Unnamed: 0').sort_values(by='SK_ID_CURR')
client_ids = df['SK_ID_CURR']
client_ids_json = client_ids.to_json(orient='records')
#load features and filter the data to only include these features
features = pd.read_csv(FEATURE_URL, sep=',').drop(columns='Unnamed: 0')['features'].values
df = df.filter(items=features)
#load estimator
if local :
    estimator = joblib.load(MODEL_URL)
else :
    estimator = joblib.load(urllib.request.urlopen(MODEL_URL))

#send client data
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


if __name__ == '__main__':
    app.run(debug=True)