from flask import Flask, request
import joblib, os, json
import pandas as pd
import numpy as np
import shap


app = Flask(__name__)
@app.route('/')
def index():
    return "<h1>Home Credit Score App</h1>"


LOCAL_PATH = os.getcwd()  + '//Data//'
REMOTE_PATH = 'https://github.com/eva505/Project7/blob/main/Data/'
MODELNAME = 'LogRegr0'
CLIENTDATA = 'data_processed_min.csv'
DATAPATH = REMOTE_PATH
DATA_URL = DATAPATH + CLIENTDATA
MODEL_URL = DATAPATH + MODELNAME


#load data
df = pd.read_csv(DATA_URL).drop(columns='Unnamed: 0').sort_values(by='SK_ID_CURR')
client_ids = df['SK_ID_CURR']
client_ids_json = client_ids.to_json(orient='records')
#load estimator
estimator = joblib.load(filename=MODEL_URL)


@app.route('/client_ids', methods=['POST'])
def return_client_ids(client_ids=client_ids):
    #client_ids = client_ids.to_json(orient='records')
    return client_ids_json

@app.route('/client_data', methods=['POST'])
def return_client_data(df=df):
    client_id = json.loads(request.data)["client_id"]
    try:
        client_id = int(client_id)
    except:
        client_id = None
    client_data = df[df['SK_ID_CURR'] == client_id]
    return json.dumps({'data' : client_data.to_json()})

@app.route('/prediction', methods=['POST'])
def return_prediction(estimator=estimator):
    #utiliser la fonction présedent
    client_data = pd.read_json(json.loads(request.data)["client_data"])
    if len(client_data) :
        y_pred = estimator.predict_proba(client_data)[:, 1][0]
    else :
        y_pred = None
    return json.dumps({'pred' : y_pred})


if __name__ == '__main__':
    app.run(debug=True)