import streamlit as st
import pandas as pd
import requests
import plotly.graph_objs as go

def initialize():
    if 'home' not in st.session_state :
        st.session_state.home = True
    if 'client_id' not in st.session_state :
        st.session_state.client_id = None
    if 'filter_dict' not in st.session_state :
        st.session_state.filter_dict = {}
    if 'client_id_selected' not in st.session_state:
        st.session_state.client_id = None
    if st.session_state.home == True :
        st.session_state.client_id = None

def client_id_submitted():
    st.session_state.home = False
    st.session_state.client_id = st.session_state.client_id_selected

@st.experimental_memo
def load_client_ids(clients_uri):
    clients_ids = requests.request(method='POST', url=clients_uri)
    return clients_ids.json()

@st.experimental_memo
def load_client_data(client_id, clients_data_uri):
    headers = {"Content-Type": "application/json"}
    data_json = {'client_id': str(client_id)}
    response = requests.request(method='POST', headers=headers, url=clients_data_uri, json=data_json)
    return pd.read_json(response.json()["data"])

@st.experimental_memo
def get_client_prediction(client_data, prediction_uri):
    headers = {"Content-Type": "application/json"}
    data_json = {'client_data': client_data.to_json(orient='records')}
    response = requests.request(method='POST', headers=headers, url=prediction_uri, json=data_json)
    return float(response.json()["pred"])

def create_gauge(prediction):
    steps_range = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    steps_colors = ['green', 'lightgreen', 'yellow', 'orange', 'red']
    #steps_list = [{'range' : [steps_range[i], steps_range[i+1]], 
    #               'color' :  steps_colors[i] } for i in range(len(steps_range)-1)]
    pred_color = steps_colors[0]
    for i, step in enumerate(steps_range):
        if prediction <= step :
            break
        pred_color = steps_colors[i]
    fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prediction,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    #title = {'text': "Risk"},
                    gauge = { 'axis': {'range':[0,1]},
                              #'steps': steps_list,
                              'bar': {'color': pred_color},
                              'threshold' : {'line': {'color': "red", 'width': 4}, 
                                             'thickness': 0.75, 
                                             'value': 0.7}
                            },
                    ))
    fig.update_layout(autosize=False,
                      width=600,
                      height=300,
                      margin=dict(l=40, r=40, b=0, t=0, pad=4),
                      paper_bgcolor="#F0F2F6",
                      )
    fig.update_yaxes(automargin=True)
    return fig

LOCALHOST_URI = 'http://127.0.0.1:5000'
REMOTEHOST_URI = 'https://home-credit-score.herokuapp.com'
HOST_URI = REMOTEHOST_URI
CLIENTS_URI = HOST_URI + '/client_ids'
CLIENT_DATA_URI = HOST_URI + '/client_data'
PREDICTION_URI = HOST_URI + '/prediction'
client_ids = load_client_ids(CLIENTS_URI)
initialize()

#Sidebar for choosing the Client
inputs = st.sidebar.text_input("Search the Client's ID", key='client_id_text')
with st.sidebar.form(key="client_id_select") :
     st.selectbox("Select the Client", options=[opt for opt in client_ids if str(opt).startswith(inputs)], key='client_id_selected')
     submit = st.form_submit_button("Submit", on_click=client_id_submitted)

#Columns for Client Number and Gauge of Risk Prediction
title_col1, title_col2= st.columns([1,2])
with title_col1 :
    st.write("## Client Number:", st.session_state.client_id)

# Get Information for a Client
if st.session_state.home == False:
    try :
        client_data = load_client_data(st.session_state.client_id, CLIENT_DATA_URI)
        prediction = get_client_prediction(client_data, PREDICTION_URI)
        with title_col2 :
            st.write('### Default Risk')
            st.plotly_chart(create_gauge(prediction), use_container_width=True)
    except :
        st.write("Something went wrong... try refreshing the page....")
    

    



