from threading import local
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots


#whether or not we are connecting to the local server or Heroku
local = False

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

def initialize_filter():
    st.session_state.filter_dict = {}
    i = 0
    for state_instance in st.session_state.keys():
        if state_instance.startswith('select') and (st.session_state[state_instance] == True):
            filter_list = state_instance.split('_')
            st.session_state.filter_dict[i] = {'type' : filter_list[1],
                                               'func' : filter_list[2],
                                               'value' : float(filter_list[3])}
            i += 1

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
def get_client_prediction(client_id, prediction_uri):
    headers = {"Content-Type": "application/json"}
    data_json = {'client_id': str(client_id)}
    response = requests.request(method='POST', headers=headers, url=prediction_uri, json=data_json)
    return float(response.json()["pred"])

@st.experimental_memo
def get_client_feature_importance(client_id, shap_uri):
    headers = {"Content-Type": "application/json"}
    data_json = {'client_id': str(client_id)}
    response = requests.request(method='POST', headers=headers, url=shap_uri, json=data_json)
    return pd.read_json(response.json()["SHAP_data"])

@st.cache
def load_feature_filter(feature_filter_uri):
    headers = {"Content-Type": "application/json"}
    data_json = {'filter_dict' : st.session_state.filter_dict,
                 'client_id' : str(st.session_state.client_id)}
    response = requests.request(method='POST', headers=headers, url=feature_filter_uri, json=data_json)
    return pd.read_json(response.json()["feature_filter"])

def load_feature_data(feature_name, feature_uri):
    headers = {"Content-Type": "application/json"}
    data_json = {'feature': feature_name}
    response = requests.request(method='POST', headers=headers, url=feature_uri, json=data_json)
    return pd.read_json(response.json()["feature_data"])

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

def create_feature_importance(client_data, n_features=15):
    data = client_data.iloc[:n_features, :].filter(['SHAP', 'Data'])
    rest_name = 'Following ' + str(len(client_data)-n_features) +' Features'
    rest = pd.DataFrame([[client_data['SHAP'].iloc[n_features:].sum(axis=0), np.nan]], index=[rest_name],
                        columns=['SHAP', 'Data'])
    data=pd.concat([data, rest]).iloc[::-1,:]
    fig = px.bar(data, x="SHAP", y=data.index, color="SHAP", text="Data",
                labels={"SHAP":'Weight', "index":'Features'},
                color_continuous_scale=px.colors.diverging.RdYlGn[::-1], range_color=[-0.5,0.5],)
    fig.update_layout(width=1000, height=n_features*30 + 100)
    fig.update_xaxes(range = [-0.5,0.5])
    return fig

def create_feature_comparison(feature_data, select, vline):
    # Group data together
    selected_data = (feature_data.mask(~select).dropna().values.ravel())
    feature_data = (feature_data.dropna().values.ravel())
    hist_data = [feature_data, selected_data]
    group_labels = ['All Clients', 'Filtered']
    colors = ['cornflowerblue', 'tomato']
    linecolors =['darkblue', 'darkred']

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[1,4], vertical_spacing=0.1)
    for x, label, color, linecolor in zip(hist_data, group_labels, colors, linecolors):
        fig.add_trace(go.Box(x=x, name=label, legendgroup=label, fillcolor=color, line={'color':linecolor},
                             boxmean=True, showlegend=False), row=1, col=1)
        fig.add_trace(go.Histogram(x=x, name=label, legendgroup=label, marker={'color':color}, ), 
                      row=2, col=1)
    fig.add_vline(vline)
    return fig

LOCALHOST_URI = 'http://127.0.0.1:5000'
REMOTEHOST_URI = 'https://home-credit-score-dev.herokuapp.com'
if local :
    HOST_URI = LOCALHOST_URI
else :
    HOST_URI = REMOTEHOST_URI
CLIENTS_URI = HOST_URI + '/client_ids'
CLIENT_DATA_URI = HOST_URI + '/client_data'
PREDICTION_URI = HOST_URI + '/prediction'
SHAP_URI = HOST_URI + '/shapvalues'
FEATURE_URI = HOST_URI + '/features'
FEATURE_FILTER_URI = HOST_URI + '/filter'


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
        # Show client default risk
        #client_data = load_client_data(st.session_state.client_id, CLIENT_DATA_URI)
        prediction = get_client_prediction(st.session_state.client_id, PREDICTION_URI)
        with title_col2 :
            st.write('### Default Risk')
            st.plotly_chart(create_gauge(prediction), use_container_width=True)

        #Show feature importance
        shap_data = get_client_feature_importance(st.session_state.client_id, SHAP_URI)
        st.write('### Feature Importance')
        st.slider('Number of Top Features to Display', min_value=5, max_value=30, value=15, step=1, 
                  key='n_features', disabled=False)
        st.plotly_chart(create_feature_importance(shap_data, int(st.session_state.n_features)), 
                        use_container_width=True)

        #Show feature comparisons
        st.write('### Feature Comparisions among Clients')
        #Add a filter
        st.write('#### Filter Options')
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        with filter_col1:
            st.write('Gender')
            client_gender = load_feature_data('APPL_CODE_GENDER', FEATURE_URI).copy(
                            ).client_gender.iloc[client_ids.index(int(st.session_state.client_id))].values[0]
            st.checkbox('1', value=(client_gender==1), key='select_gender_eq_1')
            st.checkbox('0', value=(client_gender==0), key='select_gender_eq_0')
        with filter_col2:
            st.write('Defaulted')
            st.checkbox('Yes', value=(prediction >= 0.5), key='select_defaulted_eq_1')
            st.checkbox('No', value=(prediction < 0.5), key='select_defaulted_eq_0')
        with filter_col3:
            st.write('Income Range')
            st.checkbox("+/- 10% of client's", value=False, key='select_income_range_10')
            st.checkbox("+/- 20% of client's", value=False, key='select_income_range_20')
        initialize_filter()

        select = load_feature_filter(FEATURE_FILTER_URI).copy()
        #show the features
        for i in range(3):
            num=i+1
            feature_key = 'feature'+str(num)+'_selected'
            st.write('#### Feature #' + str(num))
            st.selectbox("Select Feature #"+str(num), options=shap_data.index.values[:50], index=i, key=feature_key)
            feature = st.session_state[feature_key]
            feature_data = load_feature_data(feature, FEATURE_URI).copy()
            client_value = feature_data.iloc[client_ids.index(int(st.session_state.client_id))].values[0]
            st.plotly_chart(create_feature_comparison(feature_data, select, client_value))
    except :
        st.write("Something went wrong... try refreshing the page....")
    

    



