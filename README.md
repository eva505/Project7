 # Home Credit Scoring API

A home credit scoring API, which returns the probability of a client defaulting (or not) based on the client's application and other data that has been collected by the credit lender.

## How-to-Launch-App

### To access the Dashboard :

Open the following link in a webbrowser for the newest version : 
https://share.streamlit.io/eva505/project7/dev2/Dashboard/dashboard.py

Version 0.2.0
https://share.streamlit.io/eva505/project7/dev0/Dashboard/dashboard.py

### To access the API:

https://home-credit-score-dev2.herokuapp.com/

For a specific client go to https://home-credit-score-dev2.herokuapp.com/prediction/client_id

The client_id has to be an integer.

Examples for valid a client_id that are in the data base : 
* 241571 - a client that did not default
* 429767 - a client that did default


## Description

The prediction of whether a client will default or not is made with a LightBGM model (https://lightgbm.readthedocs.io/en/latest/) and the feature importances are determined through Shapely values (SHAP).

## GitHubLinks

https://github.com/eva505/Project7/tree/dev2/Data

https://github.com/eva505/Project7/tree/main/Data

## Authors

Eva Bookjans

## Version History
* 0.3
    * 0.3.1 Prediction for Client available through API
    * Implemented LGBM model
* 0.2
    * 0.2.0 Updated README
    * Added "Added 'Feature Comparison' Functionality
* 0.1
    * 0.1.1 working API
    * Added "Feature Importance" Functionality
* 0.0
    * 0.0.1 updated model in preparation for version 0.1
    * Initial Release

## Original Data from the Home Credit Group

* [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/data)

## Acknowledgments

* [WILL KOEHRSEN](https://www.kaggle.com/code/willkoehrsen/start-here-a-gentle-introduction/notebook)
* [AGUIAR](https://www.kaggle.com/code/jsaguiar/lightgbm-with-simple-features/script)