 # Home Credit Scoring API

A home credit scoring API, which returns the probability of a client defaulting (or not) based on the client's application and other data that has been collected by the credit lender.

## How-to-Launch-App

Open the following link in a webbrowser for the newest version : 
https://share.streamlit.io/eva505/project7/dev2/Dashboard/dashboard.py

Version 0.2.0
https://share.streamlit.io/eva505/project7/dev0/Dashboard/dashboard.py

Version 0.1.1:
https://share.streamlit.io/eva505/project7/main/Dashboard/dashboard.py

## Description

The prediction of whether a client will default or not is made with a LightBGM model (https://lightgbm.readthedocs.io/en/latest/) and the feature importances are determined through Shapely values (SHAP).

## GitHubLink to the most current developments

https://github.com/eva505/Project7/tree/dev2/Data

## Authors

Eva Bookjans

## Version History
* 0.3
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