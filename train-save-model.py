#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score


from tqdm.auto import tqdm

df = pd.read_csv("Telco-Customer-Churn.csv")
df.head(2)

# check for missing values and the number of unique values in df
for col in df.columns:
    print('*' * 80)
    print(f'"{col.upper()}" Data Types, No. of Missing Values and Unique Values')
    print('*' * 80)
    print(f"data type: {df[col].dtype}")
    print(f"no. missing values: {df[col].isna().sum()}")
    print(f"no. of unique values: {df[col].nunique()}")
    
    print()


# Clean Data
def clean_values(value):
    """
    value (str):
        a row text value or column name
    Return (str):
        a lowercase text with spaces replaced with _ and all leading and training spaces removed
    """
    dummy = value.strip().lower().replace(' ', '_')
    return dummy


# convert all columns to lower case
columns = list(map(clean_values, df.columns))
df.columns = columns

# Get Numerical and Categorical Columns
# total charges is suppose to be an integer from examining the dataframe but it's bject
numerical = list(df.dtypes[df.dtypes != 'object'].index) + ['totalcharges']

# while senior citizen should be a cetegorical feature
numerical.remove('seniorcitizen')

# clean all non-numerical column values 
categorical = list(df.dtypes[df.dtypes == 'object'].index) 

# total charges should be a numerical column
for col in ['totalcharges', 'churn', 'customerid']:
    categorical.remove(col)

# convert all categorical column values to lower case
df = df.apply(lambda x: x.str.lower() if x.name in categorical else x)

# add senior citizen column to categorical
categorical.append('seniorcitizen')

# Change the type of totalcharges and churn column
# total charges is an object type but holds numerical values: convert it
df['totalcharges'] = pd.to_numeric(df['totalcharges'], errors='coerce')
# values that cannot be converted to integer are replaced with NaN: fill the missing values with zeroes
df['totalcharges'] = df['totalcharges'].fillna(0)

# change the type of churn column
# let yes be 1
df['churn'] = (df['churn'] == 'Yes').astype('int')

# Train, Validation and Test Dataset
# Split the dataframe into full_train, train, validation and test dataset
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

# Reset the index values
df_full_train.reset_index(drop=True, inplace = True)

# All the target variables from the train, test and validation set
y_full_train = df_full_train['churn']
# y_train = df_train['churn']
# y_val = df_val['churn']
y_test = df_test['churn']


# Training Function
def train(df_train, y_train, c=1.0):
    """
    df_train (DataFrame):
        The training feature exam
    y_train (Series):
        The training target variables
    
    Return:
        dv (DictVectorizer): fitted one-hot encoding object
        model (LogisticRegresion): trained logistic regression model
    """
    # convert the data frame to list of dictionary
    # each dicitionary is a key value pair of a particular record/observation 
    dict_df = df_train[categorical + numerical].to_dict(orient='records') 
    
    # one-hot encode the categorical variable
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dict_df)
    
    model = LogisticRegression(C=c, max_iter=1000)
    model.fit(X_train, y_train)
    
    return dv, model


# Predict Function
def predict(dv, model, df):
    """
    dv: 
        fitted one-hot encoding object
    model: 
        trained logistic regression model
    df (DataGFrame):
        Test dataframe 
        
    Return (series):
        predicted values of the test dataset
    """
    # convert the data frame to dictionary 
    dict_df = df[categorical + numerical].to_dict(orient='records') 
    
    # one-hot encode the categorical variable
    X_test= dv.transform(dict_df)
    
    # predict the test data
    y_pred = model.predict_proba(X_test)[:, 1]
    
    return y_pred


# Validation
C = [0.001, 0.01, 0.1, 0.5, 1, 5, 10] 

for c in tqdm(C):
    scores = []
    
    n_split = 5
    
    # this is a generator
    kfolds = KFold(n_splits=n_split, random_state=1, shuffle=True)
    
    for train_idx, val_idx in tqdm(kfolds.split(df_full_train)):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]

        y_train = df_train.churn.values
        y_val = df_val.churn.values

        dv, model = train(df_train, y_train, c)
        pred_val = predict(dv, model, df_val)

        auc = roc_auc_score(y_val, pred_val)
        scores.append(auc)
        
    print(f"C <=> {c}: {round(np.array(scores).mean(), 3)} +/- {round(np.array(scores).std(), 3)}")


# Final Test Prediction
c=0.1
dv, model = train(df_full_train, y_full_train, c)
pred_test = predict(dv, model, df_test)
auc = roc_auc_score(y_test, pred_test)

# Saving the Model
import pickle

file_name = f'model_c={c}.bin'
with open(file_name, 'wb') as f:
    pickle.dump((dv, model), f)








