# import
import pandas as pd
import numpy as np
import os
from pydataset import data
from env import get_connection
import acquire


# import splitting and imputing functions
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer



# function for iris to  get clean data
def prep_iris(df):
    df.drop(columns = ['species_id', 'measurement_id'], inplace=True)
    df.rename(columns={'species_name': 'species'}, inplace=True)
    dummies = pd.get_dummies(df['species'], drop_first=True)
    df = pd.concat([df, dummies], axis=1)
    return df


# function for iris train, val, test
def iris_train_val_test(iris_df):
    seed = 42
    
    train_iris, test_iris = train_test_split(iris_df, train_size=0.7, random_state=seed, stratify=iris_df['versicolor'])

    train_iris, val_test_iris = train_test_split(iris_df, train_size=0.7, random_state=seed, stratify=iris_df['versicolor'])
   
    validate_iris, test_iris = train_test_split(val_test_iris, train_size=0.5, random_state=seed, stratify=val_test_iris['versicolor'])

    
    return train_iris, validate_iris, test_iris




# function for titanic to  get clean data
def prep_titanic(df):
    
    df.drop(columns = ['class', 'embarked','deck', 'passenger_id', 'age'], inplace=True )
    
    df['embark_town'].fillna('Southampton', inplace=True)
    
    dummies = pd.get_dummies(df[['sex', 'embark_town']], drop_first=True)
    
    df = pd.concat([df, dummies],axis=1)

    return df


# function for titanic train, val, test
def titanic_train_val_test(titanic_df):
    seed = 42
    
    train_titanic, test_titanic = train_test_split(titanic_df, train_size=0.7, random_state=seed, stratify=titanic_df['survived'])

    train_titanic, val_test_titanic = train_test_split(titanic_df, train_size=0.7, random_state=seed, stratify=titanic_df['survived'])

    validate_titanic, test_titanic = train_test_split(val_test_titanic, train_size=0.5, random_state=seed, stratify=val_test_titanic['survived'])

    
    return train_titanic, validate_titanic, test_titanic




# function for telco to  get clean data
def prep_telco(df):
    to_drop = ['payment_type_id', 'contract_type_id', 'internet_service_type_id','customer_id' ]   
    df.drop(columns = to_drop, inplace=True)

    to_dummies = ['gender', 'partner', 'dependents', 'phone_service', 'multiple_lines', 'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies', 'paperless_billing', 'churn', 'internet_service_type', 'contract_type', 'payment_type']
    dummies = pd.get_dummies(df[to_dummies],drop_first=True)
    
    df = pd.concat([df, dummies],axis=1)

    return df


# function for telco train, val, test
def telco_train_val_test(telco_df):
    seed = 42
    
    train_telco, test_telco = train_test_split(telco_df, train_size=0.5, random_state=seed, stratify=telco_df['churn'])

    train_telco, val_test_telco = train_test_split(telco_df, train_size=0.7, random_state=seed, stratify=telco_df['churn'])

    validate_telco, test_telco = train_test_split(val_test_telco, train_size=0.5, random_state=seed, stratify=val_test_telco['churn'])
    
    return train_telco, validate_telco, test_telco