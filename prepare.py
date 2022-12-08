# imports
import pandas as pd
import numpy as np
import os
from pydataset import data
from env import get_connection
import acquire


# import splitting and imputing functions
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer



# function takes original data and return cleaned data
def prep_iris(df):
    df.drop(columns = ['species_id', 'measurement_id'], inplace=True)
    df.rename(columns={'species_name': 'species'}, inplace=True)
    dummies = pd.get_dummies(df['species'], drop_first=True)
    df = pd.concat([df, dummies], axis=1)
    return df


# function takes original data and return cleaned data
def prep_titanic(df):
    
    df.drop(columns = ['class', 'embarked','deck', 'passenger_id', 'age'], inplace=True )
    
    df['embark_town'].fillna('Southampton', inplace=True)
    
    dummies = pd.get_dummies(df[['sex', 'embark_town']], drop_first=True)
    
    df = pd.concat([df, dummies],axis=1)

    return df


# function takes original data and return cleaned data
def prep_telco(df):
    to_drop = ['payment_type_id', 'contract_type_id', 'internet_service_type_id','customer_id' ]   
    df.drop(columns = to_drop, inplace=True)

    to_dummies = ['gender', 'partner', 'dependents', 'phone_service', 'multiple_lines', 'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies', 'paperless_billing', 'churn', 'internet_service_type', 'contract_type', 'payment_type']
    dummies = pd.get_dummies(df[to_dummies],drop_first=True)
    
    df = pd.concat([df, dummies],axis=1)

    return df


# function takes cleaned data and return train data, validate data and test data
def train_validate_test_split(df, target, seed=42):
    
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed, 
                                            stratify=df[target])
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed,
                                       stratify=train_validate[target])
    return train, validate, test