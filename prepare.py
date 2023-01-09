# imports
import pandas as pd
import numpy as np
import os
from env import get_connection
import acquire


# import splitting and imputing functions
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


def prep_iris(df):
    '''
    This function takes in a dataframe and makes changes in a dataframe and return clean dataframe
    '''
    
    # drop unnecessary columns  
    df.drop(columns = ['species_id', 'measurement_id'], inplace=True)
    
    # rename column name 
    df.rename(columns={'species_name': 'species'}, inplace=True)
    
    # create dummy variable 
    dummies = pd.get_dummies(df['species'], drop_first=True)
    
    # concate dummy varibles with dataframe
    df = pd.concat([df, dummies], axis=1)
    
    # return dataframe
    return df


def prep_titanic(df):
    '''
    This function takes in a dataframe and makes changes in a dataframe and return clean dataframe
    '''
    
    # drop unnecessary columns  
    df.drop(columns = ['class', 'embarked','deck', 'passenger_id', 'age'], inplace=True )
    
    # fill Null values 
    df['embark_town'].fillna('Southampton', inplace=True)
   
    # create dummy variable 
    dummies = pd.get_dummies(df[['sex', 'embark_town']], drop_first=True)
    
    
    # concate dummy varibles with dataframe
    df = pd.concat([df, dummies],axis=1)
    
    # drop unnecessary columns  
    df.drop(columns = ['sex','embark_town'],inplace=True)
    # return dataframe
    return df


def prep_telco(df):
    '''
    This function takes in a dataframe and makes changes in a dataframe and return clean dataframe
    '''
    
    # create a list of columns names that will be dropped  
    to_drop = ['payment_type_id', 'contract_type_id', 'internet_service_type_id','customer_id' ]  
    
    # drop unnecessary columns 
    df.drop(columns = to_drop, inplace=True)
    
    # In column total_charges, replace white space with '0' and convert data type to float
    df.total_charges= df.total_charges.str.replace(' ','0').astype('float64')

    # # create a list of columns names to create dummies
    to_dummies = ['gender', 'partner', 'dependents', 'phone_service', 'multiple_lines', 'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies', 'paperless_billing', 'churn', 'internet_service_type', 'contract_type', 'payment_type']
    
    # create dummy variable
    dummies = pd.get_dummies(df[to_dummies],drop_first=True)
    
    # concate dummy varibles with dataframe
    df = pd.concat([df, dummies],axis=1)

    # return dataframe
    return df


def train_validate_test_split(df, target, seed=42):
    '''
    This function takes in a dataframe and return train, validate and test dataframe; stratify on target
    '''
    
    # split data into 80% train_validate, 20% test
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed, 
                                            stratify=df[target])
    
    # split train_validate data into 70% train, 30% validate
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed,
                                       stratify=train_validate[target])
    # return train, validate, test
    return train, validate, test