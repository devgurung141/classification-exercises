# imports

import pandas as pd
import numpy as np
import os
from env import get_connection


def get_titanic_data():
    '''
    This function reads in titanic data from Codeup database using sql querry into a df, 
    writes data to a csv file if a local file doesn't exist, and return df and if a local file exists
    return return df
    '''
    
    filename = "titanic.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        # read the SQL query into a dataframe
        df = pd.read_sql('SELECT * FROM passengers', get_connection('titanic_db'))

        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename, index = False)

        # Return the dataframe to the calling code
        return df 


querry_iris = """
        SELECT * FROM species
        JOIN measurements USING(species_id);
        """

def get_iris_data():
    '''
    This function reads in iris data from Codeup database using sql querry into a df, 
    writes data to a csv file if a local file doesn't exist, and return df and if a local file exists
    return return df
    '''
    
    filename = "iris.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        # read the SQL query into a dataframe
        df = pd.read_sql(querry_iris, get_connection('iris_db'))

        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename, index=False)

        # Return the dataframe to the calling code
        return df  

    
querry_telco = """
    SELECT * 
    FROM customers 
    LEFT JOIN internet_service_types USING(internet_service_type_id)
    JOIN contract_types USING (contract_type_id)
    JOIN payment_types USING (payment_type_id);
    """


def get_telco_data():
    '''
    This function reads in telco data from Codeup database using sql querry into a df, 
    writes data to a csv file if a local file doesn't exist, and return df and if a local file exists
    return return df
    '''
    
    filename = "telco.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        # read the SQL query into a dataframe
        df = pd.read_sql(querry_telco, get_connection('telco_churn'))

        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename, index=False)

        # Return the dataframe to the calling code
        return df 