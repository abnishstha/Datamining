# -*- coding: utf-8 -*-
"""
Created on Sun May 19 02:15:15 2019

@author: admin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from apyori import apriori


def data_prep():
    # lets import the dataset
    df = pd.read_csv('dataset/online retail.csv', encoding = "ISO-8859-1", na_filter=False)
    
    # Cleaning up data
    #df.info()
    
    
    #1. Some of the GIFT DESCRIPTION has '?' and is errornous, then we can replace them to remove in later step
    df['Description'] = df['Description'].str.replace('\?','',case=False)
     
    
    #2. There are some "blank" value which should be converted to null
    df= df.replace('', np.nan)
    # Check tatal NANs in the dataset
    #df.isnull().sum() 
    
    print("Row # before dropping errorneous rows", len(df))
    #3. If customer ID is NAN, then the row is not useul, so remove those rows
    df.dropna(subset=['CustomerID','Description'],inplace=True)
    
    print("Row # after dropping errorneous rows", len(df))
    
    #4. Now lets convert the datatype for Customer ID
    df['CustomerID'] = df['CustomerID'].astype(int)
    
    return df

def convert_apriori_results_to_pandas_df(results):
    rules = []
    
    for rule_set in results:
        for rule in rule_set.ordered_statistics:
            # items_base = left side of rules, items_add = right side
            # support, confidence and lift for respective rules
            rules.append([','.join(rule.items_base), ','.join(rule.items_add),
                         rule_set.support, rule.confidence, rule.lift]) 
    
    # typecast it to pandas df
    return pd.DataFrame(rules, columns=['Left_side', 'Right_side', 'Support', 'Confidence', 'Lift']) 



def Association():

    df = data_prep()
    
    # group by customerID, then list all GIFT DESCRIPTION
    transactions = df.groupby(['CustomerID'])['Description'].apply(list)

    print('*********************************************************************************')
    print('Transactions')
    
    print(transactions.head(5))
    
    
    # type cast the transactions from pandas into normal list format and run apriori
    transaction_list = list(transactions)
    results = list(apriori(transaction_list, min_support=0.025))

    print('*********************************************************************************')
    print('1st 5 rules')    
    # print first 5 rules
    print(results[:5])
    
    
    result_df = convert_apriori_results_to_pandas_df(results)


    print('*********************************************************************************')
    print('After conversion')    
    print(result_df.head(20))
    
    
    print('*********************************************************************************')
    print('Result sorted by Lift')     
    # sort all acquired rules descending by lift
    result_df = result_df.sort_values(by='Lift', ascending=False)
    new_result = result_df.head(10)
    #pd.get_option('max_colwidth')
    #pd.set_option('max_colwidth',80)
    print(new_result.to_string())

    print('*********************************************************************************')
    print('Result sorted by confidence') 
    # sort all acquired rules descending by lift
    result_df = result_df.sort_values(by='Confidence', ascending=False)
    new_result = result_df.head(10)
    #pd.get_option('max_colwidth')
    #pd.set_option('max_colwidth',50)
    print(new_result.to_string())
    
    
    print('*********************************************************************************')
    print('Result sorted by support') 
    # sort all acquired rules descending by lift
    result_df = result_df.sort_values(by='Support', ascending=False)
    new_result = result_df.head(100)
    #pd.get_option('max_colwidth')
    #pd.set_option('max_colwidth',50)
    print(new_result.to_string())

    
    print('*********************************************************************************')
    print('Result containing HERB MARKER CHIVES')     
    # print all the result involving values HERB MARKER CHIVES
    result_df_search_left = result_df[result_df['Left_side'].str.contains('HERB MARKER CHIVES')]
    result_df_search_right = result_df[result_df['Right_side'].str.contains('HERB MARKER CHIVES')]
    #pd.get_option('max_colwidth')
    #pd.set_option('max_colwidth',50)
    print(result_df_search_left.to_string())
    print(result_df_search_right.to_string())
    #return result_df
       
if __name__ == '__main__':

    Association()

