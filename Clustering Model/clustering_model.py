# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 09:19:30 2019

@author: admin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from kmodes.kprototypes import KPrototypes
from sklearn.metrics import silhouette_score
from kmodes.kmodes import KModes

def data_prep():
    
    # lets import the dataset
    df = pd.read_csv('dataset/online_shoppers_intention.csv', na_filter=False)


    # Cleaning up data

    #df.info()

    # 1. There are some "blank" value which should be converted to null
    df= df.replace('', np.nan)

    df.isnull().sum()

    #2. Change the datatype of these variables from object to float
    df['Administrative'] = df['Administrative'].astype(float)
    df['Administrative_Duration'] = df['Administrative_Duration'].astype(float)
    df['Informational'] = df['Informational'].astype(float)
    df['Informational_Duration'] = df['Informational_Duration'].astype(float)
    df['ProductRelated'] = df['ProductRelated'].astype(float)
    df['ProductRelated_Duration'] = df['ProductRelated_Duration'].astype(float)
    df['Region'] = df['Region'].astype(float)
    df['Weekend'] = df['Weekend'].astype(float)


    # Duration or no of visit in the page can't be negative, we can replace them with NAN
    np.sum((df['Administrative'] < 0).values)
    np.sum((df['Administrative_Duration'] < 0).values)
    np.sum((df['Informational'] < 0).values)
    np.sum((df['Informational_Duration'] < 0).values)
    np.sum((df['ProductRelated'] < 0).values)
    np.sum((df['ProductRelated_Duration'] < 0).values)

    mask = df['Administrative_Duration'] < 0
    df.loc[mask, 'Administrative_Duration'] = np.nan
    
    mask = df['Informational_Duration'] < 0
    df.loc[mask, 'Informational_Duration'] = np.nan
    
    mask = df['ProductRelated_Duration'] < 0
    df.loc[mask, 'ProductRelated_Duration'] = np.nan

    mask = df['ProductRelated_Duration'] < 0
    df.loc[mask, 'ProductRelated_Duration'] = np.nan  
    
    mask = df['Region'] < 0
    df.loc[mask, 'Region'] = np.nan

    mask = df['SpecialDay'] < 0
    df.loc[mask, 'SpecialDay'] = np.nan   
    
 
    # Duration of sum of all  the three pages should be at least 1 or else can be considered error record.
    # Now Lets remove all the row where sum of duration for all the three page is <=1
    # before
    print("Row # before dropping errorneous rows", len(df))
    df = df[(df['ProductRelated_Duration']+df['Administrative_Duration']+df['Informational_Duration'])>= 1]
    # after
    print("Row # arter dropping errorneous rows", len(df))
          
    return df
      

def Kmeans_task2():

    df = data_prep()
    
    # take 3 variables and drop the rest
    df2 = df[['Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration']]
    
    # convert df2 to matrix
    X = df2.as_matrix()
    
    # scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # random state, we will use 42 instead of 10 for a change
    rs = 42

   
    print('*********************************************************************************')
    print('Finding Optimal K')        
    
    # list to save the clusters and cost
    clusters = []
    inertia_vals = []
    
    # this whole process should take a while
    for k in range(2, 15, 2):
        # train clustering with the specified K
        model = KMeans(n_clusters=k, random_state=rs, n_jobs=10)
        model.fit(X)
        
        # append model to cluster list
        clusters.append(model)
        inertia_vals.append(model.inertia_)
    
    
    # plot the inertia vs K values
    plt.plot(range(2,15,2), inertia_vals, marker='*')
    plt.show()

    from sklearn.metrics import silhouette_score

    print(clusters[1])
    print("Silhouette score for k=4", silhouette_score(X, clusters[1].predict(X)))
    
    print(clusters[2])
    print("Silhouette score for k=6", silhouette_score(X, clusters[2].predict(X)))        
    

    print('*********************************************************************************')
    print('Clustering for K=4')
 

    # visualisation of K=4 clustering solution
    model = KMeans(n_clusters=4, random_state=rs).fit(X)
   
    y = model.predict(X)
    df2['Cluster_ID'] = y
    
    # sum of intra-cluster distances
    print("Sum of intra-cluster distance:", model.inertia_)
    
    print("Centroid locations:")
    for centroid in model.cluster_centers_:
        print(centroid)
    
    # how many in each
    print("Cluster membership")
    print(df2['Cluster_ID'].value_counts())
    
    # pairplot
    # added alpha value to assist with overlapping points
    cluster_g = sns.pairplot(df2, hue='Cluster_ID', plot_kws={'alpha': 0.5})
    plt.show()
    

    print('*********************************************************************************')
    print('visualise variable distribution')

    
    # prepare the column and bin size. Increase bin size to be more specific, but 20 is more than enough
    cols = ['Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration']
    n_bins = 20

    # inspecting cluster 0 and 1
    clusters_to_inspect = [0,1]

    for cluster in clusters_to_inspect:
        # inspecting cluster 0
        print("Distribution for cluster {}".format(cluster))

        # create subplots
        fig, ax = plt.subplots(nrows=3)
        ax[0].set_title("Cluster {}".format(cluster))

        for j, col in enumerate(cols):
            # create the bins
            bins = np.linspace(min(df2[col]), max(df2[col]), 20)
            # plot distribution of the cluster using histogram
            sns.distplot(df2[df2['Cluster_ID'] == cluster][col], bins=bins, ax=ax[j], norm_hist=True)
            # plot the normal distribution with a black line
            sns.distplot(df2[col], bins=bins, ax=ax[j], hist=False, color="k")

        plt.tight_layout()
        plt.show()


def Kmeans_task3():  
    
    df = data_prep()
    df2 = df[['Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration','Region','SpecialDay','Month']]
    
    # mapping
    Month_map = {'Feb':2, 'Mar': 3, 'May': 5, 'June': 6, 'Jul': 7, 'Aug': 8, 'Sep': 7, 'Oct': 8, 'Nov': 9, 'Dec': 10}
    df2['Month'] = df2['Month'].map(Month_map)
    # convert df to matrix
    X = df2.as_matrix()
    
    # scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)


    print('*********************************************************************************')
    print('Finding Optimal K')
    
    # list to save the clusters and cost
    clusters = []
    inertia_vals = []
    
    # this whole process should take a while
    for k in range(2, 10, 2):
        # train clustering with the specified K
        model = KPrototypes(n_clusters=k, init='Huang', n_init=5, verbose=0) 
        model.fit_predict(X, categorical=[5])
        
        # append model to cluster list
        clusters.append(model)
        inertia_vals.append(model.cost_)
        
        print('done with: ', k)
    
    # plot the inertia vs K values
    plt.plot(range(2,10,2), inertia_vals, marker='*')
    plt.show()
    
    print(clusters[1])
    print("Silhouette score for k=4", silhouette_score(X, clusters[1].fit_predict(X, categorical=[5])))
    
    print(clusters[2])
    print("Silhouette score for k=6", silhouette_score(X, clusters[2].fit_predict(X, categorical=[5])))   
       
 
    print('*********************************************************************************')
    print('Clustering for K=6')
 

    # visualisation of K=4 clustering solution
    model = KPrototypes(n_clusters=6, init='Huang', n_init=5, verbose=0) 
    model.fit_predict(X, categorical=[5])
    '''
    # sum of intra-cluster distances
    print("Sum of intra-cluster distance:", model.inertia_)
    
    print("Centroid locations:")
    for centroid in model.cluster_centers_:
        print(centroid)
    '''
    y = model.predict(X, categorical=[5])
    df2['Cluster_ID'] = y
    
    # how many in each
    print("Cluster membership")
    print(df2['Cluster_ID'].value_counts())
    
    # pairplot
    # added alpha value to assist with overlapping points
    cluster_g = sns.pairplot(df2, hue='Cluster_ID', plot_kws={'alpha': 0.5})
    plt.show()
    

    print('*********************************************************************************')
    print('visualise variable distribution')

    
    # prepare the column and bin size. Increase bin size to be more specific, but 20 is more than enough
    cols = ['Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration','Region','SpecialDay','Month']
    n_bins = 20

    # inspecting cluster 0 and 1
    clusters_to_inspect = [0,1]

    for cluster in clusters_to_inspect:
        # inspecting cluster 0
        print("Distribution for cluster {}".format(cluster))

        # create subplots
        fig, ax = plt.subplots(nrows=6)
        ax[0].set_title("Cluster {}".format(cluster))

        for j, col in enumerate(cols):
            # create the bins
            bins = np.linspace(min(df2[col]), max(df2[col]), 20)
            # plot distribution of the cluster using histogram
            sns.distplot(df2[df2['Cluster_ID'] == cluster][col], bins=bins, ax=ax[j], norm_hist=True)
            # plot the normal distribution with a black line
            sns.distplot(df2[col], bins=bins, ax=ax[j], hist=False, color="k")

        plt.tight_layout()
        plt.show()   

if __name__ == '__main__':

    Kmeans_task2()
    Kmeans_task3()