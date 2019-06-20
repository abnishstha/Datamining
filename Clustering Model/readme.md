The “online_shoppers_intention” data set contains 12,330 sessions. The dataset was formed so that each session would belong to a different user in a 1-year period. This was done to avoid any tendency to a specific campaign, special day, user profile, or period. The dataset consists of 10 numerical and 8 categorical attributes.

The company wants to learn the user characteristics in terms of users’ time spent on the website. We are helping the company to understand those characteristics by profiling the customers using the k-means analysis.

<br /><br />
Q1. k-means analysis for numerical attributes.
Build a clustering model to profile the customers based on the time they spend on the website.

Code: clustering_model.py

Finding Optimal K <br /> <br />elbow_method: <br /> 

![ScreenShot](https://github.com/abnishstha/Datamining/blob/master/Clustering%20Model/diags/1_elbow_method.png) <br />


Silhouette score: <br />

KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,<br />
    n_clusters=4, n_init=10, n_jobs=10, precompute_distances='auto',<br />
    random_state=42, tol=0.0001, verbose=0)<br />
Silhouette score for k=4 0.6439386087119756<br /><br />
KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,<br />
    n_clusters=6, n_init=10, n_jobs=10, precompute_distances='auto',<br />
    random_state=42, tol=0.0001, verbose=0)<br />
Silhouette score for k=6 0.5597365643208325<br />


Result for K=4
![ScreenShot](https://github.com/abnishstha/Datamining/blob/master/Clustering%20Model/diags/1_result.png)

<br /><br />
Q2. k-prototypes analysis for numerical+catagorical attributes.

Add more information such as where the users come from and when they access the website, to the clustering analysis that you have conducted in the previous task.
<br />
<br /><br />Code: clustering_model.py
<br />
<br /><br />Finding Optimal K<br /><br />

elbow_method:<br />


![ScreenShot](https://github.com/abnishstha/Datamining/blob/master/Clustering%20Model/diags/2_elbow_method.png) <br />

￼
Silhouette score: <br />


KPrototypes(cat_dissim=<function matching_dissim at 0x000001FFE7073E18>,<br />
      gamma=0.5, init='Huang', max_iter=100, n_clusters=4, n_init=5,<br />
      n_jobs=1,<br />
      num_dissim=<function euclidean_dissim at 0x000001FFE7073EA0>,<br />
      random_state=None, verbose=0)<br />
Silhouette score for k=4 0.3167903379194475<br /><br />
KPrototypes(cat_dissim=<function matching_dissim at 0x000001FFE7073E18>,<br />
      gamma=0.5, init='Huang', max_iter=100, n_clusters=6, n_init=5,<br />
      n_jobs=1,<br />
      num_dissim=<function euclidean_dissim at 0x000001FFE7073EA0>,<br />
      random_state=None, verbose=0)<br />
Silhouette score for k=6 0.3349575101728609<br />
<br /><br />

Result for K=6<br /><br />


![ScreenShot](https://github.com/abnishstha/Datamining/blob/master/Clustering%20Model/diags/2_result.png)<br />
