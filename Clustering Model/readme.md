The “online_shoppers_intention” data set contains 12,330 sessions. The dataset was formed so that each session would belong to a different user in a 1-year period. This was done to avoid any tendency to a specific campaign, special day, user profile, or period. The dataset consists of 10 numerical and 8 categorical attributes.

The company wants to learn the user characteristics in terms of users’ time spent on the website. We are helping the company to understand those characteristics by profiling the customers using the k-means analysis.

1. k-means analysis for numerical attributes.
Build a clustering model to profile the customers based on the time they spend on the website.

Code: clustering_model.py

Finding Optimal K

elbow_method:
![ScreenShot](https://github.com/abnishstha/Datamining/blob/master/Clustering%20Model/diags/1_elbow_method.png)

￼
Silhouette score: 
KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
    n_clusters=4, n_init=10, n_jobs=10, precompute_distances='auto',
    random_state=42, tol=0.0001, verbose=0)
Silhouette score for k=4 0.6439386087119756
KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
    n_clusters=6, n_init=10, n_jobs=10, precompute_distances='auto',
    random_state=42, tol=0.0001, verbose=0)
Silhouette score for k=6 0.5597365643208325


Result for K=4
![ScreenShot](https://github.com/abnishstha/Datamining/blob/master/Clustering%20Model/diags/1_result.png)


2. k-prototypes analysis for numerical+catagorical attributes.

Add more information such as where the users come from and when they access the website, to the clustering analysis that you have conducted in the previous task.

Code: clustering_model.py

Finding Optimal K

elbow_method:
![ScreenShot](https://github.com/abnishstha/Datamining/blob/master/Clustering%20Model/diags/2_elbow_method.png)

￼
Silhouette score: 
KPrototypes(cat_dissim=<function matching_dissim at 0x000001FFE7073E18>,
      gamma=0.5, init='Huang', max_iter=100, n_clusters=4, n_init=5,
      n_jobs=1,
      num_dissim=<function euclidean_dissim at 0x000001FFE7073EA0>,
      random_state=None, verbose=0)
Silhouette score for k=4 0.3167903379194475
KPrototypes(cat_dissim=<function matching_dissim at 0x000001FFE7073E18>,
      gamma=0.5, init='Huang', max_iter=100, n_clusters=6, n_init=5,
      n_jobs=1,
      num_dissim=<function euclidean_dissim at 0x000001FFE7073EA0>,
      random_state=None, verbose=0)
Silhouette score for k=6 0.3349575101728609


Result for K=6
![ScreenShot](https://github.com/abnishstha/Datamining/blob/master/Clustering%20Model/diags/2_result.png)
