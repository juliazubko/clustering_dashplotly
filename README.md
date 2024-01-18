# clustering_dashplotly
### Clusters the data and makes a dashboard with some basic plots 
### [UMAP, DBSCAN,  Agglomerative, Dash,  Plotly] 
(Year 1 Data Science(HVE course) school aasignment 2022))

![main](https://user-images.githubusercontent.com/102211232/165947985-082f0488-a410-4d05-bc47-29d24bb83fde.png)

- Takes in pre-processed data (no NaNs, encoded);

 -  Scales the data;
   
 -  Makes 2D UMAP embedding;
   
  - Performs DBSCAN and AgglomerativeClusterer hyperparameter tuning
   (for-loops);
   
  - Runs DBSCAN and AgglomerativeClusterer on the data, appends obtained cluster labels to the original
   dataframe;
   
 - Plots the results (basic Dash Plotly dashboard)
   > - 3 exploratory scatterplots (UMAP data embedding, Dbscan clustering results on the embedding, 
   Agglomerative clustering results on the embedding) with some clustering evaluation metrics displayed
   (Silhouette, Davis-Bouldin, Calinski-Harabasz)
   > - 2 callback-changeable plots: bar- and donut chart (displays feature distribution per 
   chosen algorithm,  per chosen cluster) 
   
  ![Namnlös](https://user-images.githubusercontent.com/102211232/168421953-11a07df7-098a-4e97-a83f-9bda3b1ba6e3.png)
   
   
   
