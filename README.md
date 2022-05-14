# clustering_dashplotly_oop
### Clusters the data and makes a dashboard with some basic exploratory plots 
### [UMAP, DBSCAN,  Agglomerative, Dash,  Plotly] 
### *school assignment*

![main](https://user-images.githubusercontent.com/102211232/165947985-082f0488-a410-4d05-bc47-29d24bb83fde.png)

- Takes in pre-processed data (no NaNs, encoded);

 -  Scales the data;
   
 -  Makes 2D UMAP embedding;
   
  - Performs DBSCAN and AgglomerativeClusterer hyperparameter tuning
   (for-loops, => suitable for rather small datasets);
   
  - Runs clusterers on the data, appends obtained cluster labels to the original
   dataframe;
   
- Puts everything together into a simple/ pretty basic/ Dash Plotly dashboard 
   > - 3 exploratory scatterplots (UMAP data embedding, Dbscan results on the embedding, 
   Agglomerative results on the embedding) with some clustering evaluation metrics displayed
   (Silhouette, Davis-Bouldin, Calinski-Harabasz)
   > - 2 callback-changeable plots: bar- and donut chart (displays feature distribution per 
   chosen algorithm,  per chosen cluster) 
   
   ![results](https://user-images.githubusercontent.com/102211232/165948191-928ee8fc-e7d0-4c08-b906-856bca43e53e.png) 
   
   
   
