import warnings

warnings.simplefilter(action='ignore')
import pandas as pd
from sklearn import metrics
import numpy as np

from sklearn.preprocessing import MinMaxScaler
import umap.umap_ as umap
from sklearn.cluster import DBSCAN, AgglomerativeClustering

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
import plotly.express as px

template = load_figure_template('cerulean')


class GetDashboard:

    def __init__(self, path):
        self.df = pd.read_csv(path, usecols=lambda x: x not in ['Id', 'ID'])
        self.df1 = self.df.copy()
        self.df1 = pd.DataFrame(MinMaxScaler().fit_transform(self.df1),
                                columns=self.df1.columns)

        # first run all the dim.reduction and clustering tasks, find the methods below.
        self._umap_embedding()
        self._dbscan_umap()
        self._agglomerative_umap()

        # then start put it all together into a dashboard
        # here, explicitly choose dashboard bootstrap component theme to automatically get
        # some default CSS elements to facilitate visual dashboard formatting
        self.new_app = dash.Dash(external_stylesheets=[dbc.themes.CERULEAN])

        # dashboard will be built according to the following plan:
        # >> ROW 1: 3 plots (UMAP 2D embedding, DBSCAN and Agglomerative clusters visualisation on the
        #           UMAP embedding)
        # >> CONTROLS: one radioItem per model/algorithm (DBSCAN or Agglomerative),
        #              dropDown menu to choose individual cluster
        # >> ROW 2: bar- and pie chart per chosen model+cluster number combination,
        #           dynamically changeable on the callback.
        #
        # will use 2 callback functions: 1 -- to change dropdown menu accordingly to
        # chosen radioItem (algorithm), as number of clusters may vary för each clustering method;
        # 2 -- to update bar & piechart accordingly to the chosen model+cluster number combination;
        

        # first row and its components (contains 3 columns / containers for one individual graph and styling)
        content_first_row = dbc.Row(
            [
                dbc.Col([
                    html.H6('Projection of data with number dimensions '
                            'reduced to 2 (UMAP)'),
                    html.Br(), html.Br(), html.Br(), html.Br(),

                    # UMAP 2D data embedding
                    dcc.Graph(id='umap_embedding',
                              figure=px.scatter(self.embedding,
                                                x=self.embedding[:, 0],
                                                y=self.embedding[:, 1],
                                                template=template,
                                                ).update_traces(marker=dict(color='rgb(50,131,254)',
                                                                            size=15,
                                                                            opacity=0.5))),
                ], width=True),

                # DBSCAN clustering results on UMAP 2D embedding >> plot + display some
                # performance evaluation metrics
                dbc.Col([
                    html.H6('DBSCAN on data with number dimensions reduced to 2 (UMAP)'),
                    html.Div(f'''Nuber of clusters: {len(set(self.dblabels))}'''),
                    html.Div(
                        f'''Silhouette score: 
                                {round(metrics.silhouette_score(self.embedding, self.dblabels), 3)}'''),
                    html.Div(
                        f'''Davies Bouldin score: 
                               {round(metrics.davies_bouldin_score(self.embedding, self.dblabels), 3)}'''),
                    html.Div(
                        f'''Calinski Harabasz score: 
                               {round(metrics.calinski_harabasz_score(self.embedding, self.dblabels), 3)}'''),
                    dcc.Graph(id='dbscan_clusters',
                              figure=px.scatter(self.embedding,
                                                x=self.embedding[:, 0],
                                                y=self.embedding[:, 1],
                                                template=template).update_traces(marker=dict(color=self.dblabels,
                                                                                             size=15,
                                                                                             opacity=0.5
                                                                                             )))
                ], width=True),

                # AGGLOMERATIVE clustering results on UMAP 2D embedding >> plot + display some
                # performance evaluation metrics
                dbc.Col([
                    html.H6('AgglomerativeClustering on data with number dimensions reduced to 2 (UMAP)'),
                    html.Div(f'''Nuber of clusters: {len(set(self.aggl_labels))}'''),
                    html.Div(
                        f'''Silhouette score: 
                               {round(metrics.silhouette_score(self.embedding, self.aggl_labels), 3)}'''),
                    html.Div(
                        f'''Davies Bouldin score: 
                              {round(metrics.davies_bouldin_score(self.embedding, self.aggl_labels), 3)}'''),
                    html.Div(
                        f'''Calinski Harabasz score: 
                             {round(metrics.calinski_harabasz_score(self.embedding, self.aggl_labels), 3)}'''),
                    dcc.Graph(id='agglomerative_clusters',
                              figure=px.scatter(self.embedding,
                                                x=self.embedding[:, 0],
                                                y=self.embedding[:, 1],
                                                template=template, ).update_traces(
                                  marker=dict(color=self.aggl_labels,
                                              size=15,
                                              opacity=0.5)))
                ], width=True)])

        # here 2 graphs declared only; they will be created and updated
        # inside the callback function
        content_second_row = dbc.Row(

            [
                dbc.Col(
                    dcc.Graph(id='bar_plot_clustered'), md=6
                ),
                dbc.Col(
                    dcc.Graph(id='pie_chart_clustered'), md=6
                )
            ]
        )

        # RadioItems and Dropdown menu declared
        controls = dbc.Row([

            dbc.Col([
                html.Br(),
                html.P('Choose clustering algorithm',
                       style={'textAlign': 'center'}),
                dbc.Card([dbc.RadioItems(
                    id='clusterer_algorithm',
                    options=[{'label': x, 'value': x}
                             for x in ['DBSCAN', 'Agglomerative']],
                    value='DBSCAN',
                    style={'margin': 'auto'}
                )])
            ]),

            dbc.Col([
                html.Br(),
                html.P('Choose cluster number ', style={
                    'textAlign': 'center'}),
                # id only as Dropdown will be populated on the callback
                dcc.Dropdown(id='clusters_dropdown')

            ])])

        # put everything together: the 'content' will be later put in the app.layout()
        content = html.Div([
            html.H2('Clustered Data Exploratory Plots',
                    style={'textAlign': 'center',
                           'color': 'rgb(50,131,254)'}),
            html.Hr(),
            content_first_row,
            html.Hr(),
            controls,
            html.Hr(),
            content_second_row
        ],
            style={'margin': 'auto'
                   })

        # callback one: change dropdown menu according to the chosen RadioItem
        @self.new_app.callback(
            [Output('clusters_dropdown', 'options'),
             Output('clusters_dropdown', 'value')],
            Input('clusterer_algorithm', 'value'))
        def dropdown_options(radio_value):
            if radio_value == 'DBSCAN':
                options = [{'label': x, 'value': x} for x in sorted(self.df['DBSCAN'].unique())]
                value = '(choose from dropdown)'
            else:
                options = [{'label': x, 'value': x} for x in sorted(self.df['Agglomerative'].unique())]
                value = '(choose from dropdown)'
            return options, value

        # callback 2: update bar & piechart plots 
        @self.new_app.callback(
            Output('bar_plot_clustered', 'figure'),
            Output('pie_chart_clustered', 'figure'),
            Input('clusters_dropdown', 'value'),
            Input('clusterer_algorithm', 'value')
        )
        def update_theplots_on_callback(selected_clus_num, selected_alg):
            algorithms = ['DBSCAN', 'Agglomerative']
            dff = pd.DataFrame(self.df.groupby(selected_alg, as_index=False).mean())
            dff = dff[dff[selected_alg] == selected_clus_num]
            cols = [col for col in dff.columns if col not in algorithms]
            fig_bar = px.bar(dff, x=selected_alg, y=cols,
                             labels={'variable': 'Features', 'value': 'Values'},
                             barmode='group',
                             title=f'Bar plot: feature distribution in {selected_alg}, cluster {selected_clus_num}',
                             text_auto='.2s')
            fig_bar.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)

            df_pie = dff[cols].mean()
            fig = px.pie(df_pie, values=df_pie.values, names=df_pie.keys(),
                         title=f'Pie chart: feature distribution in {selected_alg}, cluster {selected_clus_num}')
            fig.update_traces(hole=.4, textposition='inside', textinfo='percent+label')
            return fig_bar, fig

        # finally, provide app.layout and run_server
        self.new_app.layout = dbc.Container([content], fluid=True, className="dbc")
        self.new_app.run_server(debug=True, host='127.0.0.1')

    #################
    # class methods #
    #################
    
    # this to get 2D UMAP embedding
    def _umap_embedding(self):
        reducer = umap.UMAP(n_neighbors=10, min_dist=.2)
        self.embedding = reducer.fit_transform(self.df1)

    # search for best DBSCAN parameters
    def _dbscan_tuner(self):
        samples = range(3, 15)
        epsilons = [.05, .1, .2, .3, .4, .5, .6, .7, .8, .9]
        best_silhouette = -1
        best_davis_bouldin = np.inf
        self.best_epsilon = 0
        self.best_min_samples = 0
        for i in samples:
            for j in epsilons:
                dbscan = DBSCAN(eps=j, min_samples=i).fit(self.embedding)
                dblabels = dbscan.labels_
                if len(set(dblabels)) > 1:
                    ss = metrics.silhouette_score(self.embedding, dblabels)
                    dbi = metrics.davies_bouldin_score(self.embedding, dblabels)
                    if ss > best_silhouette and dbi < best_davis_bouldin:
                        best_silhouette = ss
                        best_davis_bouldin = dbi
                        self.best_epsilon = j
                        self.best_min_samples = i

    # get the best parameters for AgglomerativeClustering
    def _agglomerative_tuner(self):
        nclusters = range(1, 15)
        best_silhouette = -1
        best_davis_bouldin = np.inf
        self.best_nclusters = 0
        for i in nclusters:
            clusterer = AgglomerativeClustering(n_clusters=i, affinity='euclidean', linkage='ward')
            labels = clusterer.fit_predict(self.embedding)
            if len(set(labels)) > 1:
                ss = metrics.silhouette_score(self.embedding, labels)
                dbi = metrics.davies_bouldin_score(self.embedding, labels)
                if ss > best_silhouette and dbi < best_davis_bouldin:
                    best_silhouette = ss
                    best_davis_bouldin = dbi
                    self.best_nclusters = i

    # run dbscan and agglomerative clusterer, save cluster labels explicitly to use in plotly plots
    # to calculate and display some error metrics (otherwise just appending them to the df would be enough)
    def _dbscan_umap(self):
        self._dbscan_tuner()
        dbscan = DBSCAN(eps=self.best_epsilon, min_samples=self.best_min_samples).fit(self.embedding)
        self.dblabels = dbscan.labels_
        self.df['DBSCAN'] = self.dblabels

    def _agglomerative_umap(self):
        self._agglomerative_tuner()
        clusterer = AgglomerativeClustering(n_clusters=self.best_nclusters, affinity='euclidean', linkage='ward')
        self.aggl_labels = clusterer.fit_predict(self.embedding)
        self.df['Agglomerative'] = self.aggl_labels
