
import getdashboard_
"""
Class GetDashboard takes in clean, pre-processed and encoded dataset, 
runs clustering on the data and plots the results in Plotly Dash dashboard.
Just feed one neat dataframe to it.

"""

if __name__ == '__main__':

    # run on the Starbucks dataset:
    clusterer = getdashboard_.GetDashboard('Starbucks_cleaned.csv')

    # run on the Wine dataset:
    # another_clusterer = getdashboard.GetDashboard('wine-clustering.csv')

    # run on the Absenteeism dataset:
    # clusterer3 = getdashboard.GetDashboard('Absenteeism_at_work.csv')

