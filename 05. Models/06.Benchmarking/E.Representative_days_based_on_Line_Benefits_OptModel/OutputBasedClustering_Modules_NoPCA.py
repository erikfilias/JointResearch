# libraries
import os
import time
import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
# from kneed import KneeLocator
from pyomo.environ      import *
from pyomo.opt          import SolverFactory

#%% Defining the clustering functions
def KMeansMethod(OptClusters, Y_sklearn, _path_0, _path_1, CaseName_0, CaseName_1, table, data, cluster_type, procedure_type):
    print("Shape of clustered frame: ", Y_sklearn.shape)
    # Running the K-means with the optimal number of clusters. Setting up the initializer and random state.
    kmeans_pca = KMeans(n_clusters=OptClusters, init='k-means++', random_state=0)
    kmeans_pca.fit(Y_sklearn)
    df_segm_pca_kmeans = pd.concat([table.reset_index(drop=True), pd.DataFrame(Y_sklearn)], axis=1)
    df_segm_pca_kmeans.columns.values[-3:] = ['Component 1', 'Component 2', 'Component 3']
    df_segm_pca_kmeans['Segment K-means PCA'] = kmeans_pca.labels_
    # Storing clusters in the first table
    table = table.copy()
    table['Segment K-means PCA'] = 0
    table.loc[:, 'Segment K-means PCA'] = kmeans_pca.labels_
    table = table.reset_index()
    if cluster_type == 'hourly':
        table = table.set_index(['LoadLevel', 'Day', 'Month', 'Segment K-means PCA'])
    elif cluster_type == 'daily with hourly resolution':
        table = table.set_index(['Day', 'Month', 'Segment K-means PCA'])
    elif cluster_type == 'weekly with hourly resolution':
        table = table.set_index(['Week', 'Segment K-means PCA'])
    # Stacking the table to also have the lines as index
    df = table.stack()
    df = df.reset_index()
    data = data.set_index(['LoadLevel'])
    #
    LoadLevel1 = df['LoadLevel'].unique()
    # filter rows using the LoadLevel index and the df index
    data = data.loc[LoadLevel1]
    # Adding a new column with the cluster for each LoadLevel
    data.reset_index(inplace=True)
    data['Segment K-means PCA'] = np.where(data['Variable'] == df['Variable'], df['Segment K-means PCA'], df['Segment K-means PCA'])
    # Adding the duration to each LoadLevel
    data['Duration'] = 0
    # Getting only the relevant information to build the new CSV file in CaseName_ByStages
    if procedure_type == 0:
        data['Stage'] = data['Segment K-means PCA'].map(lambda x: f'stp{x + 1}' if 0 <= x < 8000 else f'stp{x}')
    elif procedure_type == 1:
        data['Stage'] = data['Segment K-means PCA'].map(lambda x: f'stn{x + 1}' if 0 <= x < 8000 else f'stn{x}')
    elif procedure_type == 2:
        data['Stage'] = data['Segment K-means PCA'].map(lambda x: f'sti{x + 1}' if 0 <= x < 8000 else f'sti{x}')
    #
    # stages = data['Stage'].unique()
    # stages = np.sort(stages)
    # loadlevels = data['LoadLevel'].unique()

    data['HourOfYear'] = 0
    df_split = len(df['Variable'].unique())
    for i in data.index:
        if i < df_split:
            data.loc[i, 'HourOfYear'] = 0
        else:
            data.loc[i, 'HourOfYear'] = int(i/df_split)
    # find the first appearance of each stage in the data
    first_occurrences = data.groupby('Stage').first().reset_index()
    #
    if cluster_type == 'hourly':
        # data duration
        idx = first_occurrences['HourOfYear'].values
        dfHourToStage = pd.DataFrame(idx, columns=['Hour'])
        dfHourToStage = dfHourToStage.copy()
        for k in dfHourToStage.index:
            data.loc[data['HourOfYear'] == dfHourToStage['Hour'][k], 'Duration'] = 1
    elif cluster_type == 'daily with hourly resolution':
        idx = first_occurrences['Day'].values
        dfDayToStage = pd.DataFrame(idx, columns=['Day'])
        dfDayToStage = dfDayToStage + 1
        for k in dfDayToStage.index:
            data.loc[data['Day'] == dfDayToStage['Day'][k], 'Duration'] = 1
    elif cluster_type == 'weekly with hourly resolution':
        idx = first_occurrences['Week'].values
        dfWeekToStage = pd.DataFrame(idx, columns=['Day'])
        dfWeekToStage = dfWeekToStage + 1
        for k in dfWeekToStage.index:
            data.loc[data['Week'] == dfWeekToStage['Week'][k], 'Duration'] = 1
    # Getting only the relevant information to build the new CSV file in CaseName_ByStages
    data = data[
        ['LoadLevel', 'Stage', 'Execution', 'Duration', 'Value']]
    # Shaping the dataframe to be saved in CSV files
    TableToFile = pd.pivot_table(data, values='Value', index=['LoadLevel', 'Stage', 'Duration'],
                                 columns=['Execution'], fill_value=0)
    TableToFile = TableToFile.reset_index()
    LoadLevelToStage = TableToFile[['LoadLevel', 'Stage', 'Duration']]
    LoadLevelToStage.loc[:, 'Duration'] = 1
    LoadLevelToStage = pd.pivot_table(LoadLevelToStage, values='Duration', index=['LoadLevel'], columns='Stage', fill_value=0)
    # save the LoadLevelToStage dataframe
    LoadLevelToStage.index.name = None
    LoadLevelToStage.to_csv(os.path.join(_path_1, '2.Par', 'oT_Data_LoadLevelToStage_' + CaseName_1 + '.csv'), sep=',', index=True)
    # select rows based on condition that Duration is 1
    TableToFile = TableToFile[TableToFile['Duration'] == 1]
    # Creating the dataframe to generate oT_Data_Duration
    dfDuration = pd.DataFrame(0, index=TableToFile.index, columns=['LoadLevel', 'Duration', 'Stage'])
    dfDuration['LoadLevel'] = TableToFile['LoadLevel']
    dfDuration['Duration'] = TableToFile['Duration']
    dfDuration['Stage'] = TableToFile['Stage']
    # dfDuration.to_csv(os.path.join(_path_1, '2.Par', 'oT_Data_Duration_' + CaseName_1 + '.csv'), sep=',', index=False)
    # Identifying the Stages
    Stages = dfDuration.Stage.unique()
    Stages = np.sort(Stages)
    # Creating the dataframe to generate oT_Data_Stages
    dfa = pd.DataFrame({'Weight': data['Stage']})
    dfa = dfa['Weight'].value_counts()
    if cluster_type == 'hourly':
        dfa = dfa/int(df_split)
    elif cluster_type == 'daily with hourly resolution':
        dfa = dfa/int(24*df_split)
    elif cluster_type == 'weekly with hourly resolution':
        dfa = dfa/int(168*df_split)
    dfa = dfa.sort_index()
    dfStages = pd.DataFrame(dfa.values, index=dfa.index, columns=['Weight'])
    dfStages.index.name = None
    # dfStages.to_csv(os.path.join(_path_1, '2.Par', 'oT_Data_Stage_' + CaseName_1 + '.csv'), sep=',')
    # Creating the dataframe to generate oT_Dict_Stages
    dict_Stages = pd.DataFrame(Stages, columns=['Stage'])
    # dict_Stages.to_csv(os.path.join(_path_1, '1.Set', 'oT_Dict_Stage_' + CaseName_1 + '.csv'), sep=',', index=False)

    print('End of the process for clustering ' + CaseName_1 + '...' + str(procedure_type) + ' with ' + cluster_type + ' resolution')

    return kmeans_pca, dfDuration, dfStages, dict_Stages


def KMedoidsMethod(OptClusters, Y_sklearn, _path_0, _path_1, CaseName_0, CaseName_1, table, data, cluster_type, procedure_type):
    # Running the K-means with the optimal number of clusters. Setting up the initializer and random state.
    # kmedoids_pca = KMedoids(metric="euclidean", n_clusters=OptClusters, init="heuristic", max_iter=2, random_state=42)
    kmedoids_pca = KMedoids(metric="euclidean", n_clusters=OptClusters, init='k-medoids++', random_state=42)
    print("Shape of clustered frame: ", Y_sklearn.shape)

    # kmedoids_pca = KMedoids(metric="euclidean", n_clusters=OptClusters, init='k-medoids++', random_state=0)
    kmedoids_pca.fit(Y_sklearn)
    df_segm_pca_kmedoids = pd.concat([table.reset_index(drop=True), pd.DataFrame(Y_sklearn)], axis=1)
    df_segm_pca_kmedoids.columns.values[-3:] = ['Component 1', 'Component 2', 'Component 3']
    df_segm_pca_kmedoids['Segment K-medoids PCA'] = kmedoids_pca.labels_
    # Storing clusters in the first table
    print("--- Storing clusters in the first table " + CaseName_1 + "..." + str(procedure_type))
    table = table.copy()
    table['Segment K-medoids PCA'] = 0
    table.loc[:, 'Segment K-medoids PCA'] = kmedoids_pca.labels_
    table = table.reset_index()
    if cluster_type == 'hourly':
        table = table.set_index(['LoadLevel', 'Day', 'Month', 'Segment K-medoids PCA'])
    elif cluster_type == 'daily with hourly resolution':
        table = table.set_index(['Day', 'Month', 'Segment K-medoids PCA'])
    elif cluster_type == 'weekly with hourly resolution':
        table = table.set_index(['Week', 'Segment K-medoids PCA'])
    # Stacking the table to also have the lines as index
    print("--- Stacking the table to also have the lines as index " + CaseName_1 + "..." + str(procedure_type))
    df = table.stack()
    df = df.reset_index()
    data = data.set_index(['LoadLevel'])
    #
    LoadLevel1 = df['LoadLevel'].unique()
    # filter rows using the LoadLevel index and the df index
    data = data.loc[LoadLevel1]
    # Adding a new column with the cluster for each LoadLevel
    data.reset_index(inplace=True)
    print("--- Adding a new column with the cluster for each LoadLevel " + CaseName_1 + "..." + str(procedure_type))
    data['Segment K-medoids PCA'] = np.where(data['Variable'] == df['Variable'], df['Segment K-medoids PCA'], df['Segment K-medoids PCA'])
    # Adding the duration to each LoadLevel
    print("--- Adding the duration to each LoadLevel " + CaseName_1 + "..." + str(procedure_type))
    data['Duration'] = 0
    # Getting only the relevant information to build the new CSV file in CaseName_ByStages
    if procedure_type == 0:
        data['Stage'] = data['Segment K-medoids PCA'].map(lambda x: f'stp{x + 1}' if 0 <= x < 8000 else f'stp{x}')
    elif procedure_type == 1:
        data['Stage'] = data['Segment K-medoids PCA'].map(lambda x: f'stn{x + 1}' if 0 <= x < 8000 else f'stn{x}')
    elif procedure_type == 2:
        data['Stage'] = data['Segment K-medoids PCA'].map(lambda x: f'sti{x + 1}' if 0 <= x < 8000 else f'sti{x}')
    #
    print("--- Getting only the relevant information to build the new CSV file in CaseName_ByStages " + CaseName_1 + "..." + str(procedure_type))
    idx = kmedoids_pca.medoid_indices_
    # data['HourOfYear'] = (data['Day']-1)*24 + data['Hour']
    data['HourOfYear'] = 0
    df_split = len(df['Variable'].unique())
    for i in data.index:
        if i < df_split:
            data.loc[i, 'HourOfYear'] = 0
        else:
            data.loc[i, 'HourOfYear'] = int(i/df_split)
    if cluster_type == 'hourly':
        dfHourToStage = pd.DataFrame(idx, columns=['Hour'])
        dfHourToStage = dfHourToStage.copy()
        for k in dfHourToStage.index:
            data.loc[data['HourOfYear'] == dfHourToStage['Hour'][k], 'Duration'] = 1
    elif cluster_type == 'daily with hourly resolution':
        dfDayToStage = pd.DataFrame(idx, columns=['Day'])
        dfDayToStage = dfDayToStage + 1
        for k in dfDayToStage.index:
            data.loc[data['Day'] == dfDayToStage['Day'][k], 'Duration'] = 1
    elif cluster_type == 'weekly with hourly resolution':
        dfWeekToStage = pd.DataFrame(idx, columns=['Week'])
        dfWeekToStage = dfWeekToStage + 1
        for k in dfWeekToStage.index:
            data.loc[data['Week'] == dfWeekToStage['Week'][k], 'Duration'] = 1
    # Getting only the relevant information to build the new CSV file in CaseName_ByStages
    data = data[
        ['LoadLevel', 'Stage', 'Execution', 'Duration', 'Value']]
    # Shaping the dataframe to be saved in CSV files
    TableToFile = pd.pivot_table(data, values='Value', index=['LoadLevel', 'Stage', 'Duration'],
                                 columns=['Execution'], fill_value=0)
    TableToFile = TableToFile.reset_index()
    # Creating the dataframe to generate oT_Data_Duration
    dfDuration = pd.DataFrame(0, index=TableToFile.index, columns=['LoadLevel', 'Duration', 'Stage'])
    dfDuration['LoadLevel'] = TableToFile['LoadLevel']
    dfDuration['Duration'] = TableToFile['Duration']
    dfDuration['Stage'] = TableToFile['Stage']
    # dfDuration.to_csv(os.path.join(_path_1, '2.Par', 'oT_Data_Duration_' + CaseName_1 + '.csv'), sep=',', index=False)
    # Identifying the Stages
    Stages = dfDuration.Stage.unique()
    Stages = np.sort(Stages)
    # Creating the dataframe to generate oT_Data_Stages
    dfa = pd.DataFrame({'Weight': dfDuration['Stage']})
    dfa = dfa['Weight'].value_counts()
    if cluster_type == 'hourly':
        dfa = dfa
    elif cluster_type == 'daily with hourly resolution':
        dfa = dfa/24
    elif cluster_type == 'weekly with hourly resolution':
        dfa = dfa/168
    dfa = dfa.sort_index()
    dfStages = pd.DataFrame(dfa.values, index=dfa.index, columns=['Weight'])
    dfStages.index.name = None
    # dfStages.to_csv(os.path.join(_path_1, '2.Par', 'oT_Data_Stage_' + CaseName_1 + '.csv'), sep=',')
    # Creating the dataframe to generate oT_Dict_Stages
    dict_Stages = pd.DataFrame(Stages, columns=['Stage'])
    # dict_Stages.to_csv(os.path.join(_path_1, '1.Set', 'oT_Dict_Stage_' + CaseName_1 + '.csv'), sep=',', index=False)

    print("--- End of the Kmedoids clustering" + CaseName_1 + "..." + str(procedure_type))
    return kmedoids_pca, dfDuration, dfStages, dict_Stages

def ClusteringProcess(X,y, IndOptCluster, opt_cluster, _path_0, _path_1, CaseName_0, CaseName_1, table, data, cluster_type, procedure_type, max_cluster, cluster_method):
    #
    print("-- Clustering" + CaseName_1 + "..." + str(procedure_type))
    # Indicator save figure:
    IndFigure = 0
    # Prints
    # print(X)
    # print(y)
    # Standardizing of the data
    X_std = StandardScaler().fit_transform(X)

    #%% Clustering method
    print("-- Starting the clustering method" + CaseName_1 + "..." + str(procedure_type))
    if cluster_method == 0:
        results, dfDuration, dfStages, dict_Stages      = KMeansMethod(  opt_cluster, X_std, _path_0, _path_1, CaseName_0, CaseName_1, table, data, cluster_type, procedure_type)
    elif cluster_method == 1:
        results, dfDuration, dfStages, dict_Stages      = KMedoidsMethod(opt_cluster, X_std, _path_0, _path_1, CaseName_0, CaseName_1, table, data, cluster_type, procedure_type)
    # print('End of the process...')
    print("-- End of the clustering" + CaseName_1 + "..." + str(procedure_type))

    return results, dfDuration, dfStages, dict_Stages


def main(IndOptCluster, DirName, opt_cluster, CaseName_Base):
    #%% Setting up the path a cases

    if IndOptCluster == 1:
        CaseName_ByStages = CaseName_Base+'_ByStages'
    elif IndOptCluster == 0:
        CaseName_ByStages = CaseName_Base+'_ByStages_nc'+str(opt_cluster)

    CSV_name =  'oT_LineBenefit_Data_' + CaseName_Base

    _path_0 = os.path.join(DirName, CaseName_Base)
    _path_1 = os.path.join(DirName, CaseName_ByStages)

    #%% Selecting the maximum number of cluster to plot
    max_cluster = 1000

    # type of cluster method (0: k-means; 1:k-medoids)
    cluster_method = 1

    #%% type of clustering  ('hourly'; 'daily with hourly resolution'; 'weekly with hourly resolution')
    clustering_type = 'hourly'

    # if classification before clustering is performed
    IndClassify = 0

    output_directory = DirName + '/' + CaseName_ByStages + '/'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    df_1 = pd.read_csv(_path_0+'/3.Out'+'/'+CSV_name+'.csv', index_col=0, header=[0])
     # from multilevel header to single level header
    df_1.columns = ['_'.join(col) for col in df_1.columns.values]
    diff_df_1 = df_1

    #%% Loading Sets from CSV
    dictSets = DataPortal()
    dictSets.load(filename=_path_0+'/1.Set'+'/oT_Dict_LoadLevel_'   +CaseName_Base+'.csv', set='n'   , format='set')

    diff_df_1['LoadLevel'] = dictSets['n' ]
    diff_df_1.set_index('LoadLevel', inplace=True)

    # Create an explicit copy of the DataFrame
    diff_df_2 = diff_df_1.copy()

    # sum all the columns and save it into a column
    diff_df_2.loc[:, 'TotalInv'] = diff_df_2.sum(axis=1)

    # Selecting the columns
    diff_df_2 = diff_df_2['TotalInv']

    # Separate positive and negative values
    pos_values = diff_df_2[diff_df_2 > 0].sort_values(ascending=False)
    neg_values = diff_df_2[diff_df_2 < 0].sort_values()

    # Calculate the 90th percentile of positive and negative values
    pos_threshold = pos_values.quantile(0.9)
    neg_threshold = neg_values.quantile(0.1)  # use 0.1 because neg_values is sorted in ascending order

    # Create a new Series to store the labels, with the same index as diff_df_2
    labels = pd.Series('Irrelevant', index=diff_df_2.index)

    if IndClassify == 1:
        # Classify the values
        labels[pos_values[pos_values > pos_threshold].index] = 'Positive'
        labels[neg_values[neg_values < neg_threshold].index] = 'Negative'
    elif IndClassify == 0:
        # Classify the values
        labels[pos_values[pos_values > len(diff_df_1.columns)+1].index] = 'Positive'
        labels[neg_values[neg_values < 0].index] = 'Negative'

    # Add a new column 'Mark' to diff_df_2 and assign the labels to it
    diff_df_2 = labels
    diff_df_2 = diff_df_2.to_frame(name='Mark') # convert Series to DataFrame
    # diff_df_1 = diff_df_1.join(diff_df_2)
    # rename column
    ddf_1 = diff_df_1.stack().reset_index().rename(columns={'LoadLevel': 'LoadLevel', 'level_1': 'Execution', 0: 'Value'})

    ddf_1 = ddf_1.reset_index()

    ddf_1['Date'] = ddf_1.reset_index()['LoadLevel']
    ddf_1['Date'] = ddf_1['Date'].str.slice(0, -6)

    ddf_1['Date'] = pd.to_datetime(ddf_1['Date'], format='%m-%d %H:%M:%S', errors='coerce')

    ddf_1['Hour' ] = ddf_1['Date'].dt.hour
    ddf_1['Day'  ] = ddf_1['Date'].dt.dayofyear
    ddf_1['Week' ] = ddf_1['Date'].dt.isocalendar().week
    ddf_1['Month'] = ddf_1['Date'].dt.month

    if clustering_type == 'hourly':
        ddf_1['Variable'] = ddf_1.reset_index()['Execution']
    elif clustering_type == 'daily with hourly resolution':
        ddf_1['Variable'] = ddf_1['Execution'] + '_' + ddf_1['Hour'].astype(str)
    elif clustering_type == 'weekly with hourly resolution':
        ddf_1['Variable'] = ddf_1['Execution'] + '_' + ddf_1['Hour'].astype(str)

    if clustering_type == 'hourly':
        table = pd.pivot_table(ddf_1, values='Value', index=['LoadLevel', 'Month', 'Day'], columns=['Variable'], aggfunc='sum')
    elif clustering_type == 'daily with hourly resolution':
        table = pd.pivot_table(ddf_1, values='Value', index=['Month', 'Day'], columns=['Variable'], aggfunc='sum')
    elif clustering_type == 'weekly with hourly resolution':
        table = pd.pivot_table(ddf_1, values='Value', index=['Month', 'Week'], columns=['Variable'], aggfunc='sum')

    table = table.reset_index()

    if clustering_type == 'hourly':
        table = table.set_index(['LoadLevel'])
    elif clustering_type == 'daily with hourly resolution':
        table = table.set_index(['Day'])
    elif clustering_type == 'weekly with hourly resolution':
        table = table.set_index(['Week'])

    # split the table using the marks in diff_df_2
    table_positive = table[diff_df_2['Mark'] == 'Positive']
    table_negative = table[diff_df_2['Mark'] == 'Negative']
    table_irrelevant = table[diff_df_2['Mark'] == 'Irrelevant']

    # perform the clustering on the tables and merge the results
    # splitting the opt_cluster in the same proportion of the original table but having the sum equal to opt_cluster
    if IndOptCluster == 0:
        opt_cluster_positive = math.ceil(opt_cluster*len(table_positive)/len(table))
        print('- opt_cluster_positive', opt_cluster_positive)
        opt_cluster_negative = math.ceil(opt_cluster*len(table_negative)/len(table))
        print('- opt_cluster_negative', opt_cluster_negative)
        opt_cluster_irrelevant = math.ceil(opt_cluster*len(table_irrelevant)/len(table))
        print('- opt_cluster_irrelevant', opt_cluster_irrelevant)
        print('- sum of clusters', opt_cluster_positive + opt_cluster_negative + opt_cluster_irrelevant)
    else:
        opt_cluster_positive = opt_cluster
        opt_cluster_negative = opt_cluster
        opt_cluster_irrelevant = opt_cluster

    if len(table_positive):
        print('Clustering positive...', len(table_positive))
        X_positive = table_positive.iloc[:,1:len(table_positive.columns)+1].values
        y_positive = table_positive.iloc[:,0].values
        results_positive, dfDuration_positive, dfStages_positive, dictStages_positive = ClusteringProcess(X_positive, y_positive, IndOptCluster, opt_cluster_positive, _path_0, _path_1, CaseName_Base, CaseName_ByStages, table_positive, ddf_1, clustering_type, 0, max_cluster, cluster_method)
    if len(table_negative):
        print('Clustering negative...', len(table_negative))
        X_negative = table_negative.iloc[:,1:len(table_negative.columns)+1].values
        y_negative = table_negative.iloc[:,0].values
        results_negative, dfDuration_negative, dfStages_negative, dictStages_negative = ClusteringProcess(X_negative, y_negative, IndOptCluster, opt_cluster_negative, _path_0, _path_1, CaseName_Base, CaseName_ByStages, table_negative, ddf_1, clustering_type, 1, max_cluster, cluster_method)

    if len(table_irrelevant):
        print('Clustering irrelevant...', len(table_irrelevant))
        X_irrelevant = table_irrelevant.iloc[:,1:len(table_irrelevant.columns)+1].values
        y_irrelevant = table_irrelevant.iloc[:,0].values
        results_irrelevant, dfDuration_irrelevant, dfStages_irrelevant, dictStages_irrelevant = ClusteringProcess(X_irrelevant, y_irrelevant, IndOptCluster, opt_cluster_irrelevant, _path_0, _path_1, CaseName_Base, CaseName_ByStages, table_irrelevant, ddf_1, clustering_type, 2, max_cluster, cluster_method)

    # Merging and ordering the dfDuration dataframes
    if len(table_positive) and len(table_negative) and len(table_irrelevant):
        dfDuration = pd.concat([dfDuration_positive, dfDuration_negative, dfDuration_irrelevant])
        dfDuration = dfDuration.sort_values(by=['LoadLevel', 'Stage'])
    elif len(table_positive) and len(table_negative):
        dfDuration = pd.concat([dfDuration_positive, dfDuration_negative])
        dfDuration = dfDuration.sort_values(by=['LoadLevel', 'Stage'])
    elif len(table_positive) and len(table_irrelevant):
        dfDuration = pd.concat([dfDuration_positive, dfDuration_irrelevant])
        dfDuration = dfDuration.sort_values(by=['LoadLevel', 'Stage'])
    elif len(table_negative) and len(table_irrelevant):
        dfDuration = pd.concat([dfDuration_negative, dfDuration_irrelevant])
        dfDuration = dfDuration.sort_values(by=['LoadLevel', 'Stage'])
    elif len(table_positive):
        dfDuration = dfDuration_positive
        dfDuration = dfDuration.sort_values(by=['LoadLevel', 'Stage'])
    elif len(table_negative):
        dfDuration = dfDuration_negative
        dfDuration = dfDuration.sort_values(by=['LoadLevel', 'Stage'])
    elif len(table_irrelevant):
        dfDuration = dfDuration_irrelevant
        dfDuration = dfDuration.sort_values(by=['LoadLevel', 'Stage'])

    # Merging and ordering the dfStages dataframes
    if len(table_positive) and len(table_negative) and len(table_irrelevant):
        dfStages = pd.concat([dfStages_positive, dfStages_negative, dfStages_irrelevant])
        dfStages = dfStages.sort_index()
    elif len(table_positive) and len(table_negative):
        dfStages = pd.concat([dfStages_positive, dfStages_negative])
        dfStages = dfStages.sort_index()
    elif len(table_positive) and len(table_irrelevant):
        dfStages = pd.concat([dfStages_positive, dfStages_irrelevant])
        dfStages = dfStages.sort_index()
    elif len(table_negative) and len(table_irrelevant):
        dfStages = pd.concat([dfStages_negative, dfStages_irrelevant])
        dfStages = dfStages.sort_index()
    elif len(table_positive):
        dfStages = dfStages_positive
        dfStages = dfStages.sort_index()
    elif len(table_negative):
        dfStages = dfStages_negative
        dfStages = dfStages.sort_index()
    elif len(table_irrelevant):
        dfStages = dfStages_irrelevant
        dfStages = dfStages.sort_index()

    # Merge the dict_Stages dataframes
    if len(table_positive) and len(table_negative) and len(table_irrelevant):
        dict_Stages = pd.concat([dictStages_positive, dictStages_negative, dictStages_irrelevant])
        dict_Stages = dict_Stages.sort_index()
    elif len(table_positive) and len(table_negative):
        dict_Stages = pd.concat([dictStages_positive, dictStages_negative])
        dict_Stages = dict_Stages.sort_index()
    elif len(table_positive) and len(table_irrelevant):
        dict_Stages = pd.concat([dictStages_positive, dictStages_irrelevant])
        dict_Stages = dict_Stages.sort_index()
    elif len(table_negative) and len(table_irrelevant):
        dict_Stages = pd.concat([dictStages_negative, dictStages_irrelevant])
        dict_Stages = dict_Stages.sort_index()
    elif len(table_positive):
        dict_Stages = dictStages_positive
        dict_Stages = dict_Stages.sort_index()
    elif len(table_negative):
        dict_Stages = dictStages_negative
        dict_Stages = dict_Stages.sort_index()
    elif len(table_irrelevant):
        dict_Stages = dictStages_irrelevant
        dict_Stages = dict_Stages.sort_index()

    # %%
    # Saving the DataFrames in CSV files
    # Define paths
    path_duration = os.path.join(_path_1, '2.ParWoPCA')
    path_stages = os.path.join(_path_1, '2.ParWoPCA')
    path_dict_stages = os.path.join(_path_1, '1.SetWoPCA')

    # Ensure directories exist, create them if they don't
    os.makedirs(path_duration, exist_ok=True)
    os.makedirs(path_dict_stages, exist_ok=True)

    # Save CSV files
    dfDuration.to_csv(os.path.join(path_duration, 'oT_Data_Duration_' + CaseName_ByStages + '.csv'), sep=',',
                      index=False)
    dfStages.to_csv(os.path.join(path_stages, 'oT_Data_Stage_' + CaseName_ByStages + '.csv'), sep=',')
    dict_Stages.to_csv(os.path.join(path_dict_stages, 'oT_Dict_Stage_' + CaseName_ByStages + '.csv'), sep=',',
                       index=False)

    print('- Number of representative stages: ' + str(dfDuration['Duration'].sum()))

    print('- End of the process for ' + CaseName_ByStages + '...')
