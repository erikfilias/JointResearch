#%% Libraries
import argparse
import os
import numpy         as np
import pandas        as pd
import time          # count clock time
import math          # access to math operations
import psutil        # access the number of CPUs
from   collections   import defaultdict
from   pyomo.environ import ConcreteModel, Set, Param, Var, Objective, minimize, Constraint, DataPortal, PositiveIntegers, NonNegativeIntegers, Boolean, NonNegativeReals, UnitInterval, PositiveReals, Any, Binary, Reals, Suffix
from   pyomo.opt     import SolverFactory
from   oSN_Main      import openStarNet_run

#%% Defining metadata
parser = argparse.ArgumentParser(description='Introducing main parameters...')
parser.add_argument('--case',   type=str, default=None)
parser.add_argument('--dir',    type=str, default=None)
parser.add_argument('--solver', type=str, default=None)

DIR    = os.path.dirname(__file__)
CASE   = '9n'
SOLVER = 'gurobi'

# Initial time counter for all the code
initial_time = time.time()

# Calling the main function
def main():
    args = parser.parse_args()
    if args.dir is None:
        args.dir    = input('Input Dir    Name (Default {}): '.format(DIR))
        if args.dir == '':
            args.dir = DIR
    if args.case is None:
        args.case   = input('Input Case   Name (Default {}): '.format(CASE))
        if args.case == '':
            args.case = CASE
    if args.solver is None:
        args.solver = input('Input Solver Name (Default {}): '.format(SOLVER))
        if args.solver == '':
            args.solver = SOLVER
    print(args.case)
    print(args.dir)
    print(args.solver)
    import sys
    print(sys.argv)
    print(args)

    #%% Reading data from CSV files
    _path      = os.path.join(args.dir, args.case)
    start_time = time.time()

    dictSets = DataPortal()
    dictSets.load(filename=_path+'/1.Set/oT_Dict_Period_'      +args.case+'.csv', set='p'   , format='set')
    dictSets.load(filename=_path+'/1.Set/oT_Dict_Scenario_'    +args.case+'.csv', set='sc'  , format='set')
    dictSets.load(filename=_path+'/1.Set/oT_Dict_Stage_'       +args.case+'.csv', set='st'  , format='set')
    dictSets.load(filename=_path+'/1.Set/oT_Dict_LoadLevel_'   +args.case+'.csv', set='n'   , format='set')
    dictSets.load(filename=_path+'/1.Set/oT_Dict_Generation_'  +args.case+'.csv', set='g'   , format='set')
    dictSets.load(filename=_path+'/1.Set/oT_Dict_Technology_'  +args.case+'.csv', set='gt'  , format='set')
    dictSets.load(filename=_path+'/1.Set/oT_Dict_Storage_'     +args.case+'.csv', set='et'  , format='set')
    dictSets.load(filename=_path+'/1.Set/oT_Dict_Node_'        +args.case+'.csv', set='nd'  , format='set')
    dictSets.load(filename=_path+'/1.Set/oT_Dict_Zone_'        +args.case+'.csv', set='zn'  , format='set')
    dictSets.load(filename=_path+'/1.Set/oT_Dict_Area_'        +args.case+'.csv', set='ar'  , format='set')
    dictSets.load(filename=_path+'/1.Set/oT_Dict_Region_'      +args.case+'.csv', set='rg'  , format='set')
    dictSets.load(filename=_path+'/1.Set/oT_Dict_Circuit_'     +args.case+'.csv', set='cc'  , format='set')
    dictSets.load(filename=_path+'/1.Set/oT_Dict_Line_'        +args.case+'.csv', set='lt'  , format='set')

    dictSets.load(filename=_path+'/1.Set/oT_Dict_NodeToZone_'  +args.case+'.csv', set='ndzn', format='set')
    dictSets.load(filename=_path+'/1.Set/oT_Dict_ZoneToArea_'  +args.case+'.csv', set='znar', format='set')
    dictSets.load(filename=_path+'/1.Set/oT_Dict_AreaToRegion_'+args.case+'.csv', set='arrg', format='set')

    reading_data_time = time.time() - start_time
    start_time        = time.time()
    print('Reading    the sets of the model      ... ', round(reading_data_time), 's')

    #%% Reading the network data
    df_Network    = pd.read_csv(_path+'/2.Par/oT_Data_Network_'   +args.case+'.csv', index_col=[0,1,2])
    df_Generation = pd.read_csv(_path+'/2.Par/oT_Data_Generation_'+args.case+'.csv', index_col=[0    ])

    df_Network_full    = pd.read_csv(_path+'/2.Par/oT_Data_Network_'   +args.case+'.csv', index_col=[0,1,2])
    df_Generation_full = pd.read_csv(_path+'/2.Par/oT_Data_Generation_'+args.case+'.csv', index_col=[0    ])

    df_Network_existing    = pd.read_csv(_path+'/2.Par/oT_Data_Network_'   +args.case+'.csv', index_col=[0,1,2])
    df_Generation_existing = pd.read_csv(_path+'/2.Par/oT_Data_Generation_'+args.case+'.csv', index_col=[0    ])

    # Running the openStarNet with all the full network and generation units
    ## Modyfing the dataframes to run the openStarNet
    df_Network_full[   "InitialPeriod"]    = 2020
    df_Generation_full["InitialPeriod"]    = 2020

    df_Network_full[   "Sensitivity"  ]    = 'Yes'
    df_Generation_full["Sensitivity"  ]    = 'Yes'

    df_Network_full[   "InvestmentFixed"]  = 1
    df_Generation_full["InvestmentFixed"]  = 1

    df_Network_full.to_csv(   _path+'/2.Par/oT_Data_Network_'   +args.case+'.csv')
    df_Generation_full.to_csv(_path+'/2.Par/oT_Data_Generation_'+args.case+'.csv')

    ## Running the openStarNet
    # os.system('python3 openStarNet.py -d '+args.dir+' -c '+args.case+'_full -s '+args.solver)
    oSN   = ConcreteModel()
    model = openStarNet_run(args.dir, args.case, args.solver, oSN)

    for i in dictSets['nd']:
        print(f"Node: {i}")

    # for (i,j,k) in df_Network.index*df_Generation.index:
    #     print(f"line {i} {j} {k}, generation {k}")
    #     print("line", i, j, k, "generation", k")
    #     if df_Network.loc[i,j,'Type'] == 'Line':
    #         df_Network.loc[i,j,'Capacity'] = 9999999999

    reading_data_time = time.time() - start_time
    start_time        = time.time()
    print('Reading    the parameters of the model ... ', round(reading_data_time), 's')

    #%% Saving the input data
    df_input_data = pd.DataFrame(columns=['Execution'], index=pd.MultiIndex.from_tuples(model.psn))
    df_input_data.index.names = ['Period','Scenario','LoadLevel']

    # Defining the dataframe for the input data of the neural network
    for (p,sc,n) in model.psn:
        df_input_data.loc[(p,sc,n),'Execution'] = "Network_Full_Generation_Full"
    
    # Extracting the demand data
    df_demand = pd.Series(data=[model.pDemand[p,sc,n,nd] for p,sc,n,nd in model.psnnd], index=pd.MultiIndex.from_tuples(model.psnnd))
    df_demand.index.names = ['Period','Scenario','LoadLevel','Node']
    df_demand = df_demand.reset_index().pivot_table(index=['Period','Scenario','LoadLevel'], columns='Node', values=0)

    # Extracting the network data (admittance matrix)
    size = len(model.nd)
    # Convert the set to a list
    nodes = list(model.nd)
    admittance_matrix = np.zeros((size, size), dtype=complex)
    # Iterate over each row in the DataFrame and populate the admittance matrix
    for (ni,nf,cc) in model.la:
        node_1 = ni
        node_2 = nf
        reactance = model.pLineX[ni,nf,cc]
        resistance = model.pLineR[ni,nf,cc]
        susceptance = model.pLineBsh[ni,nf,cc]()

        # find the index of the nodes in the admittance matrix
        # index_1 = np.where(list(model.nd) == node_1)[0][0]
        # index_2 = np.where(list(model.nd) == node_2)[0][0]
        index_1 = nodes.index(ni)
        index_2 = nodes.index(nf)

        admittance = 1 / (reactance + resistance * 1j + susceptance * 1j)
        admittance_matrix[index_1][index_2] -= admittance
        admittance_matrix[index_2][index_1] -= admittance

    print(admittance_matrix)
    df = pd.DataFrame(admittance_matrix).stack().reset_index()
    df.columns = ['Node1', 'Node2', 'Admittance']
    df.set_index(['Node1', 'Node2'], inplace=True)
    print(df)
    df_Y_matrix = pd.DataFrame(index=pd.MultiIndex.from_tuples(model.psn))

    for (p,sc,n) in model.psn:
        for (ni,nf) in df.index:
            df_Y_matrix.loc[(p,sc,n),'Node_'+str(ni+1)+'_Node_'+str(nf+1)] = df.loc[(ni,nf),'Admittance']


    print(df_Y_matrix)

    # Extracting the maximum power generation data
    df_max_power = pd.Series(data=[model.pMaxPower[p,sc,n,g] for p,sc,n,g in model.psng], index=pd.MultiIndex.from_tuples(model.psng))
    df_max_power.index.names = ['Period','Scenario','LoadLevel','Unit']
    print(df_max_power)
    df_max_power = df_max_power.reset_index().pivot_table(index=['Period','Scenario','LoadLevel'], columns='Unit', values=0)

    # Merging all the data
    df_input_data = pd.concat([df_input_data, df_demand, df_Y_matrix, df_max_power], axis=1)
    df_input_data.to_csv(_path+'/3.Out/oT_Result_NN_Input_'+args.case+'.csv', index=True)

    #%% Saving the results
    df_output_data = pd.DataFrame(columns=['vTotalSCost','vTotalFCost','vTotalGCost','vTotalCCost','vTotalECost','vTotalRCost'], index=pd.MultiIndex.from_tuples(model.psn))
    df_output_data.index.names = ['Period','Scenario','LoadLevel']

    for (p,sc,n) in model.psn:
        df_output_data.loc[(p,sc,n),'vTotalSCost'] = model.vTotalSCost()
        df_output_data.loc[(p,sc,n),'vTotalFCost'] = model.pDiscountFactor[p]                           * model.vTotalFCost[p     ]()
        df_output_data.loc[(p,sc,n),'vTotalGCost'] = model.pDiscountFactor[p] * model.pScenProb[p,sc]() * model.vTotalGCost[p,sc,n]()
        df_output_data.loc[(p,sc,n),'vTotalCCost'] = model.pDiscountFactor[p] * model.pScenProb[p,sc]() * model.vTotalCCost[p,sc,n]()
        df_output_data.loc[(p,sc,n),'vTotalECost'] = model.pDiscountFactor[p] * model.pScenProb[p,sc]() * model.vTotalECost[p,sc,n]()
        df_output_data.loc[(p,sc,n),'vTotalRCost'] = model.pDiscountFactor[p] * model.pScenProb[p,sc]() * model.vTotalRCost[p,sc,n]()

    print(df_output_data)
    df_output_data.to_csv(_path+'/3.Out/oT_Result_NN_Output_'+args.case+'.csv', index=True)

    #%% Restoring the dataframes
    df_Network.to_csv(   _path+'/2.Par/oT_Data_Network_'   +args.case+'.csv')
    df_Generation.to_csv(_path+'/2.Par/oT_Data_Generation_'+args.case+'.csv')

if __name__ == '__main__':
    main()