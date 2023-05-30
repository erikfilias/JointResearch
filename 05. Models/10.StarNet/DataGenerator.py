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


def ModelRun(m, execution, path, dir, case, solver, dictSets):
    start_time = time.time()
    _path = path
    model = openStarNet_run(dir, case, solver, m)

    # for i in dictSets['nd']:
    #     print(f"Node: {i}")

    # for (i,j,k) in df_Network.index*df_Generation.index:
    #     print(f"line {i} {j} {k}, generation {k}")
    #     print("line", i, j, k, "generation", k")
    #     if df_Network.loc[i,j,'Type'] == 'Line':
    #         df_Network.loc[i,j,'Capacity'] = 9999999999

    reading_data_time = time.time() - start_time
    start_time = time.time()
    print('Reading    the parameters of the model ... ', round(reading_data_time), 's')

    # %% Saving the input data
    # Extracting the demand data
    df_demand = pd.Series(data=[model.pDemand[p, sc, n, nd] for p, sc, n, nd in model.psnnd],
                          index=pd.MultiIndex.from_tuples(model.psnnd))
    df_demand.index.names = ['Period', 'Scenario', 'LoadLevel', 'Variable']
    df_demand = df_demand.reset_index().pivot_table(index=['Period', 'Scenario', 'LoadLevel', 'Variable'], values=0)
    df_demand.rename(columns={0: 'Value'}, inplace=True)
    df_demand['Dataset'] = 'ElectricityDemand'
    df_demand['Execution'] = execution

    data_time = time.time() - start_time
    start_time = time.time()
    print('Getting the electricity demand         ... ', round(data_time), 's')

    # Extracting the network data (admittance matrix)
    size = len(model.nd)
    # Convert the set to a list
    nodes = list(model.nd)
    admittance_matrix = np.zeros((size, size), dtype=complex)
    # Iterate over each row in the DataFrame and populate the admittance matrix
    for (ni, nf, cc) in model.la:
        reactance = model.pLineX[ni, nf, cc]
        resistance = model.pLineR[ni, nf, cc]
        susceptance = model.pLineBsh[ni, nf, cc]()

        # find the index of the nodes in the admittance matrix
        index_1 = nodes.index(ni)
        index_2 = nodes.index(nf)

        admittance = 1 / (reactance + resistance * 1j + susceptance * 1j)
        admittance_matrix[index_1][index_2] -= admittance
        admittance_matrix[index_2][index_1] -= admittance

    df = pd.DataFrame(admittance_matrix).stack().reset_index()
    df.columns = ['Node1', 'Node2', 'Admittance']
    df.set_index(['Node1', 'Node2'], inplace=True)
    df_Y_matrix = pd.DataFrame(index=pd.MultiIndex.from_tuples(model.psn))

    for (p, sc, n) in model.psn:
        for (ni, nf) in df.index:
            df_Y_matrix.loc[(p, sc, n), 'Node_' + str(ni + 1) + '_Node_' + str(nf + 1)] = df.loc[(ni, nf), 'Admittance']

    df_Y_matrix = df_Y_matrix.stack()
    df_Y_matrix.index.names = ['Period', 'Scenario', 'LoadLevel', 'Variable']
    df_Y_matrix = df_Y_matrix.to_frame(name='Value')
    df_Y_matrix['Dataset'] = 'MatrixY'
    df_Y_matrix['Execution'] = execution

    data_time = time.time() - start_time
    start_time = time.time()
    print('Getting the Y matrix                   ... ', round(data_time), 's')

    # Extracting the maximum power generation data
    df_max_power = pd.Series(data=[model.pMaxPower[p, sc, n, g] for p, sc, n, g in model.psng],
                             index=pd.MultiIndex.from_tuples(model.psng))
    df_max_power.index.names = ['Period', 'Scenario', 'LoadLevel', 'Variable']
    df_max_power = df_max_power.reset_index().pivot_table(index=['Period', 'Scenario', 'LoadLevel', 'Variable'],
                                                          values=0)
    df_max_power.rename(columns={0: 'Value'}, inplace=True)
    df_max_power['Dataset'] = 'MaxPowerGeneration'
    df_max_power['Execution'] = execution

    data_time = time.time() - start_time
    start_time = time.time()
    print('Getting the max power generation       ... ', round(data_time), 's')

    # Merging all the data
    df_input_data = pd.concat([df_demand, df_Y_matrix, df_max_power])
    # df_input_data.to_csv(_path + '/3.Out/oT_Result_NN_Input_' + args.case + '.csv', index=True)

    data_time = time.time() - start_time
    start_time = time.time()
    print('Getting the input data file            ... ', round(data_time), 's')

    # %% Saving the results
    df_output_data = pd.DataFrame(
        columns=['vTotalSCost', 'vTotalFCost', 'vTotalGCost', 'vTotalCCost', 'vTotalECost', 'vTotalRCost'],
        index=pd.MultiIndex.from_tuples(model.psn))
    df_output_data.index.names = ['Period', 'Scenario', 'LoadLevel']

    for (p, sc, n) in model.psn:
        df_output_data.loc[(p, sc, n), 'vTotalSCost'] = model.vTotalSCost()
        df_output_data.loc[(p, sc, n), 'vTotalFCost'] = model.pDiscountFactor[p] * model.vTotalFCost[p]()
        df_output_data.loc[(p, sc, n), 'vTotalGCost'] = model.pDiscountFactor[p] * model.pScenProb[p, sc]() * \
                                                        model.vTotalGCost[p, sc, n]()
        df_output_data.loc[(p, sc, n), 'vTotalCCost'] = model.pDiscountFactor[p] * model.pScenProb[p, sc]() * \
                                                        model.vTotalCCost[p, sc, n]()
        df_output_data.loc[(p, sc, n), 'vTotalECost'] = model.pDiscountFactor[p] * model.pScenProb[p, sc]() * \
                                                        model.vTotalECost[p, sc, n]()
        df_output_data.loc[(p, sc, n), 'vTotalRCost'] = model.pDiscountFactor[p] * model.pScenProb[p, sc]() * \
                                                        model.vTotalRCost[p, sc, n]()

    df_output_data = df_output_data.stack().to_frame(name='Value')
    df_output_data.index.names = ['Period', 'Scenario', 'LoadLevel', 'Variable']
    df_output_data['Dataset'] = 'SystemCosts'
    df_output_data['Execution'] = execution
    # df_output_data.to_csv(_path + '/3.Out/oT_Result_NN_Output_' + args.case + '.csv', index=True)

    data_time = time.time() - start_time
    print('Getting the output data file           ... ', round(data_time), 's')

    return df_input_data, df_output_data


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
    # Initial time counter for all the code
    initial_time = time.time()

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

    #%% Reading the network data
    df_Network    = pd.read_csv(_path+'/2.Par/oT_Data_Network_'   +args.case+'.csv', index_col=[0,1,2])
    df_Generation = pd.read_csv(_path+'/2.Par/oT_Data_Generation_'+args.case+'.csv', index_col=[0    ])

    #%% Sequence of the full network
    print("―――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――")
    print("Sequence of the full network and generation")
    print("―――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――")
    df_Network_full    = pd.read_csv(_path+'/2.Par/oT_Data_Network_'   +args.case+'.csv', index_col=[0,1,2])
    df_Generation_full = pd.read_csv(_path+'/2.Par/oT_Data_Generation_'+args.case+'.csv', index_col=[0    ])

    ## Modyfing the dataframes to run the openStarNet
    df_Network_full[   "InitialPeriod"]   = 2020
    df_Generation_full["InitialPeriod"]   = 2020

    df_Network_full[   "Sensitivity"  ]   = 'Yes'
    df_Generation_full["Sensitivity"  ]   = 'Yes'

    df_Network_full[   "InvestmentFixed"] = 1
    df_Generation_full["InvestmentFixed"] = 1

    df_Network_full.to_csv(   _path+'/2.Par/oT_Data_Network_'   +args.case+'.csv')
    df_Generation_full.to_csv(_path+'/2.Par/oT_Data_Generation_'+args.case+'.csv')

    ## Running the openStarNet
    oSN       = ConcreteModel()
    execution = 'Network_Full_Generation_Full'

    df_Inp, df_Out = ModelRun(oSN, execution, _path, args.dir, args.case, args.solver, dictSets)

    # df_input_data  = pd.concat([df_demand, df_Y_matrix, df_max_power])
    # df_output_data = pd.concat([df_demand, df_Y_matrix, df_max_power])
    df_input_data  = df_Inp
    df_output_data = df_Out

    #%% Restoring the dataframes
    df_Network.to_csv(   _path+'/2.Par/oT_Data_Network_'   +args.case+'.csv')
    df_Generation.to_csv(_path+'/2.Par/oT_Data_Generation_'+args.case+'.csv')

    ####################################################################################################################
    #%% Sequence of the only existing network
    print("―――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――")
    print("Sequence of considering N-1 in the transmission network")
    print("―――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――")

    for (ni,nf,cc) in df_Network.index:
        if df_Network['BinaryInvestment'][ni,nf,cc] == 'Yes':
            print("――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――")
            print(f"Line {ni} {nf} {cc}")
            print("――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――")
            df_Network_Mod = pd.read_csv(_path + '/2.Par/oT_Data_Network_' + args.case + '.csv', index_col=[0, 1, 2])
            # df_Generation_Mod = pd.read_csv(_path+'/2.Par/oT_Data_Generation_'+args.case+'.csv', index_col=[0    ])

            # Modifying the dataframes to run the openStarNet
            df_Network_Mod.loc[(ni,nf,cc), 'InitialPeriod'] = 2020
            df_Network_Mod.loc[(ni,nf,cc), 'Sensitivity']   = "Yes"
            df_Network_Mod.loc[(ni,nf,cc), 'InvestmentFixed'] = 1
            for (ni2,nf2,cc2) in df_Network.index:
                if ((ni,nf,cc) != (ni2,nf2,cc2)) and df_Network['BinaryInvestment'][ni2,nf2,cc2] == 'Yes':
                    df_Network_Mod.loc[(ni2, nf2, cc2), 'InitialPeriod'] = 2049
                    df_Network_Mod.loc[(ni2, nf2, cc2), 'Sensitivity'] = "Yes"
                    df_Network_Mod.loc[(ni2, nf2, cc2), 'InvestmentFixed'] = 0

            # Saving the CSV file with the existing network
            df_Network_Mod.to_csv(_path + '/2.Par/oT_Data_Network_' + args.case + '.csv')

            # ## Modyfing the dataframes to run the openStarNet
            # df_Network_Mod.loc[df_Network_Mod['BinaryInvestment'] == 'Yes', 'InitialPeriod']   = 2049
            # df_Network_Mod.loc[df_Network_Mod['BinaryInvestment'] == 'Yes', 'Sensitivity'  ]   = 'Yes'
            # df_Network_Mod.loc[df_Network_Mod['BinaryInvestment'] == 'Yes', 'InvestmentFixed'] = 1
            # # Saving the CSV file with the existing network
            # df_Network_Mod.to_csv(   _path+'/2.Par/oT_Data_Network_'   +args.case+'.csv')

            ## Running the openStarNet
            oSN       = ConcreteModel()
            execution = 'Network_Line_Out_'+str(ni)+'_'+str(nf)+'_'+str(cc)

            df_Inp, df_Out = ModelRun(oSN, execution, _path, args.dir, args.case, args.solver, dictSets)

            df_input_data  = pd.concat([df_input_data,  df_Inp])
            df_output_data = pd.concat([df_output_data, df_Out])

            #%% Restoring the dataframes
            df_Network.to_csv(   _path+'/2.Par/oT_Data_Network_'   +args.case+'.csv')
            df_Generation.to_csv(_path+'/2.Par/oT_Data_Generation_'+args.case+'.csv')

    ####################################################################################################################

    df_input_data.to_csv( _path+'/3.Out/oT_Result_NN_Input_' +args.case+'.csv', index=True)
    df_output_data.to_csv(_path+'/3.Out/oT_Result_NN_Output_'+args.case+'.csv', index=True)

    # #%% Restoring the dataframes
    # df_Network.to_csv(   _path+'/2.Par/oT_Data_Network_'   +args.case+'.csv')
    # df_Generation.to_csv(_path+'/2.Par/oT_Data_Generation_'+args.case+'.csv')

    total_time = time.time() - initial_time
    print('########################################################')
    print('Total time                            ... ', round(total_time), 's')
    print('########################################################')

if __name__ == '__main__':
    main()