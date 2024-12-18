#%% Libraries
import argparse
import os
import itertools
import numpy         as np
import pandas        as pd
import time          # count clock time
from   collections   import defaultdict
from   pyomo.environ import ConcreteModel, Set, Param, Var, Objective, minimize, Constraint, DataPortal, PositiveIntegers, NonNegativeIntegers, Boolean, NonNegativeReals, UnitInterval, PositiveReals, Any, Binary, Reals, Suffix
from   oSN_Main_DCOPF   import *
import asyncio

def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped

#%% Defining metadata
parser = argparse.ArgumentParser(description='Introducing main parameters...')
parser.add_argument('--case',   type=str, default=None)
parser.add_argument('--dir',    type=str, default=None)
parser.add_argument('--solver', type=str, default=None)

DIR    = os.path.dirname(__file__)
CASE   = '3-bus'
SOLVER = 'gurobi'
folder_out = "DCOPF"

def ModelRun(model, optmodel, execution, path, dir, case, solver):

    # initializing the time counter
    start_time = time.time()
    # Defining the path to the model
    _path = path
    # Running the model
    pWrittingLPFile = 0
    model_p = solving_model(dir, case, solver, optmodel, pWrittingLPFile)

    reading_data_time = time.time() - start_time
    start_time = time.time()
    print('Reading    the parameters of the model ... ', round(reading_data_time), 's')

    # %% Saving the input data
    # Extracting the demand data
    df_demand = pd.Series(data=[model.pDemandP[p,sc,n,nd] for p,sc,n,nd in model.psnnd], index=pd.MultiIndex.from_tuples(model.psnnd))
    df_demand.index.names = ['Period', 'Scenario', 'LoadLevel', 'Variable']
    df_demand = df_demand.reset_index().pivot_table(index=['Period', 'Scenario', 'LoadLevel', 'Variable'], values=0)
    df_demand.rename(columns={0: 'Value'}, inplace=True)
    df_demand['Dataset'] = 'ElectricityDemand'
    df_demand['Execution'] = execution

    data_time = time.time() - start_time
    start_time = time.time()
    print('Getting the electricity demand         ... ', round(data_time), 's')

    # Extracting the maximum power generation data
    dict_techs = [tg for  tg     in model.gt  if tg in ['Hydro','Solar','Wind']]
    dict_gens  = [gg for (tg,gg) in model.t2g if tg in dict_techs]
    dict_gens.sort()
    List1 = [(p,sc,n,g) for p,sc,n in model.psn for g in dict_gens]
    df_max_power = pd.Series(data=[model.pMaxPower[p,sc,n,g] for p,sc,n,g in List1], index=pd.MultiIndex.from_tuples(List1))
    df_max_power.index.names = ['Period', 'Scenario', 'LoadLevel', 'Variable']
    df_max_power = df_max_power.reset_index().pivot_table(index=['Period', 'Scenario', 'LoadLevel', 'Variable'], values=0)
    df_max_power.rename(columns={0: 'Value'}, inplace=True)
    df_max_power['Dataset'  ] = 'MaxPowerGeneration'
    df_max_power['Execution'] = execution

    data_time = time.time() - start_time
    start_time = time.time()
    print('Getting the max power generation       ... ', round(data_time), 's')

    # Merging all the data
    df_input_data = pd.concat([df_demand, df_max_power])

    data_time = time.time() - start_time
    start_time = time.time()
    print('Getting the input data file            ... ', round(data_time), 's')

    # %% Saving the results
    # Total costs
    df_total_costs = pd.DataFrame(
        columns=['vTotalSCost', 'vTotalFCost', 'vTotalGCost', 'vTotalCCost', 'vTotalECost', 'vTotalRCost'],
        index=pd.MultiIndex.from_tuples(model.psn))
    df_total_costs.index.names = ['Period', 'Scenario', 'LoadLevel']

    for (p,sc,n) in model.psn:
        df_total_costs.loc[(p,sc,n), 'vTotalSCost'] =                                                      model_p.vTotalSCost()
        df_total_costs.loc[(p,sc,n), 'vTotalFCost'] = model.pDiscountFactor[p]                           * model_p.vTotalFCost[p]()
        df_total_costs.loc[(p,sc,n), 'vTotalGCost'] = model.pDiscountFactor[p] * model.pScenProb[p,sc]() * model_p.vTotalGCost[p,sc,n]()
        df_total_costs.loc[(p,sc,n), 'vTotalCCost'] = model.pDiscountFactor[p] * model.pScenProb[p,sc]() * model_p.vTotalCCost[p,sc,n]()
        df_total_costs.loc[(p,sc,n), 'vTotalECost'] = model.pDiscountFactor[p] * model.pScenProb[p,sc]() * model_p.vTotalECost[p,sc,n]()
        df_total_costs.loc[(p,sc,n), 'vTotalRCost'] = model.pDiscountFactor[p] * model.pScenProb[p,sc]() * model_p.vTotalRCost[p,sc,n]()

    df_total_costs = df_total_costs.stack().to_frame(name='Value')
    df_total_costs.index.names = ['Period', 'Scenario', 'LoadLevel', 'Variable']
    df_total_costs['Dataset'] = 'SystemCosts'
    df_total_costs['Execution'] = execution
    df_total_costs['Value'] = df_total_costs['Value']*1e3

    data_time = time.time() - start_time
    start_time = time.time()
    print('Getting the total costs                ... ', round(data_time), 's')

    # Power outputs
    df_power_output = pd.Series(data=[model_p.vTotalOutputP[p,sc,n,g]() for p,sc,n,g in model.psng], index=pd.MultiIndex.from_tuples(model.psng))
    df_power_output = df_power_output.to_frame(name='Value').rename_axis(['Period', 'Scenario', 'LoadLevel', 'Variable'], axis=0).reset_index().pivot_table(index=['Period', 'Scenario', 'LoadLevel','Variable'], values='Value', aggfunc=sum)
    df_power_output['Dataset'] = 'PowerOutput'
    df_power_output['Execution'] = execution

    data_time = time.time() - start_time
    start_time = time.time()
    print('Getting the power output               ... ', round(data_time), 's')

    # Power flows
    df_power_flow = pd.Series(data=[model_p.vFlow[p,sc,n,ni,nf,cc]() for p,sc,n,ni,nf,cc in model.psnla], index=pd.MultiIndex.from_tuples(model.psnla))
    df_power_flow = df_power_flow.to_frame(name='Value').rename_axis(['Period', 'Scenario', 'LoadLevel', 'InitialNode', 'FinalNode', 'Circuit'], axis=0).reset_index()
    df_power_flow['Variable'] = df_power_flow['InitialNode'] + '_' + df_power_flow['FinalNode'] + '_' + df_power_flow['Circuit']
    df_power_flow = df_power_flow.pivot_table(index=['Period','Scenario','LoadLevel','Variable'], values='Value' , aggfunc=sum)
    df_power_flow['Dataset']   = 'PowerFlow'
    df_power_flow['Execution'] = execution

    data_time = time.time() - start_time
    start_time = time.time()
    print('Getting the power flow                 ... ', round(data_time), 's')

    # # Dual eBalance
    # incoming and outgoing lines (lin) (lout)
    lin   = defaultdict(list)
    lout  = defaultdict(list)
    for ni,nf,cc in model.la:
        lin  [nf].append((ni,cc))
        lout [ni].append((nf,cc))

    List1 = [(p,sc,n,nd,st) for (p,sc,n,nd,st) in model.psnnd*model.st if (st,n) in model.s2n and sum(1 for g in model.g if (nd,g) in model.n2g) + sum(1 for lout in lout[nd]) + sum(1 for ni,cc in lin[nd])]
    df_dual_eBalance = pd.Series(data=[model_p.dual[model_p.eBalanceP[p,sc,st,n,nd]]*1e3 for p,sc,n,nd,st in List1], index=pd.MultiIndex.from_tuples(List1))
    df_dual_eBalance = df_dual_eBalance.to_frame(name='Value').rename_axis(['Period', 'Scenario', 'LoadLevel', 'Variable','Stage'], axis=0).reset_index().pivot_table(index=['Period','Scenario','LoadLevel','Variable'], values='Value' , aggfunc=sum)
    df_dual_eBalance['Dataset']   = 'Dual_eBalance'
    df_dual_eBalance['Execution'] = execution

    data_time = time.time() - start_time
    start_time = time.time()
    print('Getting the dual variable: eBalance    ... ', round(data_time), 's')

    # # Dual eLineCapacityFr_LP - from i to j
    # List2 = [(p,sc,st,n,ni,nf,cc) for p,sc,st,n,ni,nf,cc in model.ps*model.st*model.n*model.laa if (st,n) in model.s2n]
    # df_dual_eNetCapacity1 = pd.Series(data=[model_p.dual[model_p.eNetCapacityPfrUpperBound[p,sc,st,n,ni,nf,cc]]*1e3 for (p,sc,st,n,ni,nf,cc) in List2], index=pd.MultiIndex.from_tuples(List2))
    # df_dual_eNetCapacity1 = df_dual_eNetCapacity1.to_frame(name='Value').rename_axis(['Period','Scenario','Stage','LoadLevel','InitialNode','FinalNode','Circuit'], axis=0).reset_index()
    # df_dual_eNetCapacity1['Variable'] = df_dual_eNetCapacity1['InitialNode'] + '_' + df_dual_eNetCapacity1['FinalNode'] + '_' + df_dual_eNetCapacity1['Circuit']
    # df_dual_eNetCapacity1 = df_dual_eNetCapacity1.pivot_table(index=['Period','Scenario','LoadLevel','Variable'], values='Value' , aggfunc=sum)
    # df_dual_eNetCapacity1['Dataset']   = 'Dual_eNetCapacity_LowerBound'
    # df_dual_eNetCapacity1['Execution'] = execution
    #
    # data_time = time.time() - start_time
    # start_time = time.time()
    # print('Getting the dual variable: eNetCapacity1.. ', round(data_time), 's')
    #
    # # Dual eLineCapacityTo_LP - from j to i
    # df_dual_eNetCapacity2 = pd.Series(data=[model_p.dual[model_p.eLineCapacityTo_LP[p,sc,st,n,ni,nf,cc]]*1e3 for (p,sc,st,n,ni,nf,cc) in List2], index=pd.MultiIndex.from_tuples(List2))
    # df_dual_eNetCapacity2 = df_dual_eNetCapacity2.to_frame(name='Value').rename_axis(['Period','Scenario','Stage','LoadLevel','InitialNode','FinalNode','Circuit'], axis=0).reset_index()
    # df_dual_eNetCapacity2['Variable'] = df_dual_eNetCapacity2['InitialNode'] + '_' + df_dual_eNetCapacity2['FinalNode'] + '_' + df_dual_eNetCapacity2['Circuit']
    # df_dual_eNetCapacity2 = df_dual_eNetCapacity2.pivot_table(index=['Period','Scenario','LoadLevel','Variable'], values='Value' , aggfunc=sum)
    # df_dual_eNetCapacity2['Dataset']   = 'Dual_eNetCapacity_UpperBound'
    # df_dual_eNetCapacity2['Execution'] = execution
    #
    # data_time = time.time() - start_time
    # start_time = time.time()
    # print('Getting the dual variable: eNetCapacity2.. ', round(data_time), 's')

    # # Dual eGenCapacity1 - lower bound
    # List3 = [(p,sc,st,n,g) for p,sc,st,n,g in model.ps*model.st*model.n*model.g if (st,n) in model.s2n]
    # # df_dual_eGenCapacity1 = pd.Series(data=[model_p.dual[model_p.eGenCapacity1[p,sc,st,n,g]]*1e3 for (p,sc,st,n,g) in List3], index=pd.MultiIndex.from_tuples(List3))
    # # df_dual_eGenCapacity1 = df_dual_eGenCapacity1.to_frame(name='Value').rename_axis(['Period','Scenario','Stage','LoadLevel','Variable'], axis=0).pivot_table(index=['Period','Scenario','LoadLevel','Variable'], values='Value' , aggfunc=sum)
    # # df_dual_eGenCapacity1['Dataset'] = 'Dual_eGenCapacity_LowerBound'
    # # df_dual_eGenCapacity1['Execution'] = execution
    #
    # df_dual_eGenCapacity2 = pd.Series(data=[model_p.dual[model_p.eGenCapacity2[p,sc,st,n,g]]*1e3 for (p,sc,st,n,g) in List3], index=pd.MultiIndex.from_tuples(List3))
    # df_dual_eGenCapacity2 = df_dual_eGenCapacity2.to_frame(name='Value').rename_axis(['Period','Scenario','Stage','LoadLevel','Variable'], axis=0).pivot_table(index=['Period','Scenario','LoadLevel','Variable'], values='Value' , aggfunc=sum)
    # df_dual_eGenCapacity2['Dataset'] = 'Dual_eGenCapacity_UpperBound'
    # df_dual_eGenCapacity2['Execution'] = execution

    # data_time  = time.time() - start_time
    # start_time = time.time()
    # print('Getting the reduced cost: vTotalOutput ... ', round(data_time), 's')

    # Merging all the data
    # df_output_data = pd.concat([df_total_costs, df_power_output, df_power_flow, df_dual_eBalance, df_dual_eNetCapacity1, df_dual_eNetCapacity2, df_dual_eGenCapacity1, df_dual_eGenCapacity2])
    df_output_data = pd.concat([df_total_costs, df_power_output, df_power_flow, df_dual_eBalance])


    data_time = time.time() - start_time
    print('Getting the output data file           ... ', round(data_time), 's')

    return df_input_data, df_output_data


# Calling the main function
def main():
    args = parser.parse_args()
    # if args.dir is None:
    #     args.dir    = input('Input Dir    Name (Default {}): '.format(DIR))
    #     if args.dir == '':
    args.dir = DIR
    if args.case is None:
        args.case   = input('Input Case   Name (Default {}): '.format(CASE))
    if args.case == '':
        args.case = CASE
    # if args.solver is None:
    #     args.solver = input('Input Solver Name (Default {}): '.format(SOLVER))
    #     if args.solver == '':
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

    # create the base model
    base_model = ConcreteModel()

    # Activating the variable reactance
    base_model.pLineXNetInv = 0

    # Reading and data processing
    base_model = data_processing(args.dir, args.case, base_model)

    dict_lc  = [(  ni,nf,cc) for (  ni,nf,cc) in base_model.lc ]
    dict_le  = [(  ni,nf,cc) for (  ni,nf,cc) in base_model.le ]
    dict_la  = [(  ni,nf,cc) for (  ni,nf,cc) in base_model.la ]

    print('Number of lines in the network: ', len(base_model.la))

    #%% Sequence of the full network
    print("―――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――")
    print("Sequence of the existing network and generation")
    print("―――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――")

    # create the model
    oSN       = ConcreteModel()
    execution = 'Network_Existing_Generation_Full'

    # removing the sets
    base_model.del_component(base_model.la)
    base_model.del_component(base_model.lc)
    base_model.del_component(base_model.lca)
    base_model.del_component(base_model.laa)
    base_model.del_component(base_model.plc)
    base_model.del_component(base_model.psnla)

    # redefining the sets
    d_lc = []

    base_model.la    = Set(initialize=dict_le, ordered=True)
    base_model.lc    = Set(initialize=d_lc,    ordered=True)

    base_model.plc   = [(p,     ni,nf,cc) for p,     ni,nf,cc in base_model.p   * base_model.lc]
    base_model.psnla = [(p,sc,n,ni,nf,cc) for p,sc,n,ni,nf,cc in base_model.psn * base_model.la]

    # define AC candidate lines
    base_model.lca = Set(initialize=base_model.la, ordered=False, doc='AC candidate lines and     switchable lines', filter=lambda base_model, *lc: lc in base_model.lc and (lc, 'AC') in base_model.pLineType)
    base_model.laa = base_model.lea | base_model.lca

    # defining the variables
    oSN = create_variables(base_model, oSN)

    oSN.vNetworkInvest.pprint()

    # defining the constraints
    oSN = create_constraints(base_model, oSN)

    # calling the sequence for model solving and saving the results
    df_Inp, df_Out = ModelRun(base_model, oSN, execution, _path, args.dir, args.case, args.solver)

    df_input_data  = df_Inp
    df_output_data = df_Out


    # saving the input and output data
    df_Inp.to_csv(_path + f'/3.Out/{folder_out}/oT_Input_Data_'  + args.case + '_' + execution + '.csv')
    df_Out.to_csv(_path + f'/3.Out/{folder_out}/oT_Output_Data_' + args.case + '_' + execution + '.csv')

    ## restoring the candidate lines
    # removing the sets
    base_model.del_component(base_model.la)
    base_model.del_component(base_model.lc)
    base_model.del_component(base_model.lca)
    base_model.del_component(base_model.laa)
    base_model.del_component(base_model.plc)
    base_model.del_component(base_model.psnla)

    base_model.la    = Set(initialize=dict_la, ordered=True)
    base_model.lc    = Set(initialize=dict_lc, ordered=True)

    base_model.plc   = [(p,     ni,nf,cc) for p,     ni,nf,cc in base_model.p   * base_model.lc]
    base_model.psnla = [(p,sc,n,ni,nf,cc) for p,sc,n,ni,nf,cc in base_model.psn * base_model.la]

    # define AC candidate lines
    base_model.lca = Set(initialize=base_model.la, ordered=False, doc='AC candidate lines and     switchable lines', filter=lambda base_model, *lc: lc in base_model.lc and (lc, 'AC') in base_model.pLineType)
    base_model.laa = base_model.lea | base_model.lca

    clines = [(ni,nf,cc) for (ni,nf,cc) in base_model.la if base_model.pIndBinLineInvest[ni,nf,cc] == 1]
    print(f'Number of candidate lines to be considered: {len(clines)}')

    #%% Sequence of the full network
    print("―――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――")
    print("Sequence of the full network and generation")
    print("―――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――")

    # create the model
    oSN       = ConcreteModel()
    execution = 'Network_Full_Generation_Full'

    # defining the variables
    oSN = create_variables(base_model, oSN)

    # defining the constraints
    oSN = create_constraints(base_model, oSN)

    # fixing the investment variables
    for p,ni,nf,cc in base_model.plc:
        oSN.vNetworkInvest[p, ni, nf, cc].fix(1.0)

    # show the fixed variables
    oSN.vNetworkInvest.pprint()

    # calling the sequence for model solving and saving the results
    df_Inp, df_Out = ModelRun(base_model, oSN, execution, _path, args.dir, args.case, args.solver)

    df_input_data  = pd.concat([df_input_data, df_Inp])
    df_output_data = pd.concat([df_output_data, df_Out])

    # saving the input and output data
    df_Inp.to_csv(_path + f'/3.Out/{folder_out}/oT_Input_Data_'  + args.case + '_' + execution + '.csv')
    df_Out.to_csv(_path + f'/3.Out/{folder_out}/oT_Output_Data_' + args.case + '_' + execution + '.csv')

    clines = [(ni,nf,cc) for (ni,nf,cc) in base_model.la if base_model.pIndBinLineInvest[ni,nf,cc] == 1]
    print(f'Number of candidate lines to be considered: {len(clines)}')

    def solve_and_save_PINT(ni,nf,cc,df_input_data,df_output_data,counter1):
        print("――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――")
        print(f"PINT: Line {ni} {nf} {cc}")
        print("――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――")

        # Adding the line to the network
        elines = [(x, y, z) for (x, y, z) in base_model.le if base_model.pIndBinLineInvest[x, y, z] != 1]
        print(f'Number of existing lines to be considered: {len(elines)}')
        elines.append((ni, nf, cc))

        # removing the sets
        base_model.del_component(base_model.la)
        base_model.del_component(base_model.lc)
        base_model.del_component(base_model.lca)
        base_model.del_component(base_model.laa)
        base_model.del_component(base_model.plc)
        base_model.del_component(base_model.psnla)
        # redefining the sets
        d_lc = [(ni, nf, cc)]

        base_model.la = Set(initialize=elines, ordered=True)
        base_model.lc = Set(initialize=d_lc, ordered=True)

        base_model.plc = [(p, x, y, z) for p, x, y, z in base_model.p * base_model.lc]
        base_model.psnla = [(p, sc, n, x, y, z) for p, sc, n, x, y, z in base_model.psn * base_model.la]

        # define AC candidate lines
        base_model.lca = Set(initialize=base_model.la, ordered=False, doc='AC candidate lines and     switchable lines',
                             filter=lambda base_model, *lc: lc in base_model.lc and (lc, 'AC') in base_model.pLineType)
        base_model.laa = base_model.lea | base_model.lca

        # create the model
        oSN = ConcreteModel()
        execution = 'Network_Line_In_' + str(ni) + '_' + str(nf) + '_' + str(cc)

        # defining the variables
        oSN = create_variables(base_model, oSN)

        # defining the constraints
        oSN = create_constraints(base_model, oSN)

        # fixing the investment variables
        for p in base_model.p:
            oSN.vNetworkInvest[p, ni, nf, cc].fix(1.0)

        # showing the fixed variables
        oSN.vNetworkInvest.pprint()

        print(
            f'Number of lines to be considered: {len(base_model.le) + len([(p, x, y, z) for (p, x, y, z) in base_model.plc if oSN.vNetworkInvest[p, x, y, z]() == 1.0])}')

        df_Inp, df_Out = ModelRun(base_model, oSN, execution, _path, args.dir, args.case, args.solver)

        df_Inp.to_csv(_path + f'/3.Out/{folder_out}/oT_PINT_Input_Data_' + args.case + '_' + execution + '.csv')
        df_Out.to_csv(_path + f'/3.Out/{folder_out}/oT_PINT_Output_Data_' + args.case + '_' + execution + '.csv')

        df_input_data = pd.concat([df_input_data, df_Inp])
        df_output_data = pd.concat([df_output_data, df_Out])

        ## restoring the candidate lines
        # removing the sets
        base_model.del_component(base_model.la)
        base_model.del_component(base_model.lc)
        base_model.del_component(base_model.lca)
        base_model.del_component(base_model.laa)
        base_model.del_component(base_model.plc)
        base_model.del_component(base_model.psnla)

        base_model.la = Set(initialize=dict_la, ordered=True)
        base_model.lc = Set(initialize=dict_lc, ordered=True)

        base_model.plc = [(p, ni, nf, cc) for p, ni, nf, cc in base_model.p * base_model.lc]
        base_model.psnla = [(p, sc, n, ni, nf, cc) for p, sc, n, ni, nf, cc in base_model.psn * base_model.la]

        # define AC candidate lines
        base_model.lca = Set(initialize=base_model.la, ordered=False, doc='AC candidate lines and     switchable lines',
                             filter=lambda base_model, *lc: lc in base_model.lc and (lc, 'AC') in base_model.pLineType)
        base_model.laa = base_model.lea | base_model.lca

        counter1 += 1
        print("――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――")
        print(f'Remaining lines: {len(clines) - counter1}')
        print("――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――")

        ####################################################################################################################

    def solve_and_save_TOOT(ni,nf,cc,df_input_data,df_output_data,counter2):
        print("――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――")
        print(f"TOOT: Line {ni} {nf} {cc}")
        print("――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――")

        # # Adding the line to the network
        # elines = [(x, y, z) for (x, y, z) in base_model.la]
        # print(f'Number of lines to be considered: {len(elines)}')
        # elines.remove((ni, nf, cc))
        #
        # # removing the sets
        # base_model.del_component(base_model.la)
        # base_model.del_component(base_model.lc)
        # base_model.del_component(base_model.lca)
        # base_model.del_component(base_model.laa)
        # base_model.del_component(base_model.plc)
        # base_model.del_component(base_model.psnla)
        # # redefining the sets
        # d_lc = [(ni, nf, cc)]
        #
        # base_model.la = Set(initialize=elines, ordered=True)
        # base_model.lc = Set(initialize=d_lc, ordered=True)
        #
        # base_model.plc = [(p, x, y, z) for p, x, y, z in base_model.p * base_model.lc]
        # base_model.psnla = [(p, sc, n, x, y, z) for p, sc, n, x, y, z in base_model.psn * base_model.la]
        #
        # # define AC candidate lines
        # base_model.lca = Set(initialize=base_model.la, ordered=False, doc='AC candidate lines and     switchable lines',
        #                      filter=lambda base_model, *lc: lc in base_model.lc and (lc, 'AC') in base_model.pLineType)
        # base_model.laa = base_model.lea | base_model.lca

        # create the model
        oSN = ConcreteModel()
        execution = 'Network_Line_Out_' + str(ni) + '_' + str(nf) + '_' + str(cc)

        # defining the variables
        oSN = create_variables(base_model, oSN)

        # defining the constraints
        oSN = create_constraints(base_model, oSN)

        # fixing the investment variables
        for p,x,y,z in base_model.plc:
            if (x,y,z) == (ni,nf,cc):
                oSN.vNetworkInvest[p,x,y,z].fix(0.0)
            else:
                oSN.vNetworkInvest[p,x,y,z].fix(1.0)

        # showing the fixed variables
        oSN.vNetworkInvest.pprint()

        print(f'Number of lines to be considered: {len(base_model.la)}')

        df_Inp, df_Out = ModelRun(base_model, oSN, execution, _path, args.dir, args.case, args.solver)

        df_Inp.to_csv(_path + f'/3.Out/{folder_out}/oT_TOOT_Input_Data_' + args.case + '_' + execution + '.csv')
        df_Out.to_csv(_path + f'/3.Out/{folder_out}/oT_TOOT_Output_Data_' + args.case + '_' + execution + '.csv')

        df_input_data = pd.concat([df_input_data, df_Inp])
        df_output_data = pd.concat([df_output_data, df_Out])

        # ## restoring the candidate lines
        # # removing the sets
        # base_model.del_component(base_model.la)
        # base_model.del_component(base_model.lc)
        # base_model.del_component(base_model.lca)
        # base_model.del_component(base_model.laa)
        # base_model.del_component(base_model.plc)
        # base_model.del_component(base_model.psnla)
        #
        # base_model.la = Set(initialize=dict_la, ordered=True)
        # base_model.lc = Set(initialize=dict_lc, ordered=True)
        #
        # base_model.plc = [(p, ni, nf, cc) for p, ni, nf, cc in base_model.p * base_model.lc]
        # base_model.psnla = [(p, sc, n, ni, nf, cc) for p, sc, n, ni, nf, cc in base_model.psn * base_model.la]
        #
        # # define AC candidate lines
        # base_model.lca = Set(initialize=base_model.la, ordered=False, doc='AC candidate lines and     switchable lines',
        #                      filter=lambda base_model, *lc: lc in base_model.lc and (lc, 'AC') in base_model.pLineType)
        # base_model.laa = base_model.lea | base_model.lca

        counter2 += 1
        print("――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――")
        print(f'Remaining lines: {len(clines) - counter2}')
        print("――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――")

        ####################################################################################################################

    df_input_data.to_csv(_path + f'/3.Out/{folder_out}/oT_Result_NN_Input_' + args.case + '.csv', index=True)
    df_output_data.to_csv(_path + f'/3.Out/{folder_out}/oT_Result_NN_Output_' + args.case + '.csv', index=True)

    # #%% Restoring the dataframes
    # df_Network.to_csv(   _path+'/2.Par/oT_Data_Network_'   +args.case+'.csv')
    # df_Generation.to_csv(_path+'/2.Par/oT_Data_Generation_'+args.case+'.csv')

    total_time = time.time() - initial_time
    print('########################################################')
    print('Total time                            ... ', round(total_time), 's')
    print('########################################################')
    counter1 = 0
    counter2 = 0
    for (ni,nf,cc) in clines:
        solve_and_save_PINT(ni,nf,cc,df_input_data,df_output_data,counter1)
        solve_and_save_TOOT(ni,nf,cc,df_input_data,df_output_data,counter2)
        counter1 += 1
        counter2 += 1


if __name__ == '__main__':
    t_start = time.time()
    main()
    total_time = time.time() - t_start
    print('########################################################')
    print('Total time                            ... ', round(total_time), 's')
    print('########################################################')
