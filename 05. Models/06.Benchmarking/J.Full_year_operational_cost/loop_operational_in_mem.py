from   collections       import defaultdict
import oSN_Main_operational
import pyomo
import pandas as pd
import os
import time
import argparse
import re

def parse_args():
    folder_map = {"B":os.path.join("..","B.Operation_cost"),
                  "D":os.path.join("..","D.Representative_days_based_on_RES_and_Demand"),
                  "E":os.path.join("..","E.Representative_days_based_on_Line_Benefits_OptModel"),
                  "F":os.path.join("..","F.Representative_days_based_on_Line_Benefit_NN_OC_fy_1"),
                  "G":os.path.join("..","G.Representative_days_based_on_Line_Benefit_NN_OC_fy_2"),
                  "K":os.path.join("..","K.Investments_per_hour"),
                  "L": os.path.join("..", "L.Cont_Investments_per_hour"),
                  }
    parser = argparse.ArgumentParser(description='Introducing main parameters...')
    parser.add_argument('--case', type=str, default="3-bus")
    parser.add_argument('--min_nb', type=int, default=None, help='Minimum value for the integer following "nc"')
    parser.add_argument('--max_nb', type=int, default=None, help='Maximum value for the integer following "nc"')
    parser.add_argument('--origin_folder', type=str, default=None, help='Capital letter defining folder from which to fetch investment decisions')

    args = parser.parse_args()

    case = args.case
    min_nb = args.min_nb
    max_nb = args.max_nb
    origin_folder = folder_map[args.origin_folder]

    return case,min_nb,max_nb,origin_folder

def create_model_from_input(case):
    operational_model = pyomo.environ.ConcreteModel(
        'openStarNet - Open Version of the StartNetLite Model (Long Term Transmission Expansion Planning) - Version 1.0.0 - January 16, 2023')
    operational_model.pLineXNetInv = 0

    oSN_Main_operational.data_processing("", case, operational_model)
    oSN_Main_operational.create_variables(operational_model, operational_model)
    oSN_Main_operational.create_constraints(operational_model, operational_model)
    return operational_model

def find_pu_value_from_inv_results_df(df_investment_results, node_from, node_to, cac):
    return df_investment_results.set_index(["InitialNode", "FinalNode", "Circuit"]).loc[node_from, node_to, cac]["p.u."]

def fix_investment_variables_based_on_file_reading(operational_model,df_investment_results):
    idx = 0
    for var in operational_model.component_data_objects(pyomo.environ.Var, active=True, descend_into=True):
        if not var.is_continuous():
            node_from,node_to,cac = var.index()[1],var.index()[2],var.index()[3]
            value_to_fix = find_pu_value_from_inv_results_df(df_investment_results,node_from=node_from,node_to=node_to,cac=cac)
            #var.fixed = True  # fix the current value
            #print("fixing: " + str(var) + "to " + str(value_to_fix))
            var.fix(value=value_to_fix)
            idx += 1

def saving_results(DirName, CaseName, model, optmodel):

    _path = os.path.join(DirName, CaseName)
    StartTime = time.time()
    print('Objective function value                  ', model.eTotalSCost.expr())

    #%% outputting the generation operation
    SysCost     = pd.Series(data=[                                                                                                                           optmodel.vTotalSCost()                                                                                                          ], index=[' ']  ).to_frame(name='Total          System Cost').stack()
    GenInvCost  = pd.Series(data=[model.pDiscountFactor[p] * sum(model.pGenInvestCost[gc  ]                                                                * optmodel.vGenerationInvest[p,gc  ]()  for gc      in model.gc                                             )     for p in model.p], index=model.p).to_frame(name='Generation Investment Cost').stack()
    NetInvCost  = pd.Series(data=[model.pDiscountFactor[p] * sum(model.pNetFixedCost [lc  ]                                                                * optmodel.vNetworkInvest   [p,lc  ]()  for lc      in model.lc                                             )     for p in model.p], index=model.p).to_frame(name='Network    Investment Cost').stack()
    GenCost     = pd.Series(data=[model.pDiscountFactor[p] * sum(model.pScenProb     [p,sc]()                                                              * optmodel.vTotalGCost      [p,sc,n]()  for sc,n    in model.sc*model.n           if model.pScenProb[p,sc]())     for p in model.p], index=model.p).to_frame(name='Generation  Operation Cost').stack()
    ConCost     = pd.Series(data=[model.pDiscountFactor[p] * sum(model.pScenProb     [p,sc]()                                                              * optmodel.vTotalCCost      [p,sc,n]()  for sc,n    in model.sc*model.n           if model.pScenProb[p,sc]())     for p in model.p], index=model.p).to_frame(name='Consumption Operation Cost').stack()
    EmiCost     = pd.Series(data=[model.pDiscountFactor[p] * sum(model.pScenProb     [p,sc]()                                                              * optmodel.vTotalECost      [p,sc,n]()  for sc,n    in model.sc*model.n           if model.pScenProb[p,sc]())     for p in model.p], index=model.p).to_frame(name='Emission              Cost').stack()
    RelCost     = pd.Series(data=[model.pDiscountFactor[p] * sum(model.pScenProb     [p,sc]()                                                              * optmodel.vTotalRCost      [p,sc,n]()  for sc,n    in model.sc*model.n           if model.pScenProb[p,sc]())     for p in model.p], index=model.p).to_frame(name='Reliability           Cost').stack()
    CostSummary = pd.concat([SysCost, GenInvCost, NetInvCost, GenCost, ConCost, EmiCost, RelCost])
    CostSummary = CostSummary.reset_index().rename(columns={'level_0': 'Period', 'level_1': 'Cost/Payment', 0: 'MEUR'})



    output_directory = _path + '/3.Out/'
    # Check if the directory exists, and create it if not
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Now you can save the CSV file
    CostSummary.to_csv(output_directory + 'oT_Result_CostSummary_' + CaseName + '.csv', sep=',', index=False)
    WritingCostSummaryTime = time.time() - StartTime
    StartTime              = time.time()
    print('Writing         cost summary results  ... ', round(WritingCostSummaryTime), 's')

    # %% outputting the investments
    if len(model.pgc):
        OutputResults = pd.Series(data=[optmodel.vGenerationInvest[p, gc]() for p, gc in model.pgc],
                                  index=pd.MultiIndex.from_tuples(model.pgc))
        OutputResults.to_frame(name='p.u.').rename_axis(['Period', 'Unit'], axis=0).reset_index().to_csv(
            _path + '/3.Out/oT_Result_GenerationInvestment_' + CaseName + '.csv', index=False, sep=',')

    if len(model.plc):
        OutputResults = pd.Series(data=[optmodel.vNetworkInvest[p, ni, nf, cc]() for p, ni, nf, cc in model.plc],
                                  index=pd.MultiIndex.from_tuples(model.plc))
        OutputResults.to_frame(name='p.u.').rename_axis(['Period', 'InitialNode', 'FinalNode', 'Circuit'],
                                                        axis=0).reset_index().to_csv(
            _path + '/3.Out/oT_Result_NetworkInvestment_' + CaseName + '.csv', index=False, sep=',')

    WritingInvResultsTime = time.time() - StartTime
    StartTime = time.time()
    print('Writing           investment results  ... ', round(WritingInvResultsTime), 's')

    if len(model.r):
        OutputResults = pd.Series(data=[(model.pMaxPower[p,sc,n,g]-optmodel.vTotalOutputP[p,sc,n,g]())*1e3 for p,sc,n,g in model.psnr], index=pd.MultiIndex.from_tuples(model.psnr))
        OutputResults.to_frame(name='MW'  ).rename_axis(['Period','Scenario','LoadLevel','Unit'], axis=0).reset_index().to_csv(_path+'/3.Out/oT_Result_RESCurtailment_'      +CaseName+'.csv', index=False, sep=',')

    pEpsilon = 1e-6
    OutputResults = pd.Series(data=[max(optmodel.vFlow[p,sc,n,ni,nf,cc]()/(model.pLineNTCFrw[ni,nf,cc]+pEpsilon),-optmodel.vFlow[p,sc,n,ni,nf,cc]()/(model.pLineNTCFrw[ni,nf,cc]+pEpsilon)) for p,sc,n,ni,nf,cc in model.psnla], index=pd.Index(model.psnla))
    OutputResults.to_frame(name='GWh').rename_axis(['Period', 'Scenario', 'LoadLevel', 'InitialNode', 'FinalNode', 'Circuit'], axis=0).reset_index().to_csv(_path+'/3.Out/oT_Result_NetworkUtilizationPerNode_DC_'+CaseName+'.csv', index=False, sep=',')

    #%% outputting the generation cost
    OutputResults = pd.Series(data=[model.pDiscountFactor[p]*model.pScenProb[p,sc]()*model.pLoadLevelDuration[n]()*(optmodel.vTotalGCost[p,sc,n]()+optmodel.vTotalCCost[p,sc,n]()+optmodel.vTotalECost[p,sc,n]()+optmodel.vTotalRCost[p,sc,n]())*1e3 for p,sc,n in model.psn], index=pd.MultiIndex.from_tuples(model.psn))
    OutputResults.to_frame(name='mEUR').rename_axis(['Period','Scenario','LoadLevel'], axis=0).reset_index().to_csv(_path+'/3.Out/oT_Result_GenerationCost_'+CaseName+'.csv', index=False, sep=',')

    #%% outputting the ENS
    OutputResults = pd.Series(data=[model.pDiscountFactor[p]*model.pScenProb[p,sc]()*model.pLoadLevelDuration[n]()*optmodel.vENS[p,sc,n,nd]()*model.pDemandP[p,sc,n,nd]*1e3  for p,sc,n,nd in model.psnnd], index=pd.MultiIndex.from_tuples(model.psnnd))
    OutputResults.to_frame(name='MWh').rename_axis(['Period','Scenario','LoadLevel','Node'], axis=0).reset_index().to_csv(_path+'/3.Out/oT_Result_EnergyNotSupplied_'+CaseName+'.csv', index=False, sep=',')

    # Determination of the net demand
    r2r = defaultdict(list)
    for rt,re in model.rt*model.r:
        if (rt,re) in model.t2g:
            r2r[rt].append(re)

    if len(model.r):
        OutputToFile1 = pd.Series(data=[sum(optmodel.vTotalOutputP[p,sc,n,re]() for rt in model.rt for re in r2r[rt] if (p,re) in model.pre and (nd,re) in model.n2g) for p,sc,n,nd in model.psnnd], index=pd.Index(model.psnnd))
    else:
        OutputToFile1 = pd.Series(data=[0.0                                                                               for p,sc,n,nd in model.psnnd], index=pd.Index(model.psnnd))
    OutputToFile2 = pd.Series(data=[          model.pDemandP     [p,sc,n,nd]                                              for p,sc,n,nd in model.psnnd], index=pd.Index(model.psnnd))
    OutputToFile  = OutputToFile2 - OutputToFile1
    OutputToFile  *= 1e3
    OutputToFile  = OutputToFile.to_frame(name='MW'  )
    OutputToFile.reset_index().pivot_table(index=['level_0','level_1','level_2'], columns='level_3', values='MW', aggfunc='sum').rename_axis(['Period', 'Scenario', 'LoadLevel'], axis=0).rename_axis([None], axis=1).to_csv(_path+'/3.Out/oT_Result_NetworkNetDemand_'+CaseName+'.csv', sep=',')
    OutputToFile.reset_index().pivot_table(index=['level_0','level_1','level_2'],                    values='MW', aggfunc='sum').rename_axis(['Period', 'Scenario', 'LoadLevel'], axis=0).rename_axis([None], axis=1).to_csv(_path+'/3.Out/oT_Result_NetDemand_'       +CaseName+'.csv', sep=',')

    WritingOperatingResultsTime = time.time() - StartTime
    StartTime = time.time()
    print('Writing         operating results  ... ', round(WritingOperatingResultsTime), 's')

    return model

def obtain_list_of_ByStages_numbers(path_to_scan,min_nb,max_nb):
    #path_to_scan = os.path.dirname("../D.Representative_days_based_on_RES_and_Demand/")

    filter = f'{case}_ByStages_nc'

    nb_stages_l = []

    for f in os.scandir(path_to_scan):
        if f.is_dir():
            basename = os.path.basename(f.path)
            match = re.search(f'{filter}(\d+)', basename)  # Corrected the regular expression here
            if match:
                nc_number = int(match.group(1))
                if (min_nb is None or nc_number >= min_nb) and (max_nb is None or nc_number <= max_nb):
                    nb_stages_l.append(nc_number)

    nb_stages_l = sorted(nb_stages_l)
    #print("Found following ByStages results in folder {}")
    return nb_stages_l


if __name__ == '__main__':
    #Start by parsing the arguments that define which cases have to be solved
    case, min_nb, max_nb, origin_master = parse_args()

    # Then create the model a single time based on the input folder
    operational_model = create_model_from_input(case=case)

    #Next, obtain a list of the cases that need to be run based on the origin folder and the min_nb, and max_nb arguments
    list_nb_stages = obtain_list_of_ByStages_numbers(path_to_scan=origin_master,min_nb=min_nb,max_nb=max_nb)

    # print(list_nb_stages)
    for nb_stages in list_nb_stages:
        # Then fix the investment variables based on the results read in the specified folder

        origin_folder = f"{origin_master}/{case}_ByStages_nc{nb_stages}/3.Out/oT_Result_NetworkInvestment_{case}_ByStages_nc{nb_stages}.csv"
        print("Reading investment decisions from: " + origin_folder)
        df_investment_results = pd.read_csv(origin_folder)
        fix_investment_variables_based_on_file_reading(operational_model=operational_model,df_investment_results=df_investment_results)


        operational_model = oSN_Main_operational.solving_model(DirName="", CaseName=case, SolverName="gurobi",
                                                               optmodel=operational_model, pWriteLP=0)
        case_name_bs = f"{case}_ByStages_nc{nb_stages}"
        save_dir = os.path.join("Results",os.path.split(origin_master)[1])
        print(save_dir)
        saving_results(DirName=save_dir,CaseName=case_name_bs,model=operational_model,optmodel=operational_model)