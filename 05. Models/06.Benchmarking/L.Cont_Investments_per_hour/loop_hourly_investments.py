import oSN_Main_DCOPF
import pyomo
import pandas as pd
import os
import time
import argparse
import re

def parse_args():
    parser = argparse.ArgumentParser(description='Introducing main parameters...')
    parser.add_argument('--case', type=str, default="3-bus")
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)

    args = parser.parse_args()
    case = args.case
    start = args.start
    end = args.end

    return case,start,end

def update_input_files_for_hour(LoadLevel):
    #Read the duration parameter file
    df_Duration = pd.read_csv(f"{case}/2.Par/oT_Data_Duration_{case}.csv",index_col="LoadLevel")
    #Make all durations 0
    df_Duration["Duration"] = 0
    #Change the duration of desired loadlevel to 1
    df_Duration.loc[LoadLevel,"Duration"] = 1
    #Write df back to file
    df_Duration.to_csv(f"{case}/2.Par/oT_Data_Duration_{case}.csv")

def create_model_from_input(case):
    operational_model = pyomo.environ.ConcreteModel(
        'openStarNet - Open Version of the StartNetLite Model (Long Term Transmission Expansion Planning) - Version 1.0.0 - January 16, 2023')
    operational_model.pLineXNetInv = 0

    oSN_Main_DCOPF.data_processing("", case, operational_model)
    oSN_Main_DCOPF.create_variables(operational_model, operational_model)
    oSN_Main_DCOPF.create_constraints(operational_model, operational_model)
    return operational_model

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

    StartTime = time.time()
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

    return model

def get_load_levels(case):
    df_Duration = pd.read_csv(f"{case}/2.Par/oT_Data_Duration_{case}.csv", index_col="LoadLevel")
    return df_Duration.index

def extract_investment_decisions(model,load_level):
    if len(model.plc):
        ivds = pd.Series(data=[model.vNetworkInvest[p, ni, nf, cc]() for p, ni, nf, cc in model.plc],
                                  index=pd.MultiIndex.from_tuples(model.plc))
        ivds_t = pd.DataFrame(ivds).transpose()
        ivds_t["LoadLevel"] = load_level
        ivds_t.set_index("LoadLevel",inplace=True)
    return ivds_t

def extract_cost(model,load_level):
    #%% outputting the generation operation
    SysCost     = pd.Series(data=[                                                                                                                           model.vTotalSCost()                                                                                                          ], index=[' ']  ).to_frame(name='Total          System Cost').stack()
    GenInvCost  = pd.Series(data=[model.pDiscountFactor[p] * sum(model.pGenInvestCost[gc  ]                                                                * model.vGenerationInvest[p,gc  ]()  for gc      in model.gc                                             )     for p in model.p], index=model.p).to_frame(name='Generation Investment Cost').stack()
    NetInvCost  = pd.Series(data=[model.pDiscountFactor[p] * sum(model.pNetFixedCost [lc  ]                                                                * model.vNetworkInvest   [p,lc  ]()  for lc      in model.lc                                             )     for p in model.p], index=model.p).to_frame(name='Network    Investment Cost').stack()
    GenCost     = pd.Series(data=[model.pDiscountFactor[p] * sum(model.pScenProb     [p,sc]()                                                              * model.vTotalGCost      [p,sc,n]()  for sc,n    in model.sc*model.n           if model.pScenProb[p,sc]())     for p in model.p], index=model.p).to_frame(name='Generation  Operation Cost').stack()
    ConCost     = pd.Series(data=[model.pDiscountFactor[p] * sum(model.pScenProb     [p,sc]()                                                              * model.vTotalCCost      [p,sc,n]()  for sc,n    in model.sc*model.n           if model.pScenProb[p,sc]())     for p in model.p], index=model.p).to_frame(name='Consumption Operation Cost').stack()
    EmiCost     = pd.Series(data=[model.pDiscountFactor[p] * sum(model.pScenProb     [p,sc]()                                                              * model.vTotalECost      [p,sc,n]()  for sc,n    in model.sc*model.n           if model.pScenProb[p,sc]())     for p in model.p], index=model.p).to_frame(name='Emission              Cost').stack()
    RelCost     = pd.Series(data=[model.pDiscountFactor[p] * sum(model.pScenProb     [p,sc]()                                                              * model.vTotalRCost      [p,sc,n]()  for sc,n    in model.sc*model.n           if model.pScenProb[p,sc]())     for p in model.p], index=model.p).to_frame(name='Reliability           Cost').stack()
    CostSummary = pd.concat([SysCost, GenInvCost, NetInvCost, GenCost, ConCost, EmiCost, RelCost])

    CostSummary = pd.DataFrame(CostSummary).transpose()
    CostSummary["LoadLevel"] = load_level
    CostSummary.set_index("LoadLevel", inplace=True)

    return CostSummary


def save_investments(df_results, case,type):
    df_results.to_csv(f"{case}/3.Out/oT_Hourly_{type}_{case}.csv")




if __name__ == '__main__':
    StartTime = time.time()
    #Start by parsing the arguments that define which case has to be solved
    case,start,end = parse_args()
    #Get the list of load_levels to be considered
    load_levels = get_load_levels(case)
    #Loop over the hours in a year
    df_investements = pd.DataFrame()
    df_costs = pd.DataFrame()
    hour_index=0
    if end == -1:
        end = len(load_levels)

    print("Running for " , len(load_levels[start:end]),"individual hours")
    for load_level in load_levels[start:end]:
        print(load_level)
        #Then update the input files in the correct way to consider a single hour
        update_input_files_for_hour(load_level)

        # Then create the model based on the input folder
        single_hour_model = create_model_from_input(case=case)

        #Solve the model
        single_hour_model = oSN_Main_DCOPF.solving_model(DirName="", CaseName=case, SolverName="gurobi",
                                                               optmodel=single_hour_model, pWriteLP=0)
        #Extract investment decisions:
        ivds = extract_investment_decisions(single_hour_model,load_level)
        cost = extract_cost(single_hour_model,load_level)

        df_investements = pd.concat([df_investements,ivds],axis=0)
        df_costs = pd.concat([df_costs,cost],axis=0)
        #Write every 100th hour
        if hour_index%100 == 0:
            for df_result,type in zip([df_costs,df_investements],["costs","Investments"]):
                save_investments(df_result,case,type)
    for df_result, type in zip([df_costs, df_investements], ["costs", "Investments"]):
        save_investments(df_result, case, type)
    EndTime = time.time()
    elapsed_time = round(time.time() - StartTime)
    print('Elapsed time: {} seconds'.format(elapsed_time))
    path_to_write_time = os.path.join(case,"3.Out/ComputationTime.txt")
    with open(path_to_write_time, 'w') as f:
         f.write(str(elapsed_time))