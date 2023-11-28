import oSN_Main_operational
import pyomo
import pandas as pd


def find_pu_value_from_inv_results_df(df_investment_results, node_from, node_to, cac):
    return df_investment_results.set_index(["InitialNode", "FinalNode", "Circuit"]).loc[node_from, node_to, cac]["p.u."]


# Start by creating the model a single time based on the input folder

operational_model = pyomo.environ.ConcreteModel(
    'openStarNet - Open Version of the StartNetLite Model (Long Term Transmission Expansion Planning) - Version 1.0.0 - January 16, 2023')
operational_model.pLineXNetInv = 0

oSN_Main_operational.data_processing("", "3-bus", operational_model)
oSN_Main_operational.create_variables(operational_model, operational_model)
oSN_Main_operational.create_constraints(operational_model, operational_model)

# Then, read the investment decisions obatined in the by_stages runs


folder = "D.Representative_days_based_on_RES_and_Demand"
case = "3-bus"
nb_stages = 10
origin_folder = f"../{folder}/{case}_ByStages_nc{nb_stages}/3.Out/oT_Result_NetworkInvestment_{case}_ByStages_nc{nb_stages}.csv"
df_investment_results = pd.read_csv(origin_folder)

idx = 0
for var in operational_model.component_data_objects(pyomo.environ.Var, active=True, descend_into=True):
    if not var.is_continuous():
        node_from,node_to,cac = var.index()[1],var.index()[2],var.index()[3]
        value_to_fix = find_pu_value_from_inv_results_df(df_investment_results,node_from=node_from,node_to=node_to,cac=cac)
        #var.fixed = True  # fix the current value
        print("fixing: " + str(var) + "to " + str(value_to_fix))
        var.fix(value=value_to_fix)
        idx += 1

oSN_Main_operational.solving_model(DirName="", CaseName=case, SolverName="gurobi", optmodel=operational_model, pWriteLP=0)

SysCost = pd.Series(data=[operational_model.vTotalSCost()], index=[' ']).to_frame(name='Total          System Cost').stack()
print(SysCost)

OutputResults = pd.Series(data=[operational_model.vGenerationInvest[p, gc]() for p, gc in operational_model.pgc],
                          index=pd.MultiIndex.from_tuples(operational_model.pgc))
OutputResults.to_frame(name='p.u.').rename_axis(['Period', 'Unit'], axis=0).reset_index()
print(OutputResults)

