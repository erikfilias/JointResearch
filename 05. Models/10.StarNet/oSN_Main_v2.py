#                     GNU GENERAL PUBLIC LICENSE
#                        Version 3, 29 June 2007
#
#  Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
#  Everyone is permitted to copy and distribute verbatim copies
#  of this license document, but changing it is not allowed.
#
#                             Preamble
#
#   The GNU General Public License is a free, copyleft license for
# software and other kinds of works.
#
#   The licenses for most software and other practical works are designed
# to take away your freedom to share and change the works.  By contrast,
# the GNU General Public License is intended to guarantee your freedom to
# share and change all versions of a program--to make sure it remains free
# software for all its users.  We, the Free Software Foundation, use the
# GNU General Public License for most of our software; it applies also to
# any other work released this way by its authors.  You can apply it to
# your programs, too.
#
#   When we speak of free software, we are referring to freedom, not
# price.  Our General Public Licenses are designed to make sure that you
# have the freedom to distribute copies of free software (and charge for
# them if you wish), that you receive source code or can get it if you
# want it, that you can change the software or use pieces of it in new
# free programs, and that you know you can do these things.
#
#   To protect your rights, we need to prevent others from denying you
# these rights or asking you to surrender the rights.  Therefore, you have
# certain responsibilities if you distribute copies of the software, or if
# you modify it: responsibilities to respect the freedom of others.
#
#   For example, if you distribute copies of such a program, whether
# gratis or for a fee, you must pass on to the recipients the same
# freedoms that you received.  You must make sure that they, too, receive
# or can get the source code.  And you must show them these terms so they
# know their rights.
#
#   Developers that use the GNU GPL protect your rights with two steps:
# (1) assert copyright on the software, and (2) offer you this License
# giving you legal permission to copy, distribute and/or modify it.
#
#   For the developers' and authors' protection, the GPL clearly explains
# that there is no warranty for this free software.  For both users' and
# authors' sake, the GPL requires that modified versions be marked as
# changed, so that their problems will not be attributed erroneously to
# authors of previous versions.
#
#   Some devices are designed to deny users access to install or run
# modified versions of the software inside them, although the manufacturer
# can do so.  This is fundamentally incompatible with the aim of
# protecting users' freedom to change the software.  The systematic
# pattern of such abuse occurs in the area of products for individuals to
# use, which is precisely where it is most unacceptable.  Therefore, we
# have designed this version of the GPL to prohibit the practice for those
# products.  If such problems arise substantially in other domains, we
# stand ready to extend this provision to those domains in future versions
# of the GPL, as needed to protect the freedom of users.
#
#   Finally, every program is threatened constantly by software patents.
# States should not allow patents to restrict development and use of
# software on general-purpose computers, but in those that do, we wish to
# avoid the special danger that patents applied to a free program could
# make it effectively proprietary.  To prevent this, the GPL assures that
# patents cannot be used to render the program non-free.

# StarNet - Version 1.0.0 - January 13, 2023
# simplicity and transparency in power systems planning

# Developed by

#    Andres Ramos, Erik Alvarez
#    Instituto de Investigacion Tecnologica
#    Escuela Tecnica Superior de Ingenieria - ICAI
#    UNIVERSIDAD PONTIFICIA COMILLAS
#    Alberto Aguilera 23
#    28015 Madrid, Spain
#    Andres.Ramos@comillas.edu
#    Erik.Alvarez@comillas.edu
#    https://pascua.iit.comillas.edu/aramos/Ramos_CV.html

#%% Libraries
import argparse
import os
import pandas        as pd
import time          # count clock time
import math          # access to math operations
import psutil        # access the number of CPUs
from   collections   import defaultdict
from   pyomo.environ import ConcreteModel, RangeSet, Set, Param, Var, Objective, minimize, Constraint, DataPortal, PositiveIntegers, NonNegativeIntegers, Boolean, NonNegativeReals, UnitInterval, PositiveReals, Any, Binary, Reals, Suffix
from   pyomo.opt     import SolverFactory

print('\n #### Academic research license - for non-commercial use only #### \n')

StartTime = time.time()

parser = argparse.ArgumentParser(description='Introducing main parameters...')
parser.add_argument('--case',   type=str, default=None)
parser.add_argument('--dir',    type=str, default=None)
parser.add_argument('--solver', type=str, default=None)

DIR    = os.path.dirname(__file__)
CASE   = '9n'
SOLVER = 'gurobi'

# %% model declaration
openStarNet = ConcreteModel('openStarNet - Open Version of the StartNetLite Model (Long Term Transmission Expansion Planning) - Version 1.0.0 - January 16, 2023')

def main(model):
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
    openStarNet_run(args.dir, args.case, args.solver, model)

def openStarNet_run(DirName, CaseName, SolverName, model):

    _path = os.path.join(DirName, CaseName)
    StartTime = time.time()

    #%% Reactance variable according to the investment decision
    pLineXNetInv = 1

    #%% reading the sets
    dictSets = DataPortal()
    dictSets.load(filename=_path+'/1.Set/oT_Dict_Period_'      +CaseName+'.csv', set='p'   , format='set')
    dictSets.load(filename=_path+'/1.Set/oT_Dict_Scenario_'    +CaseName+'.csv', set='sc'  , format='set')
    dictSets.load(filename=_path+'/1.Set/oT_Dict_Stage_'       +CaseName+'.csv', set='st'  , format='set')
    dictSets.load(filename=_path+'/1.Set/oT_Dict_LoadLevel_'   +CaseName+'.csv', set='n'   , format='set')
    dictSets.load(filename=_path+'/1.Set/oT_Dict_Generation_'  +CaseName+'.csv', set='g'   , format='set')
    dictSets.load(filename=_path+'/1.Set/oT_Dict_Technology_'  +CaseName+'.csv', set='gt'  , format='set')
    dictSets.load(filename=_path+'/1.Set/oT_Dict_Storage_'     +CaseName+'.csv', set='et'  , format='set')
    dictSets.load(filename=_path+'/1.Set/oT_Dict_Node_'        +CaseName+'.csv', set='nd'  , format='set')
    dictSets.load(filename=_path+'/1.Set/oT_Dict_Zone_'        +CaseName+'.csv', set='zn'  , format='set')
    dictSets.load(filename=_path+'/1.Set/oT_Dict_Area_'        +CaseName+'.csv', set='ar'  , format='set')
    dictSets.load(filename=_path+'/1.Set/oT_Dict_Region_'      +CaseName+'.csv', set='rg'  , format='set')
    dictSets.load(filename=_path+'/1.Set/oT_Dict_Circuit_'     +CaseName+'.csv', set='cc'  , format='set')
    dictSets.load(filename=_path+'/1.Set/oT_Dict_Line_'        +CaseName+'.csv', set='lt'  , format='set')
    dictSets.load(filename=_path+'/1.Set/oT_Dict_BusShunt_'    +CaseName+'.csv', set='shh' , format='set')

    dictSets.load(filename=_path+'/1.Set/oT_Dict_NodeToZone_'  +CaseName+'.csv', set='ndzn', format='set')
    dictSets.load(filename=_path+'/1.Set/oT_Dict_ZoneToArea_'  +CaseName+'.csv', set='znar', format='set')
    dictSets.load(filename=_path+'/1.Set/oT_Dict_AreaToRegion_'+CaseName+'.csv', set='arrg', format='set')

    model.pp   = Set(initialize=dictSets['p'  ],   ordered=True,  doc='periods',         within=PositiveIntegers)
    model.scc  = Set(initialize=dictSets['sc' ],   ordered=True,  doc='scenarios'       )
    model.stt  = Set(initialize=dictSets['st' ],   ordered=True,  doc='stages'          )
    model.nn   = Set(initialize=dictSets['n'  ],   ordered=True,  doc='load levels'     )
    model.gg   = Set(initialize=dictSets['g'  ],   ordered=False, doc='units'           )
    model.gt   = Set(initialize=dictSets['gt' ],   ordered=False, doc='technologies'    )
    model.et   = Set(initialize=dictSets['et' ],   ordered=False, doc='ESS types'       )
    model.nd   = Set(initialize=dictSets['nd' ],   ordered=False, doc='nodes'           )
    model.ni   = Set(initialize=dictSets['nd' ],   ordered=False, doc='nodes'           )
    model.nf   = Set(initialize=dictSets['nd' ],   ordered=False, doc='nodes'           )
    model.zn   = Set(initialize=dictSets['zn' ],   ordered=False, doc='zones'           )
    model.ar   = Set(initialize=dictSets['ar' ],   ordered=False, doc='areas'           )
    model.rg   = Set(initialize=dictSets['rg' ],   ordered=False, doc='regions'         )
    model.cc   = Set(initialize=dictSets['cc' ],   ordered=False, doc='circuits'        )
    model.c2   = Set(initialize=dictSets['cc' ],   ordered=False, doc='circuits'        )
    model.lt   = Set(initialize=dictSets['lt' ],   ordered=False, doc='line types'      )
    model.shh  = Set(initialize=dictSets['shh'],   ordered=False, doc='bus shunts'      )

    model.ndzn = Set(initialize=dictSets['ndzn'], ordered=False, doc='node to zone'    )
    model.znar = Set(initialize=dictSets['znar'], ordered=False, doc='zone to area'    )
    model.arrg = Set(initialize=dictSets['arrg'], ordered=False, doc='area to region'  )

    #%% reading data from CSV files
    dfOption             = pd.read_csv(_path+'/2.Par/oT_Data_Option_'                +CaseName+'.csv', index_col=[0    ])
    dfParameter          = pd.read_csv(_path+'/2.Par/oT_Data_Parameter_'             +CaseName+'.csv', index_col=[0    ])
    dfPeriod             = pd.read_csv(_path+'/2.Par/oT_Data_Period_'                +CaseName+'.csv', index_col=[0    ])
    dfScenario           = pd.read_csv(_path+'/2.Par/oT_Data_Scenario_'              +CaseName+'.csv', index_col=[0,1  ])
    dfStage              = pd.read_csv(_path+'/2.Par/oT_Data_Stage_'                 +CaseName+'.csv', index_col=[0    ])
    dfDuration           = pd.read_csv(_path+'/2.Par/oT_Data_Duration_'              +CaseName+'.csv', index_col=[0    ])
    dfReserveMargin      = pd.read_csv(_path+'/2.Par/oT_Data_ReserveMargin_'         +CaseName+'.csv', index_col=[0    ])
    dfDemandP            = pd.read_csv(_path+'/2.Par/oT_Data_Demand_'                +CaseName+'.csv', index_col=[0,1,2])
    dfDemandQ            = pd.read_csv(_path+'/2.Par/oT_Data_ReactiveDemand_'        +CaseName+'.csv', index_col=[0,1,2])
    dfGeneration         = pd.read_csv(_path+'/2.Par/oT_Data_Generation_'            +CaseName+'.csv', index_col=[0    ])
    dfVariableMaxPower   = pd.read_csv(_path+'/2.Par/oT_Data_VariableMaxGeneration_' +CaseName+'.csv', index_col=[0,1,2])
    dfEnergyInflows      = pd.read_csv(_path+'/2.Par/oT_Data_EnergyInflows_'         +CaseName+'.csv', index_col=[0,1,2])
    dfEnergyOutflows     = pd.read_csv(_path+'/2.Par/oT_Data_EnergyOutflows_'        +CaseName+'.csv', index_col=[0,1,2])
    dfNodeLocation       = pd.read_csv(_path+'/2.Par/oT_Data_NodeLocation_'          +CaseName+'.csv', index_col=[0    ])
    dfNetwork            = pd.read_csv(_path+'/2.Par/oT_Data_Network_'               +CaseName+'.csv', index_col=[0,1,2])
    dfBusShunt           = pd.read_csv(_path+'/2.Par/oT_Data_BusShunt_'              +CaseName+'.csv', index_col=[0    ])

    # substitute NaN by 0
    dfOption.fillna            (0  , inplace=True)
    dfParameter.fillna         (0.0, inplace=True)
    dfPeriod.fillna            (0.0, inplace=True)
    dfScenario.fillna          (0.0, inplace=True)
    dfStage.fillna             (0.0, inplace=True)
    dfDuration.fillna          (0  , inplace=True)
    dfReserveMargin.fillna     (0.0, inplace=True)
    dfDemandP.fillna           (0.0, inplace=True)
    dfDemandQ.fillna           (0.0, inplace=True)
    dfGeneration.fillna        (0.0, inplace=True)
    dfVariableMaxPower.fillna  (0.0, inplace=True)
    dfEnergyInflows.fillna     (0.0, inplace=True)
    dfEnergyOutflows.fillna    (0.0, inplace=True)
    dfNodeLocation.fillna      (0.0, inplace=True)
    dfNetwork.fillna           (0.0, inplace=True)
    dfBusShunt.fillna          (0.0, inplace=True)

    dfReserveMargin      = dfReserveMargin.where     (dfReserveMargin      > 0.0, other=0.0)
    dfVariableMaxPower   = dfVariableMaxPower.where  (dfVariableMaxPower   > 0.0, other=0.0)
    dfEnergyInflows      = dfEnergyInflows.where     (dfEnergyInflows      > 0.0, other=0.0)
    dfEnergyOutflows     = dfEnergyOutflows.where    (dfEnergyOutflows     > 0.0, other=0.0)

    #%% parameters
    pIndBinGenInvest     = dfOption          ['IndBinGenInvest'    ][0].astype('int')                                                          # Indicator of binary generation expansion decisions, 0 continuous  - 1 binary - 2 no investment variables
    pIndBinNetInvest     = dfOption          ['IndBinNetInvest'    ][0].astype('int')                                                          # Indicator of binary network    expansion decisions, 0 continuous  - 1 binary - 2 no investment variables
    pIndBinGenOperat     = dfOption          ['IndBinGenOperat'    ][0].astype('int')                                                          # Indicator of binary generation operation decisions, 0 continuous  - 1 binary
    pIndBinSingleNode    = dfOption          ['IndBinSingleNode'   ][0].astype('int')                                                          # Indicator of single node although with network,     0 network     - 1 single node

    pENSCost             = dfParameter       ['ENSCost'            ][0] * 1e-3                                                                 # cost of energy not served                [MEUR/GWh]
    pCO2Cost             = dfParameter       ['CO2Cost'            ][0]                                                                        # cost of CO2 emission                     [EUR/t CO2]
    pEconomicBaseYear    = dfParameter       ['EconomicBaseYear'   ][0]                                                                        # economic base year                       [year]
    pAnnualDiscRate      = dfParameter       ['AnnualDiscountRate' ][0]                                                                        # annual discount rate                     [p.u.]
    pSBase               = dfParameter       ['SBase'              ][0] * 1e-3                                                                 # base power                               [GW]
    pReferenceNode       = dfParameter       ['ReferenceNode'      ][0]                                                                        # reference node
    pTimeStep            = dfParameter       ['TimeStep'           ][0].astype('int')                                                          # duration of the unit time step           [h]

    pPeriodWeight        = dfPeriod               ['Weight'        ].astype('int')                                                             # weights of periods                       [p.u.]
    pScenProb            = dfScenario             ['Probability'   ].astype('float')                                                           # probabilities of scenarios               [p.u.]
    pStageWeight         = dfStage                ['Weight'        ].astype('int')                                                             # weights of stages                        [p.u.]
    pDuration            = dfDuration             ['Duration'      ]    * pTimeStep                                                            # duration of load levels                  [h]
    pReserveMargin       = dfReserveMargin        ['ReserveMargin' ]                                                                           # minimum adequacy reserve margin          [p.u.]
    pLevelToStage        = dfDuration             ['Stage'         ]                                                                           # load levels assignment to stages
    pDemandP             = dfDemandP              [model.nd]    * 1e-3                                                                   # demand                                   [GW]
    pDemandQ             = dfDemandQ              [model.nd]    * 1e-3                                                                   # demand                                   [GVAr]
    pVariableMaxPower    = dfVariableMaxPower     [model.gg]    * 1e-3                                                                   # dynamic variable maximum power           [GW]
    pEnergyInflows       = dfEnergyInflows        [model.gg]    * 1e-3                                                                   # dynamic energy inflows                   [GW]
    pEnergyOutflows      = dfEnergyOutflows       [model.gg]    * 1e-3                                                                   # dynamic energy outflows                  [GW]

    # compute the demand as the mean over the time step load levels and assign it to active load levels. Idem for operating reserve, variable max power, variable min and max storage and inflows
    if pTimeStep > 1:
        pDemandP            = pDemandP.rolling           (pTimeStep).mean()
        pDemandQ            = pDemandQ.rolling           (pTimeStep).mean()
        pVariableMaxPower   = pVariableMaxPower.rolling  (pTimeStep).mean()
        pEnergyInflows      = pEnergyInflows.rolling     (pTimeStep).mean()
        pEnergyOutflows     = pEnergyOutflows.rolling    (pTimeStep).mean()

    pDemandP.fillna           (0.0, inplace=True)
    pDemandQ.fillna           (0.0, inplace=True)
    pVariableMaxPower.fillna  (0.0, inplace=True)
    pEnergyInflows.fillna     (0.0, inplace=True)
    pEnergyOutflows.fillna    (0.0, inplace=True)

    if pTimeStep > 1:
        # assign duration 0 to load levels not being considered, active load levels are at the end of every pTimeStep
        for i in range(pTimeStep-2,-1,-1):
            pDuration.iloc[[range(i,len(model.nn),pTimeStep)]] = 0

    #%% generation parameters
    pGenToNode          = dfGeneration  ['Node'                  ]                                                                            # generator location in node
    pGenToTechnology    = dfGeneration  ['Technology'            ]                                                                            # generator association to technology
    pIndBinUnitInvest   = dfGeneration  ['BinaryInvestment'      ]                                                                            # binary unit investment decision             [Yes]
    pIndBinUnitCommit   = dfGeneration  ['BinaryCommitment'      ]                                                                            # binary unit commitment decision             [Yes]
    pIndBinStorInvest   = dfGeneration  ['StorageInvestment'     ]                                                                            # storage linked to generation investment     [Yes]
    pMustRun            = dfGeneration  ['MustRun'               ]                                                                            # must-run unit                               [Yes]
    pPeriodIniGen       = dfGeneration  ['InitialPeriod'         ]                                                                            # initial period                              [year]
    pPeriodFinGen       = dfGeneration  ['FinalPeriod'           ]                                                                            # final   period                              [year]
    pAvailability       = dfGeneration  ['Availability'          ]                                                                            # unit availability for adequacy              [p.u.]
    pEFOR               = dfGeneration  ['EFOR'                  ]                                                                            # EFOR                                        [p.u.]
    pRatedMinPowerP     = dfGeneration  ['MinimumPower'          ] * 1e-3 * (1.0-dfGeneration['EFOR'])                                        # rated active minimum power                  [GW]
    pRatedMaxPowerP     = dfGeneration  ['MaximumPower'          ] * 1e-3 * (1.0-dfGeneration['EFOR'])                                        # rated active maximum power                  [GW]
    pRatedMinPowerQ     = dfGeneration  ['MinimumReactivePower'  ] * 1e-3 * (1.0-dfGeneration['EFOR'])                                        # rated reactive minimum power                [GVAr]
    pRatedMaxPowerQ     = dfGeneration  ['MaximumReactivePower'  ] * 1e-3 * (1.0-dfGeneration['EFOR'])                                        # rated reactive maximum power                [GVAr]
    pLinearFuelCost     = dfGeneration  ['LinearTerm'            ] * 1e-3 *      dfGeneration['FuelCost']                                     # fuel     term variable cost                 [MEUR/GWh]
    pLinearOMCost       = dfGeneration  ['OMVariableCost'        ] * 1e-3                                                                     # O&M      term variable cost                 [MEUR/GWh]
    pConstantVarCost    = dfGeneration  ['ConstantTerm'          ] * 1e-6 *      dfGeneration['FuelCost']                                     # constant term variable cost                 [MEUR/h]
    pStartUpCost        = dfGeneration  ['StartUpCost'           ]                                                                            # startup  cost                               [MEUR]
    pShutDownCost       = dfGeneration  ['ShutDownCost'          ]                                                                            # shutdown cost                               [MEUR]
    pCO2EmissionCost    = dfGeneration  ['CO2EmissionRate'       ] * 1e-3 * pCO2Cost                                                          # emission  cost                              [MEUR/GWh]
    pCO2EmissionRate    = dfGeneration  ['CO2EmissionRate'       ]                                                                            # emission  rate                              [tCO2/MWh]
    pGenInvestCost      = dfGeneration  ['FixedInvestmentCost'   ] *        dfGeneration['FixedChargeRate']                                   # generation fixed cost                       [MEUR]
    pRatedMinCharge     = dfGeneration  ['MinimumCharge'         ] * 1e-3                                                                     # rated minimum ESS charge                    [GW]
    pRatedMaxCharge     = dfGeneration  ['MaximumCharge'         ] * 1e-3                                                                     # rated maximum ESS charge                    [GW]
    pRatedMinStorage    = dfGeneration  ['MinimumStorage'        ]                                                                            # rated minimum ESS storage                   [GWh]
    pRatedMaxStorage    = dfGeneration  ['MaximumStorage'        ]                                                                            # rated maximum ESS storage                   [GWh]
    pInitialInventory   = dfGeneration  ['InitialStorage'        ]                                                                            # initial       ESS storage                   [GWh]
    pEfficiency         = dfGeneration  ['Efficiency'            ]                                                                            #         ESS round-trip efficiency           [p.u.]
    pStorageType        = dfGeneration  ['StorageType'           ]                                                                            #         ESS storage  type
    pOutflowsType       = dfGeneration  ['OutflowsType'          ]                                                                            #         ESS outflows type
    pGenLoInvest        = dfGeneration  ['InvestmentLo'          ]                                                                            # Lower bound of the investment decision      [p.u.]
    pGenUpInvest        = dfGeneration  ['InvestmentUp'          ]                                                                            # Upper bound of the investment decision      [p.u.]
    pGenSensitivity     = dfGeneration  ['Sensitivity'           ]                                                                            # Sensitivity                                 [Yes ]
    pGenFxInvest        = dfGeneration  ['InvestmentFixed'       ]                                                                            # Fixing the investment decision to a value   [p.u.]
    pGenSensiGroup      = dfGeneration  ['SensitivityGroup'      ]                                                                            # Sensitivity group                           [p.u.]
    pGenSensiGroupValue = dfGeneration  ['SensitivityGroupValue' ]                                                                            # Sensitivity group value                     [p.u.]

    pLinearOperCost     = pLinearFuelCost + pCO2EmissionCost
    pLinearVarCost      = pLinearFuelCost + pLinearOMCost

    pNodeLat            = dfNodeLocation['Latitude'              ]                                                                            # node latitude                               [º]
    pNodeLon            = dfNodeLocation['Longitude'             ]                                                                            # node longitude                              [º]

    pLineType           = dfNetwork     ['LineType'              ]                                                                            # line type
    pLineLength         = dfNetwork     ['Length'                ]                                                                            # line length                                 [km]
    pLineVoltage        = dfNetwork     ['Voltage'               ]                                                                            # line voltage                                [kV]
    pPeriodIniNet       = dfNetwork     ['InitialPeriod'         ]                                                                            # initial period
    pPeriodFinNet       = dfNetwork     ['FinalPeriod'           ]                                                                            # final   period
    pLineLossFactor     = dfNetwork     ['LossFactor'            ]                                                                            # loss factor                                 [p.u.]
    pLineR              = dfNetwork     ['Resistance'            ]                                                                            # resistance                                  [p.u.]
    pLineX              = dfNetwork     ['Reactance'             ].sort_index()                                                               # reactance                                   [p.u.]
    pLineBsh            = dfNetwork     ['Susceptance'           ]                                                                            # susceptance                                 [p.u.]
    pLineTAP            = dfNetwork     ['Tap'                   ]                                                                            # tap changer                                 [p.u.]
    pLineNTCFrw         = dfNetwork     ['TTC'                   ] * 1e-3 * dfNetwork['SecurityFactor' ]                                      # net transfer capacity in forward  direction [GW]
    pNetFixedCost       = dfNetwork     ['FixedInvestmentCost'   ] *        dfNetwork['FixedChargeRate']                                      # network    fixed cost                       [MEUR]
    pIndBinLineInvest   = dfNetwork     ['BinaryInvestment'      ]                                                                            # binary line investment decision             [Yes]
    pAngMin             = dfNetwork     ['AngMin'                ] * math.pi / 180                                                            # Min phase angle difference                  [rad]
    pAngMax             = dfNetwork     ['AngMax'                ] * math.pi / 180                                                            # Max phase angle difference                  [rad]
    pNetLoInvest        = dfNetwork     ['InvestmentLo'          ]                                                                            # Lower bound of the investment decision      [p.u.]
    pNetUpInvest        = dfNetwork     ['InvestmentUp'          ]                                                                            # Upper bound of the investment decision      [p.u.]
    pNetSensitivity     = dfNetwork     ['Sensitivity'           ]                                                                            # Sensitivity indicator                       [Yes ]
    pNetFxInvest        = dfNetwork     ['InvestmentFixed'       ]                                                                            # Fixing the investment decision to a value   [p.u.]
    pNetSensiGroup      = dfNetwork     ['SensitivityGroup'      ]                                                                            # Sensitivity group                           [p.u.]
    pNetSensiGroupValue = dfNetwork     ['SensitivityGroupValue' ]                                                                            # Sensitivity group value                     [p.u.]

    #%% bus shunt parameters
    pShuntToNode        = dfBusShunt    ['Node'                  ]                                                                            # generator location in node
    pPeriodIniShunt     = dfBusShunt    ['InitialPeriod'         ]                                                                            # initial period                              [year]
    pPeriodFinShunt     = dfBusShunt    ['FinalPeriod'           ]                                                                            # final   period                              [year]
    pGshb               = dfBusShunt    ['Gshb'                  ]                                                                            # Conductance of the bus shunt device         [p.u.]
    pBshb               = dfBusShunt    ['Bshb'                  ]                                                                            # Susceptance of the bus shunt device         [p.u.]
    pShuntInvestCost    = dfBusShunt    ['FixedInvestmentCost'   ] *        dfBusShunt['FixedChargeRate']                                     # generation fixed cost                       [MEUR]
    pIndBinShuntInvest  = dfBusShunt    ['BinaryInvestment'      ]                                                                            # binary bus shunt investment decision        [Yes]
    pShuntLoInvest      = dfBusShunt    ['InvestmentLo'          ]                                                                            # Lower bound of the investment decision      [p.u.]
    pShuntUpInvest      = dfBusShunt    ['InvestmentUp'          ]                                                                            # Upper bound of the investment decision      [p.u.]

    # replace pGenUpInvest = 0.0 by 1.0
    pGenUpInvest    = pGenUpInvest.where(pGenUpInvest     > 0.0, other=1.0        )
    # replace pNetUpInvest = 0.0 by 1.0
    pNetUpInvest    = pNetUpInvest.where(pNetUpInvest     > 0.0, other=1.0        )
    # replace pShuntUpInvest = 0.0 by 1.0
    pShuntUpInvest  = pShuntUpInvest.where(pShuntUpInvest > 0.0, other=1.0        )

    ReadingDataTime = time.time() - StartTime
    StartTime       = time.time()
    print('Reading    input data                 ... ', round(ReadingDataTime), 's')

    #%% Modifying line parameters

    pLineZ2     =  pLineR**2 + pLineX**2
    pLineG      =  pLineR/pLineZ2
    pLineB      = -pLineX/pLineZ2
    pLineBsh    =  pLineBsh/20

    pLineTAP    = pLineTAP.where(pLineTAP > 0.0, other=1.0        )
    pLineTAP    = 1/pLineTAP

    #%% Getting the branches from the network data
    sBr = [(ni,nf) for (ni,nf,cc) in dfNetwork.index]
    # Dropping duplicate keys
    sBrList = [(ni,nf) for n,(ni,nf) in enumerate(sBr) if (ni,nf) not in sBr[:n]]

    #%% Changing the label of the sensitivities
    sGenSen = pd.Series(data=['Gr'+str(int(pGenSensiGroup[      gg])) for  gg        in model.gg       ], index=model.gg       )
    sNetSen = pd.Series(data=['Gr'+str(int(pNetSensiGroup[ni,nf,cc])) for (ni,nf,cc) in dfNetwork.index], index=dfNetwork.index)
    # Dropping duplicate keys
    sGSList = sGenSen.unique()
    sNSList = sNetSen.unique()

    #%% defining subsets: active load levels (n,n2), thermal units (t), RES units (r), ESS units (es), candidate gen units (gc), candidate ESS units (ec), all the lines (la), candidate lines (lc), candidate DC lines (cd), existing DC lines (cd), lines with losses (ll), reference node (rf), and reactive generating units (gq)
    model.p   = Set(initialize=model.pp,          ordered=True , doc='periods'              , filter=lambda model,pp      :  pp     in model.pp  and pPeriodWeight     [pp] >  0.0)
    model.sc  = Set(initialize=model.scc,         ordered=True , doc='scenarios'            , filter=lambda model,scc     :  scc    in model.scc                                  )
    model.ps  = Set(initialize=model.p*model.sc,  ordered=True , doc='periods/scenarios'    , filter=lambda model,p,sc    :  (p,sc) in model.p*model.sc and pScenProb[p,sc] > 0.0)
    model.st  = Set(initialize=model.stt,         ordered=True , doc='stages'               , filter=lambda model,stt     :  stt    in model.stt and pStageWeight     [stt] >  0.0)
    model.n   = Set(initialize=model.nn,          ordered=True , doc='load levels'          , filter=lambda model,nn      :  nn     in model.nn  and pDuration         [nn] >  0  )
    model.n2  = Set(initialize=model.nn,          ordered=True , doc='load levels'          , filter=lambda model,nn      :  nn     in model.nn  and pDuration         [nn] >  0  )
    model.g   = Set(initialize=model.gg,          ordered=False, doc='generating      units', filter=lambda model,gg      :  gg     in model.gg  and (pRatedMaxPowerP  [gg] >  0.0 or  pRatedMaxCharge[gg] >  0.0) and pPeriodIniGen[gg] <= model.p.last() and pPeriodFinGen[gg] >= model.p.first() and pGenToNode.reset_index().set_index(['index']).isin(model.nd)['Node'][gg])  # excludes generators with empty node
    model.t   = Set(initialize=model.g ,          ordered=False, doc='thermal         units', filter=lambda model,g       :  g      in model.g   and pLinearOperCost   [g ] >  0.0)
    model.r   = Set(initialize=model.g ,          ordered=False, doc='RES             units', filter=lambda model,g       :  g      in model.g   and pLinearOperCost   [g ] == 0.0 and pRatedMaxStorage[g] == 0.0)
    model.es  = Set(initialize=model.g ,          ordered=False, doc='ESS             units', filter=lambda model,g       :  g      in model.g   and                                  (pRatedMaxStorage[g] >  0.0   or pRatedMaxCharge[g] > 0.0))
    model.gc  = Set(initialize=model.g ,          ordered=False, doc='candidate       units', filter=lambda model,g       :  g      in model.g   and pGenInvestCost    [g ] >  0.0)
    model.ec  = Set(initialize=model.es,          ordered=False, doc='candidate   ESS units', filter=lambda model,es      :  es     in model.es  and pGenInvestCost    [es] >  0.0)
    model.br  = Set(initialize=sBrList,           ordered=False, doc='all input branches'                                                                                         )
    model.ln  = Set(initialize=dfNetwork.index,   ordered=False, doc='all input lines'                                                                                            )
    model.la  = Set(initialize=model.ln,          ordered=False, doc='all real        lines', filter=lambda model,*ln     :  ln     in model.ln  and pLineX            [ln] != 0.0 and pLineNTCFrw[ln] > 0.0 and pPeriodIniNet[ln] <= model.p.last() and pPeriodFinNet[ln] >= model.p.first())
    model.lc  = Set(initialize=model.la,          ordered=False, doc='candidate       lines', filter=lambda model,*la     :  la     in model.la  and pNetFixedCost     [la] >  0.0)
    model.cd  = Set(initialize=model.la,          ordered=False, doc='             DC lines', filter=lambda model,*la     :  la     in model.la  and pNetFixedCost     [la] >  0.0 and pLineType[la] == 'DC')
    model.ed  = Set(initialize=model.la,          ordered=False, doc='             DC lines', filter=lambda model,*la     :  la     in model.la  and pNetFixedCost     [la] == 0.0 and pLineType[la] == 'DC')
    model.rf  = Set(initialize=model.nd,          ordered=True , doc='reference node'       , filter=lambda model,nd      :  nd     in                pReferenceNode              )
    model.gs  = Set(initialize=sGSList,           ordered=True , doc='all the groups which participate in the sensitivities'                                                      )
    model.ns  = Set(initialize=sNSList,           ordered=True , doc='all the groups which participate in the sensitivities'                                                      )
    model.sh  = Set(initialize=model.shh,         ordered=False, doc='bus shunts'           , filter=lambda model,shh     :  shh    in model.shh and pPeriodIniShunt[shh] <= model.p.last() and pPeriodFinShunt[shh] >= model.p.first() and pShuntToNode.reset_index().set_index(['index']).isin(model.nd)['Node'][shh])  # excludes devices with empty node
    model.shc = Set(initialize=model.sh,          ordered=False, doc='candidate bus shunt'  , filter=lambda model,sh      :  sh     in model.sh  and pShuntInvestCost  [sh] >  0.0)

    # non-RES units, they can be committed and also contribute to the operating reserves
    model.nr = model.g - model.r

    # existing lines (le)
    model.le = model.la - model.lc

    # existing bus shunt (she)
    model.she = model.sh - model.shc

    # instrumental sets
    model.psc    = [(p,sc     ) for p,sc      in model.p  *model.sc ]
    model.psn    = [(p,sc,n   ) for p,sc,n    in model.ps *model.n  ]
    model.psng   = [(p,sc,n,g ) for p,sc,n,g  in model.psn*model.g  ]
    model.psngg  = [(p,sc,n,gg) for p,sc,n,gg in model.psn*model.gg ]
    model.psnr   = [(p,sc,n,r ) for p,sc,n,r  in model.psn*model.r  ]
    model.psnnr  = [(p,sc,n,nr) for p,sc,n,nr in model.psn*model.nr ]
    model.psnes  = [(p,sc,n,es) for p,sc,n,es in model.psn*model.es ]
    model.psnec  = [(p,sc,n,ec) for p,sc,n,ec in model.psn*model.ec ]
    model.psnnd  = [(p,sc,n,nd) for p,sc,n,nd in model.psn*model.nd ]
    model.psnar  = [(p,sc,n,ar) for p,sc,n,ar in model.psn*model.ar ]
    model.psngt  = [(p,sc,n,gt) for p,sc,n,gt in model.psn*model.gt ]
    model.psnsh  = [(p,sc,n,sh) for p,sc,n,sh in model.psn*model.sh ]
    model.psnshc = [(p,sc,n,sh) for p,sc,n,sh in model.psn*model.shc]
    model.psnshe = [(p,sc,n,sh) for p,sc,n,sh in model.psn*model.she]

    model.psnla = [(p,sc,n,ni,nf,cc) for p,sc,n,ni,nf,cc in model.psn*model.la ]
    model.psnln = [(p,sc,n,ni,nf,cc) for p,sc,n,ni,nf,cc in model.psn*model.ln ]

    model.pgc   = [(p,gc           ) for p,gc            in model.p  *model.gc ]
    model.pec   = [(p,ec           ) for p,ec            in model.p  *model.ec ]
    model.plc   = [(p,ni,nf,cc     ) for p,ni,nf,cc      in model.p  *model.lc ]
    model.pshc  = [(p,shc          ) for p,shc           in model.p  *model.shc]

    # assigning a node to an area
    model.ndar = [(nd,ar) for (nd,zn,ar) in model.ndzn*model.ar if (zn,ar) in model.znar]

    # assigning a line to an area. Both nodes are in the same area. Cross-area lines not included
    model.laar = [(ni,nf,cc,ar) for ni,nf,cc,ar in model.la*model.ar if (ni,ar) in model.ndar and (nf,ar) in model.ndar]

    # replacing string values by numerical values
    idxDict = dict()
    idxDict[0    ] = 0
    idxDict[0.0  ] = 0
    idxDict['No' ] = 0
    idxDict['NO' ] = 0
    idxDict['no' ] = 0
    idxDict['N'  ] = 0
    idxDict['n'  ] = 0
    idxDict['Yes'] = 1
    idxDict['YES'] = 1
    idxDict['yes'] = 1
    idxDict['Y'  ] = 1
    idxDict['y'  ] = 1

    pIndBinUnitInvest  = pIndBinUnitInvest.map (idxDict)
    pIndBinUnitCommit  = pIndBinUnitCommit.map (idxDict)
    pIndBinStorInvest  = pIndBinStorInvest.map (idxDict)
    pMustRun           = pMustRun.map          (idxDict)
    pIndBinLineInvest  = pIndBinLineInvest.map (idxDict)
    pIndBinShuntInvest = pIndBinShuntInvest.map(idxDict)
    pGenSensitivity    = pGenSensitivity.map   (idxDict)
    pNetSensitivity    = pNetSensitivity.map   (idxDict)

    # define AC existing  lines
    model.lea = Set(initialize=model.le, ordered=False, doc='AC existing  lines and non-switchable lines', filter=lambda model,*le: le in model.le and not pLineType[le] == 'DC')
    # define AC candidate lines
    model.lca = Set(initialize=model.la, ordered=False, doc='AC candidate lines and     switchable lines', filter=lambda model,*lc: lc in model.lc and not pLineType[lc] == 'DC')

    model.laa = model.lea | model.lca

    # define DC existing  lines
    model.led = Set(initialize=model.le, ordered=False, doc='DC existing  lines and non-switchable lines', filter=lambda model,*le: le in model.le and     pLineType[le] == 'DC')
    # define DC candidate lines
    model.lcd = Set(initialize=model.la, ordered=False, doc='DC candidate lines and     switchable lines', filter=lambda model,*lc: lc in model.lc and     pLineType[lc] == 'DC')

    model.lad = model.led | model.lcd

    # line type
    pLineType = pLineType.reset_index().set_index(['level_0','level_1','level_2','LineType'])

    model.pLineType = Set(initialize=pLineType.index, ordered=False, doc='line type')

    #%% inverse index load level to stage
    pStageToLevel = pLevelToStage.reset_index().set_index('Stage').set_axis(['LoadLevel'], axis=1, copy=False)[['LoadLevel']]
    pStageToLevel = pStageToLevel.loc[pStageToLevel['LoadLevel'].isin(model.n)]
    pStage2Level  = pStageToLevel.reset_index().set_index(['Stage','LoadLevel'])

    model.s2n = Set(initialize=pStage2Level.index, ordered=False, doc='load level to stage')

    if pAnnualDiscRate == 0.0:
        pDiscountFactor = pd.Series([                        pPeriodWeight[p]                                                                                          for p in model.p], index=model.p)
    else:
        pDiscountFactor = pd.Series([((1.0+pAnnualDiscRate)**pPeriodWeight[p]-1.0) / (pAnnualDiscRate*(1.0+pAnnualDiscRate)**(pPeriodWeight[p]-1+p-pEconomicBaseYear)) for p in model.p], index=model.p)

    pLoadLevelWeight = pd.Series([0.0 for n in model.n], index=model.n)
    for st,n in model.s2n:
        pLoadLevelWeight[n] = pStageWeight[st]

    #%% inverse index node to generator
    pNodeToGen = pGenToNode.reset_index().set_index('Node').set_axis(['Generator'], axis=1, copy=False)[['Generator']]
    pNodeToGen = pNodeToGen.loc[pNodeToGen['Generator'].isin(model.g)]
    pNode2Gen  = pNodeToGen.reset_index().set_index(['Node', 'Generator'])

    model.n2g = Set(initialize=pNode2Gen.index, ordered=False, doc='node   to generator')

    pZone2Gen   = [(zn,g) for (nd,g,zn      ) in model.n2g*model.zn                     if (nd,zn) in model.ndzn                                                      ]
    pArea2Gen   = [(ar,g) for (nd,g,zn,ar   ) in model.n2g*model.zn*model.ar           if (nd,zn) in model.ndzn and [zn,ar] in model.znar                           ]
    pRegion2Gen = [(rg,g) for (nd,g,zn,ar,rg) in model.n2g*model.zn*model.ar*model.rg if (nd,zn) in model.ndzn and [zn,ar] in model.znar and [ar,rg] in model.arrg]

    model.z2g = Set(initialize=model.zn*model.g, ordered=False, doc='zone   to generator', filter=lambda model,zn,g: (zn,g) in pZone2Gen  )
    model.a2g = Set(initialize=model.ar*model.g, ordered=False, doc='area   to generator', filter=lambda model,ar,g: (ar,g) in pArea2Gen  )
    model.r2g = Set(initialize=model.rg*model.g, ordered=False, doc='region to generator', filter=lambda model,rg,g: (rg,g) in pRegion2Gen)

    #%% inverse index generator to technology
    pTechnologyToGen = pGenToTechnology.reset_index().set_index('Technology').set_axis(['Generator'], axis=1, copy=False)[['Generator']]
    pTechnologyToGen = pTechnologyToGen.loc[pTechnologyToGen['Generator'].isin(model.g)]
    pTechnology2Gen  = pTechnologyToGen.reset_index().set_index(['Technology', 'Generator'])

    model.t2g = Set(initialize=pTechnology2Gen.index, ordered=False, doc='technology to generator')

    # ESS and RES technologies
    model.ot = Set(initialize=model.gt, ordered=False, doc='ESS technologies', filter=lambda model,gt: gt in model.gt and sum(1 for es in model.es if (gt,es) in model.t2g))
    model.rt = Set(initialize=model.gt, ordered=False, doc='RES technologies', filter=lambda model,gt: gt in model.gt and sum(1 for r  in model.r  if (gt,r ) in model.t2g))

    model.psnot = [(p,sc,n,ot) for p,sc,n,ot in model.ps*model.n*model.ot]
    model.psnrt = [(p,sc,n,rt) for p,sc,n,rt in model.ps*model.n*model.rt]

    #%% inverse index node to bus shunt
    pNodeToShunt = pShuntToNode.reset_index().set_index('Node').set_axis(['Shunt'], axis=1, copy=False)[['Shunt']]
    pNodeToShunt = pNodeToShunt.loc[pNodeToShunt['Shunt'].isin(model.sh)]
    pNode2Shunt  = pNodeToShunt.reset_index().set_index(['Node', 'Shunt'])

    model.n2sh   = Set(initialize=pNode2Shunt.index, ordered=False, doc='node   to bus shunt')

    #%% inverse index sensitivity group to device
    pSenGrToGen = sGenSen.reset_index().set_index('index').set_axis([0], axis=1, copy=False)[[0]]
    pSenGrToGen = pSenGrToGen.loc[pSenGrToGen[0].isin(model.gs)]
    pSenGr2Gen  = pSenGrToGen.reset_index().set_index([0,'index'])

    model.sg2g  = Set(initialize=pSenGr2Gen.index, ordered=False, doc='sensitivity group to generator')

    pSenGrToNet = sNetSen.reset_index().set_index(['level_0','level_1','level_2']).set_axis([0], axis=1, copy=False)[[0]]
    pSenGrToNet = pSenGrToNet.loc[pSenGrToNet[0].isin(model.ns)]
    pSenGr2Net  = pSenGrToNet.reset_index().set_index([0,'level_0','level_1','level_2'])

    model.sg2la = Set(initialize=pSenGr2Net.index, ordered=False, doc='sensitivity group to line')

    # minimum and maximum variable power
    pVariableMaxPower   = pVariableMaxPower.replace(0.0, float('nan'))
    pMinPower           = pd.DataFrame([pRatedMinPowerP]*len(model.ps*model.nn), index=pd.MultiIndex.from_tuples(model.ps*model.nn), columns=pRatedMinPowerP.index)
    pMaxPower           = pd.DataFrame([pRatedMaxPowerP]*len(model.ps*model.nn), index=pd.MultiIndex.from_tuples(model.ps*model.nn), columns=pRatedMaxPowerP.index)
    pMinPower           = pMinPower.reindex        (sorted(pMinPower.columns        ), axis=1)
    pMaxPower           = pMaxPower.reindex        (sorted(pMaxPower.columns        ), axis=1)
    pVariableMaxPower   = pVariableMaxPower.reindex(sorted(pVariableMaxPower.columns), axis=1)
    pMaxPower           = pVariableMaxPower.where         (pVariableMaxPower < pMaxPower, other=pMaxPower)
    pMinPower           = pMinPower.where                 (pMinPower         > 0.0,       other=0.0)
    pMaxPower           = pMaxPower.where                 (pMaxPower         > 0.0,       other=0.0)

    # minimum and maximum variable charge
    pMinCharge          = pd.DataFrame([pRatedMinCharge]*len(model.ps*model.nn), index=pd.MultiIndex.from_tuples(model.ps*model.nn), columns=pRatedMinCharge.index)
    pMaxCharge          = pd.DataFrame([pRatedMaxCharge]*len(model.ps*model.nn), index=pd.MultiIndex.from_tuples(model.ps*model.nn), columns=pRatedMaxCharge.index)
    pMinCharge          = pMinCharge.reindex        (sorted(pMinCharge.columns        ), axis=1)
    pMaxCharge          = pMaxCharge.reindex        (sorted(pMaxCharge.columns        ), axis=1)
    pMinCharge          = pMinCharge.where                 (pMinCharge         > 0.0,        other=0.0)
    pMaxCharge          = pMaxCharge.where                 (pMaxCharge         > 0.0,        other=0.0)

    # minimum and maximum variable storage capacity
    pMinStorage         = pd.DataFrame([pRatedMinStorage]*len(model.ps*model.nn), index=pd.MultiIndex.from_tuples(model.ps*model.nn), columns=pRatedMinStorage.index)
    pMaxStorage         = pd.DataFrame([pRatedMaxStorage]*len(model.ps*model.nn), index=pd.MultiIndex.from_tuples(model.ps*model.nn), columns=pRatedMaxStorage.index)
    pMinStorage         = pMinStorage.reindex        (sorted(pMinStorage.columns        ), axis=1)
    pMaxStorage         = pMaxStorage.reindex        (sorted(pMaxStorage.columns        ), axis=1)
    pMinStorage         = pMinStorage.where                 (pMinStorage         > 0.0,         other=0.0)
    pMaxStorage         = pMaxStorage.where                 (pMaxStorage         > 0.0,         other=0.0)

    # parameter that allows the initial inventory to change with load level
    pIniInventory       = pd.DataFrame([pInitialInventory]*len(model.ps*model.nn), index=pd.MultiIndex.from_tuples(model.ps*model.nn), columns=pInitialInventory.index)

    # %% definition of the time-steps leap to observe the stored energy at an ESS
    idxCycle = dict()
    idxCycle[0        ] = 1
    idxCycle[0.0      ] = 1
    idxCycle["Hourly" ] = 1
    idxCycle["Daily"  ] = 1
    idxCycle["Weekly" ] = round(  24/pTimeStep)
    idxCycle["Monthly"] = round( 168/pTimeStep)
    idxCycle["Yearly" ] = round(8736/pTimeStep)

    idxOutflows = dict()
    idxOutflows[0        ] = 8736
    idxOutflows[0.0      ] = 8736
    idxOutflows["Hourly" ] = 1
    idxOutflows["Daily"  ] = 1
    idxOutflows["Weekly" ] = round(  24/pTimeStep)
    idxOutflows["Monthly"] = round( 168/pTimeStep)
    idxOutflows["Yearly" ] = round(8736/pTimeStep)

    pCycleTimeStep    = pStorageType.map(idxCycle)
    pOutflowsTimeStep = pOutflowsType.map(idxOutflows)

    pCycleTimeStep    = pd.concat([pCycleTimeStep,pOutflowsTimeStep], axis=1).min(axis=1)

    # drop load levels with duration 0
    pDuration         = pDuration.loc        [model.n    ]
    pDemandP          = pDemandP.loc         [model.psn  ]
    pDemandQ          = pDemandQ.loc         [model.psn  ]
    pMinPower         = pMinPower.loc        [model.psn  ]
    pMaxPower         = pMaxPower.loc        [model.psn  ]
    pMinCharge        = pMinCharge.loc       [model.psn  ]
    pMaxCharge        = pMaxCharge.loc       [model.psn  ]
    pEnergyInflows    = pEnergyInflows.loc   [model.psn  ]
    pEnergyOutflows   = pEnergyOutflows.loc  [model.psn  ]
    pMinStorage       = pMinStorage.loc      [model.psn  ]
    pMaxStorage       = pMaxStorage.loc      [model.psn  ]
    pIniInventory     = pIniInventory.loc    [model.psn  ]

    # separate positive and negative demands to avoid converting negative values to 0
    pDemandPos        = pDemandP.where        (pDemandP >= 0.0, other=0.0)
    pDemandNeg        = pDemandP.where        (pDemandP <  0.0, other=0.0)

    # small values are converted to 0
    pPeakDemand         = pd.Series([0.0 for ar in model.ar], index=model.ar)
    for ar in model.ar:
        # values < 1e-5 times the maximum demand for each area (an area is related to operating reserves procurement, i.e., country) are converted to 0
        pPeakDemand[ar] = pDemandP   [[nd for nd in model.nd if (nd,ar) in model.ndar]].sum(axis=1).max()
        pEpsilon        = pPeakDemand[ar]*2.5e-5
        # values < 1e-5 times the maximum system demand are converted to 0
        # pEpsilon      = pDemand.sum(axis=1).max()*1e-5

        # these parameters are in GW
        pDemandPos     [pDemandPos     [[nd for nd in model.nd if (nd,ar) in model.ndar]] <  pEpsilon] = 0.0
        pDemandNeg     [pDemandNeg     [[nd for nd in model.nd if (nd,ar) in model.ndar]] > -pEpsilon] = 0.0
        pMinPower      [pMinPower      [[g  for  g in model.g  if (ar,g)  in model.a2g ]] <  pEpsilon] = 0.0
        pMaxPower      [pMaxPower      [[g  for  g in model.g  if (ar,g)  in model.a2g ]] <  pEpsilon] = 0.0
        pEnergyInflows [pEnergyInflows [[es for es in model.es if (ar,es) in model.a2g ]] <  pEpsilon/pTimeStep] = 0.0
        pEnergyOutflows[pEnergyOutflows[[es for es in model.es if (ar,es) in model.a2g ]] <  pEpsilon/pTimeStep] = 0.0

        # these parameters are in GWh
        pIniInventory  [pIniInventory  [[es for es in model.es if (ar,es) in model.a2g ]] <  pEpsilon] = 0.0

        pInitialInventory.update(pd.Series([0 for es in model.es if (ar,es) in model.a2g and pInitialInventory[es] < pEpsilon], index=[es for es in model.es if (ar,es) in model.a2g and pInitialInventory[es] < pEpsilon], dtype='float64'))

        pLineNTCFrw.update(pd.Series([0.0 for ni,nf,cc in model.la if pLineNTCFrw[ni,nf,cc] < pEpsilon], index=[(ni,nf,cc) for ni,nf,cc in model.la if pLineNTCFrw[ni,nf,cc] < pEpsilon], dtype='float64'))

        # merging positive and negative values of the demand
        pDemandP           = pDemandPos.where(pDemandNeg >= 0.0, other=pDemandNeg)

        pMaxPower2ndBlock  = pMaxPower  - pMinPower
        pMaxCharge2ndBlock = pMaxCharge - pMinCharge

        pMaxPower2ndBlock [pMaxPower2ndBlock [[es for es in model.es if (ar,es) in model.a2g]] < pEpsilon] = 0.0
        pMaxCharge2ndBlock[pMaxCharge2ndBlock[[es for es in model.es if (ar,es) in model.a2g]] < pEpsilon] = 0.0

    # replace very small costs by 0
    pEpsilon = 1e-4           # this value in €/GWh is related to the smallest reduced cost, independent of the area
    pLinearOperCost.update (pd.Series([0 for gg in model.gg if pLinearOperCost [gg] < pEpsilon], index=[gg for gg in model.gg if pLinearOperCost [gg] < pEpsilon]))
    pLinearVarCost.update  (pd.Series([0 for gg in model.gg if pLinearVarCost  [gg] < pEpsilon], index=[gg for gg in model.gg if pLinearVarCost  [gg] < pEpsilon]))
    pLinearOMCost.update   (pd.Series([0 for gg in model.gg if pLinearOMCost   [gg] < pEpsilon], index=[gg for gg in model.gg if pLinearOMCost   [gg] < pEpsilon]))
    pConstantVarCost.update(pd.Series([0 for gg in model.gg if pConstantVarCost[gg] < pEpsilon], index=[gg for gg in model.gg if pConstantVarCost[gg] < pEpsilon]))
    pCO2EmissionCost.update(pd.Series([0 for gg in model.gg if pCO2EmissionCost[gg] < pEpsilon], index=[gg for gg in model.gg if pCO2EmissionCost[gg] < pEpsilon]))
    pStartUpCost.update    (pd.Series([0 for gg in model.gg if pStartUpCost    [gg] < pEpsilon], index=[gg for gg in model.gg if pStartUpCost    [gg] < pEpsilon]))
    pShutDownCost.update   (pd.Series([0 for gg in model.gg if pShutDownCost   [gg] < pEpsilon], index=[gg for gg in model.gg if pShutDownCost   [gg] < pEpsilon]))

    # replace < 0.0 by 0.0
    pMaxPower2ndBlock  = pMaxPower2ndBlock.where (pMaxPower2ndBlock  > 0.0, other=0.0)

    # BigM maximum flow to be used in the Kirchhoff's 2nd law disjunctive constraint
    pBigMFlowFrw = pLineNTCFrw*0.0
    for lea in model.lea:
        pBigMFlowFrw.loc[lea] = pLineNTCFrw[lea]
    for lca in model.lca:
        pBigMFlowFrw.loc[lca] = pLineNTCFrw[lca]*1.5
    for led in model.led:
        pBigMFlowFrw.loc[led] = pLineNTCFrw[led]
    for lcd in model.lcd:
        pBigMFlowFrw.loc[lcd] = pLineNTCFrw[lcd]*1.5

    # if BigM are 0.0 then converted to 1.0 to avoid division by 0.0
    pBigMFlowFrw = pBigMFlowFrw.where(pBigMFlowFrw != 0.0, other=1.0)

    # maximum voltage angle
    pMaxTheta = pDemandP*0.0 + math.pi/2
    pMaxTheta = pMaxTheta.loc[model.psn]

    # this option avoids a warning in the following assignments
    pd.options.mode.chained_assignment = None

    # this dataframe are converted to dictionaries
    pDemandP           = pDemandP.stack().to_dict()
    pDemandQ           = pDemandQ.stack().to_dict()
    pMinPower          = pMinPower.stack().to_dict()
    pMaxPower          = pMaxPower.stack().to_dict()
    pMinCharge         = pMinCharge.stack().to_dict()
    pMaxCharge         = pMaxCharge.stack().to_dict()
    pMaxPower2ndBlock  = pMaxPower2ndBlock.stack().to_dict()
    pMaxCharge2ndBlock = pMaxCharge2ndBlock.stack().to_dict()
    pEnergyInflows     = pEnergyInflows.stack().to_dict()
    pEnergyOutflows    = pEnergyOutflows.stack().to_dict()
    pMinStorage        = pMinStorage.stack().to_dict()
    pMaxStorage        = pMaxStorage.stack().to_dict()
    pIniInventory      = pIniInventory.stack().to_dict()
    pMaxTheta          = pMaxTheta.stack().to_dict()

    # # thermal and variable units ordered by increasing variable cost
    # model.go = pLinearVarCost.sort_values().index

    pLoadLevelDuration = pd.Series([0 for n in model.n], index=model.n)
    for n in model.n:
        pLoadLevelDuration[n] = pLoadLevelWeight[n] * pDuration[n]

    pPeriodProb = pd.Series([0.0 for p,sc in model.ps], index=pd.MultiIndex.from_tuples(model.ps))
    for p,sc in model.ps:
        pPeriodProb[p,sc] = pPeriodWeight[p] * pScenProb[p,sc]

    # if unit availability = 0 changed to 1
    for g in model.g:
        if  pAvailability[g] == 0.0:
            pAvailability[g]   =  1.0

    # if line length = 0 changed to geographical distance with an additional 10%
    for ni,nf,cc in model.la:
        if  pLineLength[ni,nf,cc] == 0.0:
            pLineLength[ni,nf,cc]  =  1.1 * 6371 * 2 * math.asin(math.sqrt(math.pow(math.sin((pNodeLat[nf]-pNodeLat[ni])*math.pi/180/2),2) + math.cos(pNodeLat[ni]*math.pi/180)*math.cos(pNodeLat[nf]*math.pi/180)*math.pow(math.sin((pNodeLon[nf]-pNodeLon[ni])*math.pi/180/2),2)))
    #%% Parameters
    model.pReserveMargin        = Param(model.ar,    initialize=pReserveMargin.to_dict()            , within=NonNegativeReals,    doc='Adequacy reserve margin'                             )
    model.pPeakDemand           = Param(model.ar,    initialize=pPeakDemand.to_dict()               , within=NonNegativeReals,    doc='Peak demand'                                         )
    model.pDemandP              = Param(model.psnnd, initialize=pDemandP                            , within=           Reals,    doc='Demand'                                              )
    model.pDemandQ              = Param(model.psnnd, initialize=pDemandQ                            , within=           Reals,    doc='Demand'                                              )
    model.pPeriodWeight         = Param(model.p,     initialize=pPeriodWeight.to_dict()             , within=NonNegativeIntegers, doc='Period weight',                          mutable=True)
    model.pDiscountFactor       = Param(model.p,     initialize=pDiscountFactor.to_dict()           , within=NonNegativeReals,    doc='Discount factor'                                     )
    model.pScenProb             = Param(model.psc,   initialize=pScenProb.to_dict()                 , within=UnitInterval    ,    doc='Probability',                            mutable=True)
    model.pStageWeight          = Param(model.stt,   initialize=pStageWeight.to_dict()              , within=NonNegativeIntegers, doc='Stage weight'                                        )
    model.pDuration             = Param(model.n,     initialize=pDuration.to_dict()                 , within=PositiveIntegers,    doc='Duration',                               mutable=True)
    model.pNodeLon              = Param(model.nd,    initialize=pNodeLon.to_dict()                  ,                             doc='Longitude'                                           )
    model.pNodeLat              = Param(model.nd,    initialize=pNodeLat.to_dict()                  ,                             doc='Latitude'                                            )
    # model.pSystemInertia        = Param(model.psnar, initialize=pSystemInertia.stack().to_dict()    , within=NonNegativeReals,    doc='System inertia'                                      )
    # model.pOperReserveUp        = Param(model.psnar, initialize=pOperReserveUp.stack().to_dict()    , within=NonNegativeReals,    doc='Upward   operating reserve'                          )
    # model.pOperReserveDw        = Param(model.psnar, initialize=pOperReserveDw.stack().to_dict()    , within=NonNegativeReals,    doc='Downward operating reserve'                          )
    model.pMinPower             = Param(model.psngg, initialize=pMinPower                           , within=NonNegativeReals,    doc='Minimum power'                                       )
    model.pMaxPower             = Param(model.psngg, initialize=pMaxPower                           , within=NonNegativeReals,    doc='Maximum power'                                       )
    model.pMinCharge            = Param(model.psngg, initialize=pMinCharge                          , within=NonNegativeReals,    doc='Minimum charge'                                      )
    model.pMaxCharge            = Param(model.psngg, initialize=pMaxCharge                          , within=NonNegativeReals,    doc='Maximum charge'                                      )
    model.pMaxPower2ndBlock     = Param(model.psngg, initialize=pMaxPower2ndBlock                   , within=NonNegativeReals,    doc='Second block power'                                  )
    model.pMaxCharge2ndBlock    = Param(model.psngg, initialize=pMaxCharge2ndBlock                  , within=NonNegativeReals,    doc='Second block charge'                                 )
    model.pEnergyInflows        = Param(model.psngg, initialize=pEnergyInflows                      , within=NonNegativeReals,    doc='Energy inflows',                         mutable=True)
    model.pEnergyOutflows       = Param(model.psngg, initialize=pEnergyOutflows                     , within=NonNegativeReals,    doc='Energy outflows',                        mutable=True)
    model.pMinStorage           = Param(model.psngg, initialize=pMinStorage                         , within=NonNegativeReals,    doc='ESS Minimum storage capacity'                        )
    model.pMaxStorage           = Param(model.psngg, initialize=pMaxStorage                         , within=NonNegativeReals,    doc='ESS Maximum storage capacity'                        )
    # model.pMinEnergy            = Param(model.psngg, initialize=pVariableMinEnergy.stack().to_dict(), within=NonNegativeReals,    doc='Unit minimum energy demand'                          )
    # model.pMaxEnergy            = Param(model.psngg, initialize=pVariableMaxEnergy.stack().to_dict(), within=NonNegativeReals,    doc='Unit maximum energy demand'                          )
    model.pRatedMaxPowerP       = Param(model.gg,    initialize=pRatedMaxPowerP.to_dict()           , within=NonNegativeReals,    doc='Rated maximum power'                                 )
    model.pRatedMaxPowerQ       = Param(model.gg,    initialize=pRatedMaxPowerQ.to_dict()           , within=           Reals,    doc='Rated maximum power'                                 )
    model.pRatedMaxCharge       = Param(model.gg,    initialize=pRatedMaxCharge.to_dict()           , within=NonNegativeReals,    doc='Rated maximum charge'                                )
    model.pMustRun              = Param(model.gg,    initialize=pMustRun.to_dict()                  , within=Binary          ,    doc='must-run unit'                                       )
    # model.pInertia              = Param(model.gg,    initialize=pInertia.to_dict()                  , within=NonNegativeReals,    doc='unit inertia constant'                               )
    model.pPeriodIniGen         = Param(model.gg,    initialize=pPeriodIniGen.to_dict()             , within=PositiveIntegers,    doc='installation year',                                  )
    model.pPeriodFinGen         = Param(model.gg,    initialize=pPeriodFinGen.to_dict()             , within=PositiveIntegers,    doc='retirement   year',                                  )
    model.pAvailability         = Param(model.gg,    initialize=pAvailability.to_dict()             , within=UnitInterval    ,    doc='unit availability',                      mutable=True)
    model.pEFOR                 = Param(model.gg,    initialize=pEFOR.to_dict()                     , within=UnitInterval    ,    doc='EFOR'                                                )
    # model.pRatedLinearVarCost   = Param(model.gg,    initialize=pRatedLinearVarCost.to_dict()       , within=NonNegativeReals,    doc='Linear   variable cost'                              )
    # model.pRatedConstantVarCost = Param(model.gg,    initialize=pRatedConstantVarCost.to_dict()     , within=NonNegativeReals,    doc='Constant variable cost'                              )
    # model.pLinearVarCost        = Param(model.psngg, initialize=pLinearVarCost.to_dict()            , within=NonNegativeReals,    doc='Linear   variable cost'                              )
    # model.pConstantVarCost      = Param(model.psngg, initialize=pConstantVarCost.to_dict()          , within=NonNegativeReals,    doc='Constant variable cost'                              )
    # model.pLinearOMCost         = Param(model.gg,    initialize=pLinearOMCost.to_dict()             , within=NonNegativeReals,    doc='Linear   O&M      cost'                              )
    # # model.pOperReserveCost      = Param(model.gg,    initialize=pOperReserveCost.to_dict()          , within=NonNegativeReals,    doc='Operating reserve cost'                              )
    # model.pCO2EmissionCost      = Param(model.gg,    initialize=pCO2EmissionCost.to_dict()          , within=Reals,               doc='CO2 Emission      cost'                              )
    # model.pCO2EmissionRate      = Param(model.gg,    initialize=pCO2EmissionRate.to_dict()          , within=Reals,               doc='CO2 Emission      rate'                              )
    # model.pStartUpCost          = Param(model.gg,    initialize=pStartUpCost.to_dict()              , within=NonNegativeReals,    doc='Startup  cost'                                       )
    # model.pShutDownCost         = Param(model.gg,    initialize=pShutDownCost.to_dict()             , within=NonNegativeReals,    doc='Shutdown cost'                                       )
    # model.pRampUp               = Param(model.gg,    initialize=pRampUp.to_dict()                   , within=NonNegativeReals,    doc='Ramp up   rate'                                      )
    # model.pRampDw               = Param(model.gg,    initialize=pRampDw.to_dict()                   , within=NonNegativeReals,    doc='Ramp down rate'                                      )
    # model.pUpTime               = Param(model.gg,    initialize=pUpTime.to_dict()                   , within=NonNegativeIntegers, doc='Up    time'                                          )
    # model.pDwTime               = Param(model.gg,    initialize=pDwTime.to_dict()                   , within=NonNegativeIntegers, doc='Down  time'                                          )
    # model.pShiftTime            = Param(model.gg,    initialize=pShiftTime.to_dict()                , within=NonNegativeIntegers, doc='Shift time'                                          )
    model.pGenInvestCost        = Param(model.gg,    initialize=pGenInvestCost.to_dict()            , within=NonNegativeReals,    doc='Generation fixed cost'                               )
    # model.pGenRetireCost        = Param(model.gg,    initialize=pGenRetireCost.to_dict()            , within=Reals           ,    doc='Generation fixed retire cost'                        )
    model.pIndBinUnitInvest     = Param(model.gg,    initialize=pIndBinUnitInvest.to_dict()         , within=Binary          ,    doc='Binary investment decision'                          )
    # model.pIndBinUnitRetire     = Param(model.gg,    initialize=pIndBinUnitRetire.to_dict()         , within=Binary          ,    doc='Binary retirement decision'                          )
    model.pIndBinUnitCommit     = Param(model.gg,    initialize=pIndBinUnitCommit.to_dict()         , within=Binary          ,    doc='Binary commitment decision'                          )
    model.pIndBinStorInvest     = Param(model.gg,    initialize=pIndBinStorInvest.to_dict()         , within=Binary          ,    doc='Storage linked to generation investment'             )
    # model.pIndOperReserve       = Param(model.gg,    initialize=pIndOperReserve.to_dict()           , within=Binary          ,    doc='Indicator of operating reserve'                      )
    model.pEfficiency           = Param(model.gg,    initialize=pEfficiency.to_dict()               , within=UnitInterval    ,    doc='Round-trip efficiency'                               )
    model.pCycleTimeStep        = Param(model.gg,    initialize=pCycleTimeStep.to_dict()            , within=PositiveIntegers,    doc='ESS Storage cycle'                                   )
    model.pOutflowsTimeStep     = Param(model.gg,    initialize=pOutflowsTimeStep.to_dict()         , within=PositiveIntegers,    doc='ESS Outflows cycle'                                  )
    # model.pEnergyTimeStep       = Param(model.gg,    initialize=pEnergyTimeStep.to_dict()           , within=PositiveIntegers,    doc='Unit energy cycle'                                   )
    model.pIniInventory         = Param(model.psngg, initialize=pIniInventory                       , within=NonNegativeReals,    doc='ESS Initial storage',                    mutable=True)
    model.pInitialInventory     = Param(model.gg,    initialize=pInitialInventory.to_dict()         , within=NonNegativeReals,    doc='ESS Initial storage without load levels'             )
    model.pStorageType          = Param(model.gg,    initialize=pStorageType.to_dict()              , within=Any             ,    doc='ESS Storage type'                                    )
    model.pGenLoInvest          = Param(model.gg,    initialize=pGenLoInvest.to_dict()              , within=NonNegativeReals,    doc='Lower bound of the investment decision', mutable=True)
    model.pGenUpInvest          = Param(model.gg,    initialize=pGenUpInvest.to_dict()              , within=NonNegativeReals,    doc='Upper bound of the investment decision', mutable=True)
    # model.pGenLoRetire          = Param(model.gg,    initialize=pGenLoRetire.to_dict()              , within=NonNegativeReals,    doc='Lower bound of the retirement decision', mutable=True)
    # model.pGenUpRetire          = Param(model.gg,    initialize=pGenUpRetire.to_dict()              , within=NonNegativeReals,    doc='Upper bound of the retirement decision', mutable=True)
    model.pGenSensitivity       = Param(model.gg,    initialize=pGenSensitivity.to_dict()           , within=NonNegativeReals,    doc='Sensitivity of the investment decision', mutable=True)
    model.pGenFxInvest          = Param(model.gg,    initialize=pGenFxInvest.to_dict()              , within=NonNegativeReals,    doc='Fixed investment cost',                  mutable=True)
    model.pGenSensiGroup        = Param(model.gg,    initialize=pGenSensiGroup.to_dict()            , within=Any             ,    doc='Sensitivity group',                      mutable=True)
    model.pGenSensiGroupValue   = Param(model.gg,    initialize=pGenSensiGroupValue.to_dict()       , within=Any             ,    doc='Sensitivity group value',                mutable=True)

    model.pLoadLevelDuration    = Param(model.n,     initialize=0                                   , within=NonNegativeIntegers, doc='Load level duration',                    mutable=True)
    for n in model.n:
        model.pLoadLevelDuration[n] = pLoadLevelWeight[n] * model.pDuration[n]

    model.pPeriodProb           = Param(model.ps,    initialize=0.0                                 , within=NonNegativeReals,    doc='Period probability',                     mutable=True)
    for p,sc in model.ps:
        # periods and scenarios are going to be solved together with their weight and probability
        model.pPeriodProb[p,sc] = model.pPeriodWeight[p] * model.pScenProb[p,sc]

    model.L = RangeSet(10)

    model.pLineLossFactor       = Param(model.ln,    initialize=pLineLossFactor.to_dict()           , within=           Reals,    doc='Loss factor'                                         )
    model.pLineR                = Param(model.ln,    initialize=pLineR.to_dict()                    , within=NonNegativeReals,    doc='Resistance'                                          )
    model.pLineX                = Param(model.ln,    initialize=pLineX.to_dict()                    , within=           Reals,    doc='Reactance'                                           )
    model.pLineBsh              = Param(model.ln,    initialize=pLineBsh.to_dict()                  , within=NonNegativeReals,    doc='Susceptance',                            mutable=True)
    model.pLineTAP              = Param(model.ln,    initialize=pLineTAP.to_dict()                  , within=NonNegativeReals,    doc='Tap changer',                            mutable=True)
    model.pLineLength           = Param(model.ln,    initialize=pLineLength.to_dict()               , within=NonNegativeReals,    doc='Length',                                 mutable=True)
    model.pPeriodIniNet         = Param(model.ln,    initialize=pPeriodIniNet.to_dict()             , within=PositiveIntegers,    doc='Installation period'                                 )
    model.pPeriodFinNet         = Param(model.ln,    initialize=pPeriodFinNet.to_dict()             , within=PositiveIntegers,    doc='Retirement   period'                                 )
    model.pLineVoltage          = Param(model.ln,    initialize=pLineVoltage.to_dict()              , within=NonNegativeReals,    doc='Voltage'                                             )
    model.pLineNTCFrw           = Param(model.ln,    initialize=pLineNTCFrw.to_dict()               , within=NonNegativeReals,    doc='NTC forward'                                         )
    # model.pLineNTCBck           = Param(model.ln,    initialize=pLineNTCBck.to_dict()               , within=NonNegativeReals,    doc='NTC backward'                                        )
    model.pNetFixedCost         = Param(model.ln,    initialize=pNetFixedCost.to_dict()             , within=NonNegativeReals,    doc='Network fixed cost'                                  )
    model.pIndBinLineInvest     = Param(model.ln,    initialize=pIndBinLineInvest.to_dict()         , within=Binary          ,    doc='Binary investment decision'                          )
    # model.pIndBinLineSwitch     = Param(model.ln,    initialize=pIndBinLineSwitch.to_dict()         , within=Binary          ,    doc='Binary switching  decision'                          )
    # model.pSwOnTime             = Param(model.ln,    initialize=pSwitchOnTime.to_dict()             , within=NonNegativeIntegers, doc='Minimum switching on  time'                          )
    # model.pSwOffTime            = Param(model.ln,    initialize=pSwitchOffTime.to_dict()            , within=NonNegativeIntegers, doc='Minimum switching off time'                          )
    # model.pBigMFlowBck          = Param(model.ln,    initialize=pBigMFlowBck.to_dict()              , within=NonNegativeReals,    doc='Maximum backward capacity',              mutable=True)
    model.pBigMFlowFrw          = Param(model.ln,    initialize=pBigMFlowFrw.to_dict()              , within=NonNegativeReals,    doc='Maximum forward  capacity',              mutable=True)
    model.pMaxTheta             = Param(model.psnnd, initialize=pMaxTheta                           , within=NonNegativeReals,    doc='Maximum voltage angle',                  mutable=True)
    model.pAngMin               = Param(model.ln,    initialize=pAngMin.to_dict()                   , within=Reals,               doc='Minimum phase angle diff',               mutable=True)
    model.pAngMax               = Param(model.ln,    initialize=pAngMax.to_dict()                   , within=Reals,               doc='Maximum phase angle diff',               mutable=True)
    model.pNetLoInvest          = Param(model.ln,    initialize=pNetLoInvest.to_dict()              , within=NonNegativeReals,    doc='Lower bound of the investment decision', mutable=True)
    model.pNetUpInvest          = Param(model.ln,    initialize=pNetUpInvest.to_dict()              , within=NonNegativeReals,    doc='Upper bound of the investment decision', mutable=True)
    model.pNetSensitivity       = Param(model.ln,    initialize=pNetSensitivity.to_dict()           , within=NonNegativeReals,    doc='Sensitivity of the investment decision', mutable=True)
    model.pNetFxInvest          = Param(model.ln,    initialize=pNetFxInvest.to_dict()              , within=NonNegativeReals,    doc='Fixed cost of the investment decision' , mutable=True)
    model.pNetSensiGroup        = Param(model.ln,    initialize=pNetSensiGroup.to_dict()            , within=NonNegativeIntegers, doc='Sensitivity group'                     , mutable=True)
    model.pNetSensiGroupValue   = Param(model.ln,    initialize=pNetSensiGroupValue.to_dict()       , within=NonNegativeReals,    doc='Sensitivity group value'               , mutable=True)
    model.pLineDelta_S          = Param(model.la,         initialize=0.,                              within=NonNegativeReals,    doc='Delta of Smax splitted by L',           mutable=True)
    model.pLineM                = Param(model.la,model.L, initialize=0.,                              within=NonNegativeReals,    doc='M partitions of Delta Smax',            mutable=True)

    for la in model.la:
        model.pLineDelta_S[la] = model.pLineNTCFrw[la]   / len(model.L)
        for l in model.L:
            model.pLineM[la,l] = (2*l-1)*model.pLineDelta_S[la]

    # if unit availability = 0 changed to 1
    for g in model.g:
        if  model.pAvailability[g]() == 0.0:
            model.pAvailability[g]   =  1.0

    # if line length = 0 changed to geographical distance with an additional 10%
    for ni,nf,cc in model.la:
        if  model.pLineLength[ni,nf,cc]() == 0.0:
            model.pLineLength[ni,nf,cc]   =  1.1 * 6371 * 2 * math.asin(math.sqrt(math.pow(math.sin((model.pNodeLat[nf]-model.pNodeLat[ni])*math.pi/180/2),2) + math.cos(model.pNodeLat[ni]*math.pi/180)*math.cos(model.pNodeLat[nf]*math.pi/180)*math.pow(math.sin((model.pNodeLon[nf]-model.pNodeLon[ni])*math.pi/180/2),2)))

    #%% variables
    model.vTotalSCost     = Var(                              within=NonNegativeReals,                                                                                          doc='total system                         cost      [MEUR]')
    model.vTotalICost     = Var(                              within=NonNegativeReals,                                                                                          doc='total system investment              cost      [MEUR]')
    model.vTotalFCost     = Var(model.p,                      within=NonNegativeReals,                                                                                          doc='total system fixed                   cost      [MEUR]')
    model.vTotalGCost     = Var(model.psn,                    within=NonNegativeReals,                                                                                          doc='total variable generation  operation cost      [MEUR]')
    model.vTotalCCost     = Var(model.psn,                    within=NonNegativeReals,                                                                                          doc='total variable consumption operation cost      [MEUR]')
    model.vTotalECost     = Var(model.psn,                    within=NonNegativeReals,                                                                                          doc='total system emission                cost      [MEUR]')
    model.vTotalRCost     = Var(model.psn,                    within=NonNegativeReals,                                                                                          doc='total system reliability             cost      [MEUR]')
    model.vTotalOutputP   = Var(model.psng ,                  within=NonNegativeReals, bounds=lambda model,p,sc,n,g : (0.0,                    pMaxPower          [p,sc,n,g]),  doc='total output of the unit                         [GW]')
    model.vTotalOutputQ   = Var(model.psnnr,                  within=           Reals, bounds=lambda model,p,sc,n,nr: (pRatedMinPowerQ[nr],    pRatedMaxPowerQ    [      nr]),  doc='total output of the unit                       [GVAr]')
    model.vOutput2ndBlock = Var(model.psnnr,                  within=NonNegativeReals, bounds=lambda model,p,sc,n,nr: (0.0,                    pMaxPower2ndBlock [p,sc,n,nr]),  doc='second block of the unit                         [GW]')
    model.vEnergyInflows  = Var(model.psnes,                  within=NonNegativeReals, bounds=lambda model,p,sc,n,es: (0.0,                    pEnergyInflows    [p,sc,n,es]),  doc='unscheduled inflows  of candidate ESS units      [GW]')
    model.vEnergyOutflows = Var(model.psnes,                  within=NonNegativeReals, bounds=lambda model,p,sc,n,es: (0.0,max(pMaxPower  [p,sc,n,es],pMaxCharge [p,sc,n,es])), doc='scheduled   outflows of all       ESS units      [GW]')
    model.vESSInventory   = Var(model.psnes,                  within=NonNegativeReals, bounds=lambda model,p,sc,n,es: (        pMinStorage[p,sc,n,es],pMaxStorage[p,sc,n,es]),  doc='ESS inventory                                   [GWh]')
    model.vESSSpillage    = Var(model.psnes,                  within=NonNegativeReals,                                                                                          doc='ESS spillage                                    [GWh]')
    model.vESSTotalCharge = Var(model.psnes,                  within=NonNegativeReals, bounds=lambda model,p,sc,n,es: (0.0,                   pMaxCharge         [p,sc,n,es]),  doc='ESS total charge power                           [GW]')
    model.vCharge2ndBlock = Var(model.psnes,                  within=NonNegativeReals, bounds=lambda model,p,sc,n,es: (0.0,                   pMaxCharge2ndBlock [p,sc,n,es]),  doc='ESS       charge power                           [GW]')
    model.vENS            = Var(model.psnnd, initialize= 0.0 ,within=NonNegativeReals, bounds=lambda model,p,sc,n,nd: (0.0,                                           1.0000),  doc='energy not served in node                        [GW]')
    model.vW              = Var(model.psnnd, initialize= 1.0 ,within=NonNegativeReals,                                                                                          doc='squared voltage in node                        [p.u.]')

    if pIndBinGenInvest == 0:
        model.vGenerationInvest  = Var(model.pgc,           within=UnitInterval,                                                                                               doc='generation investment decision exists in a year [0,1]')
    else:
        model.vGenerationInvest  = Var(model.pgc,           within=Binary,                                                                                                     doc='generation investment decision exists in a year {0,1}')

    if pIndBinNetInvest == 0:
        model.vNetworkInvest     = Var(model.plc,           within=UnitInterval,                                                                                               doc='network    investment decision exists in a year [0,1]')
    else:
        model.vNetworkInvest     = Var(model.plc,           within=Binary,                                                                                                     doc='network    investment decision exists in a year {0,1}')

    if pIndBinGenOperat == 0:
        model.vCommitment        = Var(model.psnnr,         within=UnitInterval,     initialize=0.0,                                                                           doc='commitment         of the unit                  [0,1]')
        model.vStartUp           = Var(model.psnnr,         within=UnitInterval,     initialize=0.0,                                                                           doc='startup            of the unit                  [0,1]')
        model.vShutDown          = Var(model.psnnr,         within=UnitInterval,     initialize=0.0,                                                                           doc='shutdown           of the unit                  [0,1]')
    else:
        model.vCommitment        = Var(model.psnnr,         within=Binary,           initialize=0  ,                                                                           doc='commitment         of the unit                  {0,1}')
        model.vStartUp           = Var(model.psnnr,         within=Binary,           initialize=0  ,                                                                           doc='startup            of the unit                  {0,1}')
        model.vShutDown          = Var(model.psnnr,         within=Binary,           initialize=0  ,                                                                           doc='shutdown           of the unit                  {0,1}')

    # relax binary condition in generation and network investment decisions
    for p,gc in model.pgc:
        if pIndBinGenInvest != 0 and pIndBinUnitInvest[gc      ] == 0:
            model.vGenerationInvest[p,gc      ].domain = UnitInterval
        if pIndBinGenInvest == 2:
            model.vGenerationInvest[p,gc      ].fix(0)
    for p,ni,nf,cc in model.plc:
        if pIndBinNetInvest != 0 and pIndBinLineInvest[ni,nf,cc] == 0:
            model.vNetworkInvest   [p,ni,nf,cc].domain = UnitInterval
        if pIndBinNetInvest == 2:
            model.vNetworkInvest   [p,ni,nf,cc].fix(0)

    # relax binary condition in unit generation, startup and shutdown decisions
    for p,sc,n,nr in model.psnnr:
        if pIndBinUnitCommit[nr] == 0:
            model.vCommitment   [p,sc,n,nr].domain = UnitInterval
            model.vStartUp      [p,sc,n,nr].domain = UnitInterval
            model.vShutDown     [p,sc,n,nr].domain = UnitInterval

    if pIndBinSingleNode == 0:
        # model.vS      = Var(model.ps, model.n, model.la, within=Reals,            bounds=lambda model,p,sc,n,*la: (            -pLineNTCFrw[la]      ,pLineNTCFrw[la]    ),    doc='Sine term          [p.u.]')
        # model.vC      = Var(model.ps, model.n, model.la, within=Reals,            bounds=lambda model,p,sc,n,*la: (            -pLineNTCFrw[la]      ,pLineNTCFrw[la]    ),    doc='Cosine term        [p.u.]')
        # model.vPfr    = Var(model.ps, model.n, model.la, within=Reals,            bounds=lambda model,p,sc,n,*la: (            -pLineNTCFrw[la]      ,pLineNTCFrw[la]    ),    doc='P flow from i to j [p.u.]')
        # model.vPto    = Var(model.ps, model.n, model.la, within=Reals,            bounds=lambda model,p,sc,n,*la: (            -pLineNTCFrw[la]      ,pLineNTCFrw[la]    ),    doc='P flow from j to i [p.u.]')
        # model.vQfr    = Var(model.ps, model.n, model.la, within=Reals,            bounds=lambda model,p,sc,n,*la: (            -pLineNTCFrw[la]      ,pLineNTCFrw[la]    ),    doc='Q flow from i to j [p.u.]')
        # model.vQto    = Var(model.ps, model.n, model.la, within=Reals,            bounds=lambda model,p,sc,n,*la: (            -pLineNTCFrw[la]      ,pLineNTCFrw[la]    ),    doc='Q flow from j to i [p.u.]')
        model.vS         = Var(model.ps, model.n, model.la,          initialize= 0.0, within=Reals,                                                                                                      doc='Sine term          [p.u.]')
        model.vC         = Var(model.ps, model.n, model.la,          initialize= 1.0, within=Reals,                                                                                                      doc='Cosine term        [p.u.]')
        model.vPfr       = Var(model.ps, model.n, model.la,          initialize= 0.0, within=Reals,                                                                                                      doc='P flow from i to j [p.u.]')
        model.vPto       = Var(model.ps, model.n, model.la,          initialize= 0.0, within=Reals,                                                                                                      doc='P flow from j to i [p.u.]')
        model.vQfr       = Var(model.ps, model.n, model.la,          initialize= 0.0, within=Reals,                                                                                                      doc='Q flow from i to j [p.u.]')
        model.vQto       = Var(model.ps, model.n, model.la,          initialize= 0.0, within=Reals,                                                                                                      doc='Q flow from j to i [p.u.]')
        model.vDelta_S   = Var(model.ps, model.n, model.la, model.L, initialize= 0.0, within=NonNegativeReals,                                                                                           doc='Delta Active Power Flow                   [  GW]')
        model.vDelta_C   = Var(model.ps, model.n, model.la, model.L, initialize= 0.0, within=NonNegativeReals,                                                                                           doc='Delta Reactive Power Flow                 [Gvar]')
        model.vS_max     = Var(model.ps, model.n, model.la,          initialize= 0.0, within=NonNegativeReals,                                                                                           doc='Maximum bound of the sine term            [p.u.]')
        model.vS_min     = Var(model.ps, model.n, model.la,          initialize= 0.0, within=NonNegativeReals,                                                                                           doc='Minimum bound of the sine term            [p.u.]')
        model.vC_max     = Var(model.ps, model.n, model.la,          initialize= 0.0, within=NonNegativeReals,                                                                                           doc='Maximum bound of the cosine term          [p.u.]')
        model.vC_min     = Var(model.ps, model.n, model.la,          initialize= 0.0, within=NonNegativeReals,                                                                                           doc='Minimum bound of the cosine term          [p.u.]')
        model.vDelta_Pfr = Var(model.ps, model.n, model.la, model.L, initialize= 0.0, within=NonNegativeReals,                                                                                           doc='Delta Active Power Flow                   [  GW]')
        model.vDelta_Qfr = Var(model.ps, model.n, model.la, model.L, initialize= 0.0, within=NonNegativeReals,                                                                                           doc='Delta Reactive Power Flow                 [Gvar]')
        model.vPfr_max   = Var(model.ps, model.n, model.la,          initialize= 0.0, within=NonNegativeReals,                                                                                           doc='Maximum bound of the sine term            [p.u.]')
        model.vPfr_min   = Var(model.ps, model.n, model.la,          initialize= 0.0, within=NonNegativeReals,                                                                                           doc='Minimum bound of the sine term            [p.u.]')
        model.vQfr_max   = Var(model.ps, model.n, model.la,          initialize= 0.0, within=NonNegativeReals,                                                                                           doc='Maximum bound of the cosine term          [p.u.]')
        model.vQfr_min   = Var(model.ps, model.n, model.la,          initialize= 0.0, within=NonNegativeReals,                                                                                           doc='Minimum bound of the cosine term          [p.u.]')
        model.vDelta_Pto = Var(model.ps, model.n, model.la, model.L, initialize= 0.0, within=NonNegativeReals,                                                                                           doc='Delta Active Power Flow                   [  GW]')
        model.vDelta_Qto = Var(model.ps, model.n, model.la, model.L, initialize= 0.0, within=NonNegativeReals,                                                                                           doc='Delta Reactive Power Flow                 [Gvar]')
        model.vPto_max   = Var(model.ps, model.n, model.la,          initialize= 0.0, within=NonNegativeReals,                                                                                           doc='Maximum bound of the sine term            [p.u.]')
        model.vPto_min   = Var(model.ps, model.n, model.la,          initialize= 0.0, within=NonNegativeReals,                                                                                           doc='Minimum bound of the sine term            [p.u.]')
        model.vQto_max   = Var(model.ps, model.n, model.la,          initialize= 0.0, within=NonNegativeReals,                                                                                           doc='Maximum bound of the cosine term          [p.u.]')
        model.vQto_min   = Var(model.ps, model.n, model.la,          initialize= 0.0, within=NonNegativeReals,                                                                                           doc='Minimum bound of the cosine term          [p.u.]')
    else:
        model.vFlow      = Var(model.ps, model.n, model.la, initialize= 0.0 , within=Reals,                                                                                                      doc='flow               [GW]')
    model.vTheta         = Var(model.ps, model.n, model.nd, initialize= 0.0 , within=Reals,            bounds=lambda model,p,sc,n, nd: (            -pMaxTheta[p,sc,n,nd],pMaxTheta[p,sc,n,nd]),    doc='voltage angle     [rad]')

    if pIndBinNetInvest == 0:
        model.vShuntInvest     = Var(model.pshc,            within=UnitInterval,                                                                                               doc='bus shunt  investment decision exists in a year [0,1]')
    else:
        model.vShuntInvest     = Var(model.pshc,            within=Binary,                                                                                                     doc='bus shunt  investment decision exists in a year {0,1}')

    model.vBusShuntP  = Var(model.ps, model.n, model.sh, within=Reals,                                                                                                         doc='Active   power from/to bu shunt device [p.u.]')
    model.vBusShuntQ  = Var(model.ps, model.n, model.sh, within=Reals,                                                                                                         doc='Reactive power from/to bu shunt device [p.u.]')

    model.vGenSensi   = Var(model.p,           model.gc, within=NonNegativeReals, bounds=lambda model,p,     gc:  (              0.0                 , 1.0               ),    doc='Variable to represent a small variation of a investment decision [p.u.]')
    model.vNetSensi   = Var(model.p,           model.lc, within=NonNegativeReals, bounds=lambda model,p,     *lc: (              0.0                 , 1.0               ),    doc='Variable to represent a small variation of a investment decision [p.u.]')
    model.vGenSensiGr = Var(model.p,           model.gs, within=NonNegativeReals, bounds=lambda model,p,     gs:  (              0.0                 , 1.0               ),    doc='Variable to represent a small variation of a group of investment [p.u.]')
    model.vNetSensiGr = Var(model.p,           model.ns, within=NonNegativeReals, bounds=lambda model,p,     ns:  (              0.0                 , 1.0               ),    doc='Variable to represent a small variation of a group of investment [p.u.]')

    # fix the candidate units which not participate in the sensitivity analysis
    for p,gc in model.pgc:
        #
        if pGenSensitivity[gc] == 0:
            model.vGenSensi[  p,gc].fix(0.0)
    for p,gs in model.p*model.gs:
        if gs == 'Gr0':
            model.vGenSensiGr[p,gs].fix(1.0)

    for p,ni,nf,cc in model.plc:
        #
        if pNetSensitivity[ni,nf,cc] == 0:
            model.vNetSensi[  p,ni,nf,cc].fix(0.0)
    for p,ns in model.p*model.ns:
        if ns == 'Gr0':
            model.vNetSensiGr[p,ns].fix(1.0)

    # fix the must-run units and their output
    for p,sc,n,g  in model.psn*model.g :
        # must run units must produce at least their minimum output
        if pMustRun[g] == 1:
            model.vTotalOutputP[p,sc,n,g].setlb(pMinPower[p,sc,n,g])
        # if no max power, no total output
        if pMaxPower[p,sc,n,g] == 0.0:
            model.vTotalOutputP[p,sc,n,g].fix(0.0)

    for p,sc,n,nr in model.psnnr:
        # must run units or units with no minimum power or ESS existing units are always committed and must produce at least their minimum output
        # not applicable to mutually exclusive units
        if (pMustRun[nr] == 1 or (pMinPower[p,sc,n,nr] == 0.0 and pConstantVarCost[nr] == 0.0) or nr in model.es) and nr not in model.ec:
            model.vCommitment    [p,sc,n,nr].fix(1)
            model.vStartUp       [p,sc,n,nr].fix(0)
            model.vShutDown      [p,sc,n,nr].fix(0)
        # if min and max power coincide there are neither second block, nor operating reserve
        if  pMaxPower2ndBlock[p,sc,n,nr] ==  0.0:
            model.vOutput2ndBlock[p,sc,n,nr].fix(0.0)

    for p,sc,n,es in model.psnes:
        # ESS with no charge capacity or not storage capacity can't charge
        if pMaxCharge        [p,sc,n,es] ==  0.0:
            model.vESSTotalCharge[p,sc,n,es].fix(0.0)
        if pMaxCharge        [p,sc,n,es] ==  0.0 and pMaxPower[p,sc,n,es] == 0.0:
            model.vESSInventory  [p,sc,n,es].fix(0.0)
            model.vESSSpillage   [p,sc,n,es].fix(0.0)
        if pMaxCharge2ndBlock[p,sc,n,es] ==  0.0:
            model.vCharge2ndBlock[p,sc,n,es].fix(0.0)
        if pMaxStorage       [p,sc,n,es] ==  0.0:
            model.vESSInventory  [p,sc,n,es].fix(0.0)

    # thermal and RES units ordered by increasing variable operation cost, excluding reactive generating units
    # determine the initial committed units and their output
    pInitialOutput = pd.Series([0.0]*len(model.g), model.g)
    pInitialUC     = pd.Series([0.0]*len(model.g), model.g)
    # pSystemOutput  = 0.0
    # for go in model.go:
    #     (p1,sc1,n1) = next(iter(model.psn))
    #     if pSystemOutput < sum(pDemand[p1,sc1,n1,nd] for nd in model.nd):
    #         if go in model.r:
    #             pInitialOutput[go] = pMaxPower[p1,sc1,n1,go]
    #         else:
    #             pInitialOutput[go] = pMinPower[p1,sc1,n1,go]
    #         pInitialUC    [go] = 1
    #         pSystemOutput     += pInitialOutput[go]
    # model.go = [k for k in sorted(pLinearVarCost, key=pLinearVarCost.__getitem__)                      ]
    model.go = pLinearVarCost.sort_values().index

    for p,sc in model.ps:
        if len(model.n):
            # determine the first load level of each stage
            n1 = next(iter(model.n))
            # commit the units and their output at the first load level of each stage
            pSystemOutput = 0.0
            for nr in model.nr:
                if pSystemOutput < sum(pDemandP[p,sc,n1,nd] for nd in model.nd) and pMustRun[nr] == 1:
                    pInitialOutput[nr] = pMaxPower[p,sc,n1,nr]
                    pInitialUC    [nr] = 1
                    pSystemOutput               += pInitialOutput[nr]

            # determine the initial committed units and their output at the first load level of each period, scenario, and stage
            for go in model.go:
                if pSystemOutput < sum(pDemandP[p,sc,n1,nd] for nd in model.nd) and pMustRun[go] != 1:
                    if go in model.r:
                        pInitialOutput[go] = pMaxPower[p,sc,n1,go]
                    else:
                        pInitialOutput[go] = pMinPower[p,sc,n1,go]
                    pInitialUC[go] = 1
                    pSystemOutput = pSystemOutput + pInitialOutput[go]

    # fixing the ESS inventory at the last load level of the stage for every period and scenario if between storage limits
    for p,sc,es in model.ps*model.es:
        if pInitialInventory[es] >= pMinStorage[p,sc,model.n.last(),es] and pInitialInventory[es] <= pMaxStorage[p,sc,model.n.last(),es]:
                    model.vESSInventory[p,sc,model.n.last(),es].fix(pInitialInventory[es])


    # fixing the ESS inventory at the end of the following pCycleTimeStep (weekly, yearly), i.e., for daily ESS is fixed at the end of the week, for weekly/monthly ESS is fixed at the end of the year
    for p,sc,n,es in model.psnes:
         if pStorageType[es] == 'Hourly'  and model.n.ord(n) % int(  24/pTimeStep) == 0 and pInitialInventory[es] >= pMinStorage[p,sc,n,es] and pInitialInventory[es] <= pMaxStorage[p,sc,n,es]:
                 model.vESSInventory[p,sc,n,es].fix(pInitialInventory[es])
         if pStorageType[es] == 'Daily'   and model.n.ord(n) % int( 168/pTimeStep) == 0 and pInitialInventory[es] >= pMinStorage[p,sc,n,es] and pInitialInventory[es] <= pMaxStorage[p,sc,n,es]:
                 model.vESSInventory[p,sc,n,es].fix(pInitialInventory[es])
         if pStorageType[es] == 'Weekly'  and model.n.ord(n) % int(8736/pTimeStep) == 0 and pInitialInventory[es] >= pMinStorage[p,sc,n,es] and pInitialInventory[es] <= pMaxStorage[p,sc,n,es]:
                 model.vESSInventory[p,sc,n,es].fix(pInitialInventory[es])
         if pStorageType[es] == 'Monthly' and model.n.ord(n) % int(8736/pTimeStep) == 0 and pInitialInventory[es] >= pMinStorage[p,sc,n,es] and pInitialInventory[es] <= pMaxStorage[p,sc,n,es]:
                 model.vESSInventory[p,sc,n,es].fix(pInitialInventory[es])

    for p,sc,n,es in model.psnes:
        if pEnergyInflows [p,sc,n,es] == 0.0:
            model.vEnergyInflows        [p,sc,n,es].fix(0.0)

    # if there are no energy outflows no variable is needed
    for es in model.es:
        if sum(pEnergyOutflows[p,sc,n,es] for p,sc,n in model.psn) == 0:
            for p,sc,n in model.psn:
                model.vEnergyOutflows[p,sc,n,es].fix(0.0)

    # fixing the voltage angle of the reference node for each scenario, period, and load level
    # if pIndBinSingleNode == 0:
    #     for p,sc,n in model.psn:
    #         # model.vTheta[p,sc,n,model.rf.first()].fix(0.0)
    #         model.vW    [p,sc,n,model.rf.first()].fix(1.1025)

    # fixing the ENS in nodes with no demand
    for p,sc,n,nd in model.psnnd:
        if pDemandP[p,sc,n,nd] == 0.0:
            model.vENS[p,sc,n,nd].fix(0.0)

    # tolerance to consider 0 a number
    pEpsilon = 1e-3
    for p,gc in model.pgc:
        if  pGenLoInvest   [  gc      ]   <       pEpsilon:
            pGenLoInvest   [  gc      ]   = 0
        if  pGenUpInvest   [  gc      ]   <       pEpsilon:
            pGenUpInvest   [  gc      ]   = 0
        if  pGenLoInvest   [  gc      ]   > 1.0 - pEpsilon:
            pGenLoInvest   [  gc      ]   = 1
        if  pGenUpInvest   [  gc      ]   > 1.0 - pEpsilon:
            pGenUpInvest   [  gc      ]   = 1
        if  pGenLoInvest   [  gc      ]   >   pGenUpInvest[gc      ]:
            pGenLoInvest   [  gc      ]   =   pGenUpInvest[gc      ]
        model.vGenerationInvest[p,gc      ].setlb(pGenLoInvest[gc      ])
        model.vGenerationInvest[p,gc      ].setub(pGenUpInvest[gc      ])
    for p,ni,nf,cc in model.plc:
        if  pNetLoInvest   [  ni,nf,cc]   <       pEpsilon:
            pNetLoInvest   [  ni,nf,cc]   = 0
        if  pNetUpInvest   [  ni,nf,cc]   <       pEpsilon:
            pNetUpInvest   [  ni,nf,cc]   = 0
        if  pNetLoInvest   [  ni,nf,cc]   > 1.0 - pEpsilon:
            pNetLoInvest   [  ni,nf,cc]   = 1
        if  pNetUpInvest   [  ni,nf,cc]   > 1.0 - pEpsilon:
            pNetUpInvest   [  ni,nf,cc]   = 1
        if  pNetLoInvest   [  ni,nf,cc]   >   pNetUpInvest[ni,nf,cc]:
            pNetLoInvest   [  ni,nf,cc]   =   pNetUpInvest[ni,nf,cc]
        model.vNetworkInvest   [p,ni,nf,cc].setlb(pNetLoInvest[ni,nf,cc])
        model.vNetworkInvest   [p,ni,nf,cc].setub(pNetUpInvest[ni,nf,cc])

    SettingUpDataTime = time.time() - StartTime
    StartTime         = time.time()
    print('Setting up input data                 ... ', round(SettingUpDataTime), 's')
    # tolerance to consider avoid division by 0
    pEpsilon = 1e-6

    def eTotalSCost(model):
        return model.vTotalSCost
    model.eTotalSCost            = Objective(rule=eTotalSCost, sense=minimize, doc='total system cost [MEUR]')

    def eTotalTCost(model):
        return model.vTotalSCost == model.vTotalICost + sum(pDiscountFactor[p] * pPeriodProb[p,sc] * (model.vTotalGCost[p,sc,n] + model.vTotalCCost[p,sc,n] + model.vTotalECost[p,sc,n] + model.vTotalRCost[p,sc,n]) for p,sc,n in model.psn)
    model.eTotalTCost            = Constraint(rule=eTotalTCost, doc='total system cost [MEUR]')

    def eTotalICost(model):
        return model.vTotalICost == sum(pDiscountFactor[p] * model.vTotalFCost[p] for p in model.p)
    model.eTotalICost            = Constraint(rule=eTotalICost, doc='system fixed    cost [MEUR]')

    def eTotalFCost(model,p):
        return model.vTotalFCost[p] == sum(pGenInvestCost[gc] * model.vGenerationInvest[p,gc] for gc in model.gc) + sum(pNetFixedCost[ni,nf,cc] * model.vNetworkInvest[p,ni,nf,cc] for ni,nf,cc in model.lc)
    model.eTotalFCost            = Constraint(model.p, rule=eTotalFCost, doc='system fixed    cost [MEUR]')

    def eConsecutiveGenInvest(model,p,gc):
        if p != model.p.first():
            return model.vGenerationInvest[model.p.prev(p,1),gc      ] <= model.vGenerationInvest[p,gc      ]
        else:
            return Constraint.Skip
    model.eConsecutiveGenInvest  = Constraint(model.p, model.gc, rule=eConsecutiveGenInvest, doc='generation investment in consecutive periods')

    def eConsecutiveNetInvest(model,p,ni,nf,cc):
        if p != model.p.first():
            return model.vNetworkInvest   [model.p.prev(p,1),ni,nf,cc] <= model.vNetworkInvest   [p,ni,nf,cc]
        else:
            return Constraint.Skip
    model.eConsecutiveNetInvest  = Constraint(model.p, model.lc, rule=eConsecutiveNetInvest, doc='network    investment in consecutive periods')

    def eTotalGCost(model,p,sc,st,n):
        if (st,n) in model.s2n:
            return model.vTotalGCost[p,sc,n] == (sum(pLoadLevelDuration[n] * pLinearVarCost  [nr] * model.vTotalOutputP[p,sc,n,nr]                      +
                                                     pLoadLevelDuration[n] * pConstantVarCost[nr] * model.vCommitment [p,sc,n,nr]                      +
                                                     pLoadLevelDuration[n] * pStartUpCost    [nr] * model.vStartUp [p,sc,n,nr]                         +
                                                     pLoadLevelDuration[n] * pShutDownCost   [nr] * model.vShutDown[p,sc,n,nr]    for nr in model.nr)  +
                                                 sum(pLoadLevelDuration[n] * pLinearOMCost   [r ] * model.vTotalOutputP[p,sc,n,r ] for r  in model.r ) )
        else:
            return Constraint.Skip
    model.eTotalGCost            = Constraint(model.ps, model.st, model.n, rule=eTotalGCost, doc='system variable generation operation cost [MEUR]')

    def eTotalCCost(model,p,sc,st,n):
        if (st,n) in model.s2n:
            return model.vTotalCCost    [p,sc,n] == sum(pLoadLevelDuration[n] * pLinearVarCost  [es] * model.vESSTotalCharge[p,sc,n,es] for es in model.es)
        else:
            return Constraint.Skip
    model.eTotalCCost            = Constraint(model.ps, model.st, model.n, rule=eTotalCCost, doc='system variable consumption operation cost [MEUR]')

    def eTotalECost(model,p,sc,st,n):
        if (st,n) in model.s2n and sum(pCO2EmissionCost[nr] for nr in model.nr):
            return model.vTotalECost[p,sc,n] == sum(pLoadLevelDuration[n] * pCO2EmissionCost[nr] * model.vTotalOutputP  [p,sc,n,nr] for nr in model.nr)
        else:
            return Constraint.Skip
    model.eTotalECost            = Constraint(model.ps, model.st, model.n, rule=eTotalECost, doc='system emission cost [MEUR]')

    def eTotalRCost(model,p,sc,st,n):
        if (st,n) in model.s2n:
            return     model.vTotalRCost[p,sc,n] == sum(pLoadLevelDuration[n] * pENSCost * (pDemandP[p,sc,n,nd]+pDemandQ[p,sc,n,nd]) * model.vENS[p,sc,n,nd] for nd in model.nd)
        else:
            return Constraint.Skip
    model.eTotalRCost            = Constraint(model.ps, model.st, model.n, rule=eTotalRCost, doc='system reliability cost [MEUR]')

    GeneratingOFTime = time.time() - StartTime
    StartTime        = time.time()
    print('Generating objective function         ... ', round(GeneratingOFTime), 's')

    #%% constraints
    def eInstalGenComm(model,p,sc,st,n,gc):
        if (st,n) in model.s2n and gc in model.nr and gc not in model.es and pMustRun[gc] == 0 and (pMinPower[p,sc,n,gc] > 0.0 or pConstantVarCost[gc] > 0.0):
            return model.vCommitment[p,sc,n,gc]                                 <= model.vGenerationInvest[p,gc]
        else:
            return Constraint.Skip
    model.eInstalGenComm         = Constraint(model.ps, model.st, model.n, model.gc, rule=eInstalGenComm, doc='commitment if installed unit [p.u.]')

    def eInstalESSComm(model,p,sc,st,n,ec):
        if (st,n) in model.s2n and pIndBinStorInvest[ec]:
            return model.vCommitment[p,sc,n,ec]                                 <= model.vGenerationInvest[p,ec]
        else:
            return Constraint.Skip
    model.eInstalESSComm         = Constraint(model.ps, model.st, model.n, model.ec, rule=eInstalESSComm, doc='commitment if ESS unit [p.u.]')

    def eInstalGenCap(model,p,sc,st,n,gc):
        if (st,n) in model.s2n and pMaxPower[p,sc,n,gc]:
            return model.vTotalOutputP   [p,sc,n,gc] / pMaxPower[p,sc,n,gc] <= model.vGenerationInvest[p,gc]
        else:
            return Constraint.Skip
    model.eInstalGenCap          = Constraint(model.ps, model.st, model.n, model.gc, rule=eInstalGenCap, doc='output if installed gen unit [p.u.]')

    def eInstalConESS(model,p,sc,st,n,ec):
        if (st,n) in model.s2n and pMaxCharge[p,sc,n,ec]:
            return model.vESSTotalCharge[p,sc,n,ec] / pMaxCharge[p,sc,n,ec] <= model.vGenerationInvest[p,ec]
        else:
            return Constraint.Skip
    model.eInstalConESS          = Constraint(model.ps, model.st, model.n, model.ec, rule=eInstalConESS, doc='consumption if installed ESS unit [p.u.]')

    def eAdequacyReserveMargin(model,p,ar):
        if pReserveMargin[ar] and sum(1 for g in model.g if (ar,g) in model.a2g) and len(model.gc):
            return ((sum(                                 pRatedMaxPowerP[g ] * pAvailability[g ] / (1.0-pEFOR[g ]) for g  in model.g  if (ar,g ) in model.a2g and g not in model.gc) +
                     sum(model.vGenerationInvest[p,gc]  * pRatedMaxPowerP[gc] * pAvailability[gc] / (1.0-pEFOR[gc]) for gc in model.gc if (ar,gc) in model.a2g                     )) >= pPeakDemand[ar] * pReserveMargin[ar])
        else:
            return Constraint.Skip
    model.eAdequacyReserveMargin = Constraint(model.p, model.ar, rule=eAdequacyReserveMargin, doc='system adequacy reserve margin [p.u.]')

    GeneratingInvTime = time.time() - StartTime
    StartTime         = time.time()
    print('Generating operation & investment     ... ', round(GeneratingInvTime), 's')

    # incoming and outgoing lines (lin) (lout)
    lin   = defaultdict(list)
    lout  = defaultdict(list)
    for ni,nf,cc in model.la:
        lin  [nf].append((ni,cc))
        lout [ni].append((nf,cc))

    def eBalanceP(model,p,sc,st,n,nd):
        if (st,n) in model.s2n and sum(1 for g in model.g if (nd,g) in model.n2g) + sum(1 for lout in lout[nd]) + sum(1 for ni,cc in lin[nd]):
            return (sum(model.vTotalOutputP[p,sc,n,g  ] for g     in model.g  if (nd,g  ) in model.n2g ) - sum(model.vESSTotalCharge[p,sc,n,es] for es in model.es if (nd,es) in model.n2g)
                    - pDemandP[p,sc,n,nd]*(1-model.vENS[p,sc,n,nd])
                    - sum(model.vBusShuntP[p,sc,n,sh ] for sh    in model.sh if (nd,sh  ) in model.n2sh)
                    - sum(model.vFlow[p,sc,n,nd,lout ] for lout  in lout[nd] if (nd,lout) in model.lad ) + sum(model.vFlow[p,sc,n,ni,nd,cc] for ni,cc in lin [nd] if (ni,nd,cc) in model.lad)
                    - sum(model.vPfr[ p,sc,n,nd,lout ] for lout  in lout[nd] if (nd,lout) in model.laa ) - sum(model.vPto[ p,sc,n,ni,nd,cc] for ni,cc in lin [nd] if (ni,nd,cc) in model.laa) == 0)
        else:
            return Constraint.Skip
    model.eBalanceP              = Constraint(model.ps, model.st, model.n, model.nd, rule=eBalanceP, doc='active power balance [GW]')

    def eBalanceQ(model,p,sc,st,n,nd):
        if (st,n) in model.s2n and sum(1 for g in model.g if (nd,g) in model.n2g) + sum(1 for lout in lout[nd]) + sum(1 for ni,cc in lin[nd]):
            return (sum(model.vTotalOutputQ[p,sc,n,nr] for nr    in model.nr if (nd,nr  ) in model.n2g )
                    - pDemandQ[p,sc,n,nd]*(1-model.vENS[p,sc,n,nd])
                    + sum(model.vBusShuntQ[p,sc,n,sh ] for sh    in model.sh if (nd,sh  ) in model.n2sh)
                    - sum(model.vQfr[ p,sc,n,nd,lout ] for lout  in lout[nd] if (nd,lout) in model.laa ) - sum(model.vQto[ p,sc,n,ni,nd,cc] for ni,cc in lin [nd] if (ni,nd,cc) in model.laa) == 0)
        else:
            return Constraint.Skip
    model.eBalanceQ              = Constraint(model.ps, model.st, model.n, model.nd, rule=eBalanceQ, doc='reactive power balance [GW]')

    GeneratingBalanceTime = time.time() - StartTime
    StartTime             = time.time()
    print('Generating balance                    ... ', round(GeneratingBalanceTime), 's')

    def eMaxInventory2Comm(model,p,sc,st,n,ec):
        if (st,n) in model.s2n and pIndBinStorInvest[ec] and model.n.ord(n) % pCycleTimeStep[ec] == 0 and pMaxCharge[p,sc,n,ec] + pMaxPower[p,sc,n,ec] and pMaxStorage[p,sc,n,ec]:
            return model.vESSInventory[p,sc,n,ec] <= pMaxStorage[p,sc,n,ec] * model.vCommitment[p,sc,n,ec]
        else:
            return Constraint.Skip
    model.eMaxInventory2Comm     = Constraint(model.ps, model.st, model.n, model.ec, rule=eMaxInventory2Comm, doc='ESS maximum inventory limited by commitment [GWh]')

    def eMinInventory2Comm(model,p,sc,st,n,ec):
        if (st,n) in model.s2n and pIndBinStorInvest[ec] and model.n.ord(n) % pCycleTimeStep[ec] == 0 and pMaxCharge[p,sc,n,ec] + pMaxPower[p,sc,n,ec] and pMaxStorage[p,sc,n,ec]:
            return model.vESSInventory[p,sc,n,ec] >= pMinStorage[p,sc,n,ec] * model.vCommitment[p,sc,n,ec]
        else:
            return Constraint.Skip
    model.eMinInventory2Comm     = Constraint(model.ps, model.st, model.n, model.ec, rule=eMinInventory2Comm, doc='ESS minimum inventory limited by commitment [GWh]')

    def eInflows2Comm(model,p,sc,st,n,ec):
        if (st,n) in model.s2n and pIndBinStorInvest[ec] and model.n.ord(n) % pCycleTimeStep[ec] == 0 and pMaxCharge[p,sc,n,ec] + pMaxPower[p,sc,n,ec] and pEnergyInflows[p,sc,n,ec]:
            return model.vEnergyInflows[p,sc,n,ec] <= pEnergyInflows[p,sc,n,ec] * model.vCommitment[p,sc,n,ec]
        else:
            return Constraint.Skip
    model.eInflows2Comm          = Constraint(model.ps, model.st, model.n, model.ec, rule=eInflows2Comm, doc='ESS inflows limited by commitment [GWh]')

    def eESSInventory(model,p,sc,st,n,es):
        if   (st,n) in model.s2n and model.n.ord(n) == pCycleTimeStep[es]                                              and pMaxCharge[p,sc,n,es] + pMaxPower[p,sc,n,es] and es not in model.ec:
            return pIniInventory[p,sc,n,es]                                        + sum(pDuration[n2]*(pEnergyInflows      [p,sc,n2,es] - model.vEnergyOutflows[p,sc,n2,es] - model.vTotalOutputP[p,sc,n2,es] + pEfficiency[es]*model.vESSTotalCharge[p,sc,n2,es]) for n2 in list(model.n2)[model.n.ord(n)-pCycleTimeStep[es]:model.n.ord(n)]) == model.vESSInventory[p,sc,n,es] + model.vESSSpillage[p,sc,n,es]
        elif (st,n) in model.s2n and model.n.ord(n) >  pCycleTimeStep[es] and model.n.ord(n) % pCycleTimeStep[es] == 0 and pMaxCharge[p,sc,n,es] + pMaxPower[p,sc,n,es] and es not in model.ec:
            return model.vESSInventory[p,sc,model.n.prev(n,pCycleTimeStep[es]),es] + sum(pDuration[n2]*(pEnergyInflows      [p,sc,n2,es] - model.vEnergyOutflows[p,sc,n2,es] - model.vTotalOutputP[p,sc,n2,es] + pEfficiency[es]*model.vESSTotalCharge[p,sc,n2,es]) for n2 in list(model.n2)[model.n.ord(n)-pCycleTimeStep[es]:model.n.ord(n)]) == model.vESSInventory[p,sc,n,es] + model.vESSSpillage[p,sc,n,es]
        elif (st,n) in model.s2n and model.n.ord(n) == pCycleTimeStep[es]                                              and pMaxCharge[p,sc,n,es] + pMaxPower[p,sc,n,es] and es     in model.ec:
            return pIniInventory[p,sc,n,es]                                        + sum(pDuration[n2]*(model.vEnergyInflows[p,sc,n2,es] - model.vEnergyOutflows[p,sc,n2,es] - model.vTotalOutputP[p,sc,n2,es] + pEfficiency[es]*model.vESSTotalCharge[p,sc,n2,es]) for n2 in list(model.n2)[model.n.ord(n)-pCycleTimeStep[es]:model.n.ord(n)]) == model.vESSInventory[p,sc,n,es] + model.vESSSpillage[p,sc,n,es]
        elif (st,n) in model.s2n and model.n.ord(n) >  pCycleTimeStep[es] and model.n.ord(n) % pCycleTimeStep[es] == 0 and pMaxCharge[p,sc,n,es] + pMaxPower[p,sc,n,es] and es     in model.ec:
            return model.vESSInventory[p,sc,model.n.prev(n,pCycleTimeStep[es]),es] + sum(pDuration[n2]*(model.vEnergyInflows[p,sc,n2,es] - model.vEnergyOutflows[p,sc,n2,es] - model.vTotalOutputP[p,sc,n2,es] + pEfficiency[es]*model.vESSTotalCharge[p,sc,n2,es]) for n2 in list(model.n2)[model.n.ord(n)-pCycleTimeStep[es]:model.n.ord(n)]) == model.vESSInventory[p,sc,n,es] + model.vESSSpillage[p,sc,n,es]
        else:
            return Constraint.Skip
    model.eESSInventory          = Constraint(model.ps, model.st, model.n, model.es, rule=eESSInventory, doc='ESS inventory balance [GWh]')

    def eMaxCharge(model,p,sc,st,n,es):
        if (st,n) in model.s2n and pMaxCharge[p,sc,n,es] and pMaxCharge2ndBlock[p,sc,n,es]:
            return model.vCharge2ndBlock[p,sc,n,es] / pMaxCharge2ndBlock[p,sc,n,es] <= 1.0
        else:
            return Constraint.Skip
    model.eMaxCharge             = Constraint(model.ps, model.st, model.n, model.es, rule=eMaxCharge, doc='max charge of an ESS [p.u.]')

    def eMinCharge(model,p,sc,st,n,es):
        if (st,n) in model.s2n and pMaxCharge[p,sc,n,es] and pMaxCharge2ndBlock[p,sc,n,es]:
            return model.vCharge2ndBlock[p,sc,n,es] / pMaxCharge2ndBlock[p,sc,n,es] >= 0.0
        else:
            return Constraint.Skip
    model.eMinCharge             = Constraint(model.ps, model.st, model.n, model.es, rule=eMinCharge, doc='min charge of an ESS [p.u.]')

    def eChargeDischarge(model,p,sc,st,n,es):
        if (st,n) in model.s2n and pMaxPower2ndBlock[p,sc,n,es] and pMaxCharge2ndBlock[p,sc,n,es]:
            return ((model.vOutput2ndBlock[p,sc,n,es]) / pMaxPower2ndBlock [p,sc,n,es] +
                    (model.vCharge2ndBlock[p,sc,n,es]) / pMaxCharge2ndBlock[p,sc,n,es] <= 1.0)
        else:
            return Constraint.Skip
    model.eChargeDischarge       = Constraint(model.ps, model.st, model.n, model.es, rule=eChargeDischarge, doc='incompatibility between charge and discharge [p.u.]')

    def eESSTotalCharge(model,p,sc,st,n,es):
        if   (st,n) in model.s2n and pMaxCharge[p,sc,n,es] and pMaxCharge2ndBlock[p,sc,n,es] and pMinCharge[p,sc,n,es] == 0.0:
            return model.vESSTotalCharge[p,sc,n,es]                         ==      model.vCharge2ndBlock[p,sc,n,es]
        elif (st,n) in model.s2n and pMaxCharge[p,sc,n,es] and pMaxCharge2ndBlock[p,sc,n,es]:
            return model.vESSTotalCharge[p,sc,n,es] / pMinCharge[p,sc,n,es] == 1 + (model.vCharge2ndBlock[p,sc,n,es]) / pMinCharge[p,sc,n,es]
        else:
            return Constraint.Skip
    model.eESSTotalCharge        = Constraint(model.ps, model.st, model.n, model.es, rule=eESSTotalCharge, doc='total charge of an ESS unit [GW]')

    def eEnergyOutflows(model,p,sc,st,n,es):
        if (st,n) in model.s2n and model.n.ord(n) % pOutflowsTimeStep[es] == 0 and sum(pEnergyOutflows[p,sc,n2,es] for n2 in model.n2):
            return sum(model.vEnergyOutflows[p,sc,n2,es]*pLoadLevelDuration[n2] for n2 in list(model.n2)[model.n.ord(n) - pOutflowsTimeStep[es]:model.n.ord(n)]) == sum(pEnergyOutflows[p,sc,n2,es]*pLoadLevelDuration[n2] for n2 in list(model.n2)[model.n.ord(n) - pOutflowsTimeStep[es]:model.n.ord(n)])
        else:
            return Constraint.Skip
    model.eEnergyOutflows        = Constraint(model.ps, model.st, model.n, model.es, rule=eEnergyOutflows, doc='energy outflows of an ESS unit [GW]')

    GeneratingESSTime = time.time() - StartTime
    StartTime      = time.time()
    print('Generating storage operation          ... ', round(GeneratingESSTime), 's')

    def eMaxOutput2ndBlock(model,p,sc,st,n,nr):
        if (st,n) in model.s2n and pMaxPower2ndBlock[p,sc,n,nr]:
            return model.vOutput2ndBlock[p,sc,n,nr] / pMaxPower2ndBlock[p,sc,n,nr] <= model.vCommitment[p,sc,n,nr]
        else:
            return Constraint.Skip
    model.eMaxOutput2ndBlock     = Constraint(model.ps, model.st, model.n, model.nr, rule=eMaxOutput2ndBlock, doc='max output of the second block of a committed unit [p.u.]')

    def eMinOutput2ndBlock(model,p,sc,st,n,nr):
        if (st,n) in model.s2n and pMaxPower2ndBlock[p,sc,n,nr]:
            return model.vOutput2ndBlock[p,sc,n,nr] / pMaxPower2ndBlock[p,sc,n,nr] >= 0.0
        else:
            return Constraint.Skip
    model.eMinOutput2ndBlock     = Constraint(model.ps, model.st, model.n, model.nr, rule=eMinOutput2ndBlock, doc='min output of the second block of a committed unit [p.u.]')

    def eTotalOutputP(model,p,sc,st,n,nr):
        if (st,n) in model.s2n and pMaxPower[p,sc,n,nr] and pMinPower[p,sc,n,nr] == 0.0:
            return model.vTotalOutputP[p,sc,n,nr]                        ==                                model.vOutput2ndBlock[p,sc,n,nr]
        elif (st,n) in model.s2n and pMaxPower[p,sc,n,nr]:
            return model.vTotalOutputP[p,sc,n,nr] / pMinPower[p,sc,n,nr] == model.vCommitment[p,sc,n,nr] + model.vOutput2ndBlock[p,sc,n,nr] / pMinPower[p,sc,n,nr]
        else:
            return Constraint.Skip
    model.eTotalOutputP          = Constraint(model.ps, model.st, model.n, model.nr, rule=eTotalOutputP, doc='total output of a unit [GW]')

    def eTotalOutputQ_LB(model,p,sc,st,n,nr):
        if (st,n) in model.s2n and (pRatedMaxPowerQ[nr] or pRatedMinPowerQ[nr]):
            return model.vTotalOutputQ[p,sc,n,nr] >= -model.vTotalOutputP[p,sc,n,nr]*math.tan(math.acos(0.99))
        else:
            return Constraint.Skip
    model.eTotalOutputQ_LB       = Constraint(model.ps, model.st, model.n, model.nr, rule=eTotalOutputQ_LB, doc='lower bound of the reactive power output of a unit [GVAr]')

    def eTotalOutputQ_UB(model,p,sc,st,n,nr):
        if (st,n) in model.s2n and (pRatedMaxPowerQ[nr] or pRatedMinPowerQ[nr]):
            return model.vTotalOutputQ[p,sc,n,nr] <=  model.vTotalOutputP[p,sc,n,nr]*math.tan(math.acos(0.95))
        else:
            return Constraint.Skip
    model.eTotalOutputQ_UB       = Constraint(model.ps, model.st, model.n, model.nr, rule=eTotalOutputQ_UB, doc='upper bound of the reactive power output of a unit [GVAr]')

    def eUCStrShut(model,p,sc,st,n,nr):
        if   (st,n) in model.s2n and pMustRun[nr] == 0 and (pMinPower[p,sc,n,nr] or pConstantVarCost[nr]) and nr not in model.es and n == model.n.first():
            return model.vCommitment[p,sc,n,nr] - pInitialUC[nr]                             == model.vStartUp[p,sc,n,nr] - model.vShutDown[p,sc,n,nr]
        elif (st,n) in model.s2n and pMustRun[nr] == 0 and (pMinPower[p,sc,n,nr] or pConstantVarCost[nr]) and nr not in model.es:
            return model.vCommitment[p,sc,n,nr] - model.vCommitment[p,sc,model.n.prev(n),nr] == model.vStartUp[p,sc,n,nr] - model.vShutDown[p,sc,n,nr]
        else:
            return Constraint.Skip
    model.eUCStrShut             = Constraint(model.ps, model.st, model.n, model.nr, rule=eUCStrShut, doc='relation among commitment startup and shutdown')

    GeneratingUCTime = time.time() - StartTime
    StartTime        = time.time()
    print('Generating generation commitment      ... ', round(GeneratingUCTime), 's')

    def eGenCapacity1(model,p,sc,st,n,g):
        if (st,n) in model.s2n and g in model.gc:
            return model.vTotalOutputP[p,sc,n,g] >= pMinPower[p,sc,n,g] * model.vGenerationInvest[p,gc]
        elif (st,n) in model.s2n and g not in model.gc:
            return model.vTotalOutputP[p,sc,n,g] >= pMinPower[p,sc,n,g]
        else:
            return Constraint.Skip
    model.eGenCapacity1          = Constraint(model.ps, model.st, model.n, model.g, rule=eGenCapacity1, doc='minimum power output by a generation unit [p.u.]')

    def eGenCapacity2(model,p,sc,st,n,g):
        if (st,n) in model.s2n and g in model.gc:
            return model.vTotalOutputP[p,sc,n,g] <= pMaxPower[p,sc,n,g] * model.vGenerationInvest[p,gc]
        elif (st,n) in model.s2n and g not in model.gc:
            return model.vTotalOutputP[p,sc,n,g] <= pMaxPower[p,sc,n,g]
        else:
            return Constraint.Skip
    model.eGenCapacity2          = Constraint(model.ps, model.st, model.n, model.g, rule=eGenCapacity2, doc='maximum power output by a generation unit [p.u.]')

    def eNetCapacityPfrLowerBound(model,p,sc,st,n,ni,nf,cc):
        if (st,n) in model.s2n and pIndBinSingleNode == 0 and (ni,nf,cc) in model.lc:
            return model.vPfr[p,sc,n,ni,nf,cc] / (pLineNTCFrw[ni,nf,cc]*1e3)  >= - model.vNetworkInvest[p,ni,nf,cc]
        else:
            return Constraint.Skip
    model.eNetCapacityPfrLowerBound = Constraint(model.ps, model.st, model.n, model.la, rule=eNetCapacityPfrLowerBound, doc='maximum flow by candidate network capacity [p.u.]')

    def eNetCapacityPfrUpperBound(model,p,sc,st,n,ni,nf,cc):
        if (st,n) in model.s2n and pIndBinSingleNode == 0 and (ni,nf,cc) in model.lc:
            return model.vPfr[p,sc,n,ni,nf,cc] / (pLineNTCFrw[ni,nf,cc]*1e3) <=   model.vNetworkInvest[p,ni,nf,cc]
        else:
            return Constraint.Skip
    model.eNetCapacityPfrUpperBound = Constraint(model.ps, model.st, model.n, model.la, rule=eNetCapacityPfrUpperBound, doc='maximum flow by candidate network capacity [p.u.]')

    def eNetCapacityPtoLowerBound(model,p,sc,st,n,ni,nf,cc):
        if (st,n) in model.s2n and pIndBinSingleNode == 0 and (ni,nf,cc) in model.lc:
            return model.vPto[p,sc,n,ni,nf,cc] / (pLineNTCFrw[ni,nf,cc]*1e3) >= - model.vNetworkInvest[p,ni,nf,cc]
        else:
            return Constraint.Skip
    model.eNetCapacityPtoLowerBound = Constraint(model.ps, model.st, model.n, model.la, rule=eNetCapacityPtoLowerBound, doc='maximum flow by candidate network capacity [p.u.]')

    def eNetCapacityPtoUpperBound(model,p,sc,st,n,ni,nf,cc):
        if (st,n) in model.s2n and pIndBinSingleNode == 0 and (ni,nf,cc) in model.lc:
            return model.vPto[p,sc,n,ni,nf,cc] / (pLineNTCFrw[ni,nf,cc]*1e3) <=   model.vNetworkInvest[p,ni,nf,cc]
        else:
            return Constraint.Skip
    model.eNetCapacityPtoUpperBound = Constraint(model.ps, model.st, model.n, model.la, rule=eNetCapacityPtoUpperBound, doc='maximum flow by candidate network capacity [p.u.]')

    def eNetCapacityQfrLowerBound(model,p,sc,st,n,ni,nf,cc):
        if (st,n) in model.s2n and pIndBinSingleNode == 0 and (ni,nf,cc) in model.lc:
            return model.vQfr[p,sc,n,ni,nf,cc] / (pLineNTCFrw[ni,nf,cc]*1e3) >= - model.vNetworkInvest[p,ni,nf,cc]
        else:
            return Constraint.Skip
    model.eNetCapacityQfrLowerBound = Constraint(model.ps, model.st, model.n, model.la, rule=eNetCapacityQfrLowerBound, doc='maximum flow by candidate network capacity [p.u.]')

    def eNetCapacityQfrUpperBound(model,p,sc,st,n,ni,nf,cc):
        if (st,n) in model.s2n and pIndBinSingleNode == 0 and (ni,nf,cc) in model.lc:
            return model.vQfr[p,sc,n,ni,nf,cc] / (pLineNTCFrw[ni,nf,cc]*1e3) <=   model.vNetworkInvest[p,ni,nf,cc]
        else:
            return Constraint.Skip
    model.eNetCapacityQfrUpperBound = Constraint(model.ps, model.st, model.n, model.la, rule=eNetCapacityQfrUpperBound, doc='maximum flow by candidate network capacity [p.u.]')

    def eNetCapacityQtoLowerBound(model,p,sc,st,n,ni,nf,cc):
        if (st,n) in model.s2n and pIndBinSingleNode == 0 and (ni,nf,cc) in model.lc:
            return model.vQto[p,sc,n,ni,nf,cc] / (pLineNTCFrw[ni,nf,cc]*1e3) >= - model.vNetworkInvest[p,ni,nf,cc]
        else:
            return Constraint.Skip
    model.eNetCapacityQtoLowerBound = Constraint(model.ps, model.st, model.n, model.la, rule=eNetCapacityQtoLowerBound, doc='maximum flow by candidate network capacity [p.u.]')

    def eNetCapacityQtoUpperBound(model,p,sc,st,n,ni,nf,cc):
        if (st,n) in model.s2n and pIndBinSingleNode == 0 and (ni,nf,cc) in model.lc:
            return model.vQto[p,sc,n,ni,nf,cc] / (pLineNTCFrw[ni,nf,cc]*1e3) <=   model.vNetworkInvest[p,ni,nf,cc]
        else:
            return Constraint.Skip
    model.eNetCapacityQtoUpperBound = Constraint(model.ps, model.st, model.n, model.la, rule=eNetCapacityQtoUpperBound, doc='maximum flow by candidate network capacity [p.u.]')

    def ePfrLowerBound(model,p,sc,st,n,ni,nf,cc):
        if (st,n) in model.s2n and pIndBinSingleNode == 0 and (ni,nf,cc) in model.lc:
            return model.vPfr[p,sc,n,ni,nf,cc] - pLineG[ni,nf,cc]                     *pLineTAP[ni,nf,cc]**2*model.vW[p,sc,n,ni] + pLineTAP[ni,nf,cc]*pLineG[ni,nf,cc]*model.vC[p,sc,n,ni,nf,cc] - pLineTAP[ni,nf,cc]*pLineB[ni,nf,cc]*model.vS[p,sc,n,ni,nf,cc] >= (1 - model.vNetworkInvest[p,ni,nf,cc])
        elif (st,n) in model.s2n and pIndBinSingleNode == 0 and (ni,nf,cc) not in model.lc:
            return model.vPfr[p,sc,n,ni,nf,cc] - pLineG[ni,nf,cc]                     *pLineTAP[ni,nf,cc]**2*model.vW[p,sc,n,ni] + pLineTAP[ni,nf,cc]*pLineG[ni,nf,cc]*model.vC[p,sc,n,ni,nf,cc] - pLineTAP[ni,nf,cc]*pLineB[ni,nf,cc]*model.vS[p,sc,n,ni,nf,cc] >= 0
        else:
            return Constraint.Skip
    model.ePfrLowerBound          = Constraint(model.ps, model.st, model.n, model.la, rule=ePfrLowerBound, doc='maximum flow by existing network capacity [p.u.]')

    def ePfrUpperBound(model,p,sc,st,n,ni,nf,cc):
        if (st,n) in model.s2n and pIndBinSingleNode == 0 and (ni,nf,cc) in model.lc:
            return model.vPfr[p,sc,n,ni,nf,cc] - pLineG[ni,nf,cc]                     *pLineTAP[ni,nf,cc]**2*model.vW[p,sc,n,ni] + pLineTAP[ni,nf,cc]*pLineG[ni,nf,cc]*model.vC[p,sc,n,ni,nf,cc] - pLineTAP[ni,nf,cc]*pLineB[ni,nf,cc]*model.vS[p,sc,n,ni,nf,cc] <= (1 - model.vNetworkInvest[p,ni,nf,cc])
        elif (st,n) in model.s2n and pIndBinSingleNode == 0 and (ni,nf,cc) not in model.lc:
            return model.vPfr[p,sc,n,ni,nf,cc] - pLineG[ni,nf,cc]                     *pLineTAP[ni,nf,cc]**2*model.vW[p,sc,n,ni] + pLineTAP[ni,nf,cc]*pLineG[ni,nf,cc]*model.vC[p,sc,n,ni,nf,cc] - pLineTAP[ni,nf,cc]*pLineB[ni,nf,cc]*model.vS[p,sc,n,ni,nf,cc] <= 0
        else:
            return Constraint.Skip
    model.ePfrUpperBound          = Constraint(model.ps, model.st, model.n, model.la, rule=ePfrUpperBound, doc='maximum flow by existing network capacity [p.u.]')

    def ePtoLowerBound(model,p,sc,st,n,ni,nf,cc):
        if (st,n) in model.s2n and pIndBinSingleNode == 0 and (ni,nf,cc) in model.lc:
            return model.vPto[p,sc,n,ni,nf,cc] - pLineG[ni,nf,cc]                                          *model.vW[p,sc,n,nf] + pLineTAP[ni,nf,cc]*pLineG[ni,nf,cc]*model.vC[p,sc,n,ni,nf,cc] + pLineTAP[ni,nf,cc]*pLineB[ni,nf,cc]*model.vS[p,sc,n,ni,nf,cc] >= (1 - model.vNetworkInvest[p,ni,nf,cc])
        elif (st,n) in model.s2n and pIndBinSingleNode == 0 and (ni,nf,cc) not in model.lc:
            return model.vPto[p,sc,n,ni,nf,cc] - pLineG[ni,nf,cc]                                          *model.vW[p,sc,n,nf] + pLineTAP[ni,nf,cc]*pLineG[ni,nf,cc]*model.vC[p,sc,n,ni,nf,cc] + pLineTAP[ni,nf,cc]*pLineB[ni,nf,cc]*model.vS[p,sc,n,ni,nf,cc] >= 0
        else:
            return Constraint.Skip
    model.ePtoLowerBound          = Constraint(model.ps, model.st, model.n, model.la, rule=ePtoLowerBound, doc='maximum flow by existing network capacity [p.u.]')

    def ePtoUpperBound(model,p,sc,st,n,ni,nf,cc):
        if (st,n) in model.s2n and pIndBinSingleNode == 0 and (ni,nf,cc) in model.lc:
            return model.vPto[p,sc,n,ni,nf,cc] - pLineG[ni,nf,cc]                                          *model.vW[p,sc,n,nf] + pLineTAP[ni,nf,cc]*pLineG[ni,nf,cc]*model.vC[p,sc,n,ni,nf,cc] + pLineTAP[ni,nf,cc]*pLineB[ni,nf,cc]*model.vS[p,sc,n,ni,nf,cc] <= (1 - model.vNetworkInvest[p,ni,nf,cc])
        elif (st,n) in model.s2n and pIndBinSingleNode == 0 and (ni,nf,cc) not in model.lc:
            return model.vPto[p,sc,n,ni,nf,cc] - pLineG[ni,nf,cc]                                          *model.vW[p,sc,n,nf] + pLineTAP[ni,nf,cc]*pLineG[ni,nf,cc]*model.vC[p,sc,n,ni,nf,cc] + pLineTAP[ni,nf,cc]*pLineB[ni,nf,cc]*model.vS[p,sc,n,ni,nf,cc] <= 0
        else:
            return Constraint.Skip
    model.ePtoUpperBound          = Constraint(model.ps, model.st, model.n, model.la, rule=ePtoUpperBound, doc='maximum flow by existing network capacity [p.u.]')

    def eQfrLowerBound(model,p,sc,st,n,ni,nf,cc):
        if (st,n) in model.s2n and pIndBinSingleNode == 0 and (ni,nf,cc) in model.lc:
            return model.vQfr[p,sc,n,ni,nf,cc] + (pLineB[ni,nf,cc]+pLineBsh[ni,nf,cc])*pLineTAP[ni,nf,cc]**2*model.vW[p,sc,n,ni] - pLineTAP[ni,nf,cc]*pLineG[ni,nf,cc]*model.vS[p,sc,n,ni,nf,cc] - pLineTAP[ni,nf,cc]*pLineB[ni,nf,cc]*model.vC[p,sc,n,ni,nf,cc] >= (1 - model.vNetworkInvest[p,ni,nf,cc])
        elif (st,n) in model.s2n and pIndBinSingleNode == 0 and (ni,nf,cc) not in model.lc:
            return model.vQfr[p,sc,n,ni,nf,cc] + (pLineB[ni,nf,cc]+pLineBsh[ni,nf,cc])*pLineTAP[ni,nf,cc]**2*model.vW[p,sc,n,ni] - pLineTAP[ni,nf,cc]*pLineG[ni,nf,cc]*model.vS[p,sc,n,ni,nf,cc] - pLineTAP[ni,nf,cc]*pLineB[ni,nf,cc]*model.vC[p,sc,n,ni,nf,cc] >= 0
        else:
            return Constraint.Skip
    model.eQfrLowerBound          = Constraint(model.ps, model.st, model.n, model.la, rule=eQfrLowerBound, doc='maximum flow by existing network capacity [p.u.]')

    def eQfrUpperBound(model,p,sc,st,n,ni,nf,cc):
        if (st,n) in model.s2n and pIndBinSingleNode == 0 and (ni,nf,cc) in model.lc:
            return model.vQfr[p,sc,n,ni,nf,cc] + (pLineB[ni,nf,cc]+pLineBsh[ni,nf,cc])*pLineTAP[ni,nf,cc]**2*model.vW[p,sc,n,ni] - pLineTAP[ni,nf,cc]*pLineG[ni,nf,cc]*model.vS[p,sc,n,ni,nf,cc] - pLineTAP[ni,nf,cc]*pLineB[ni,nf,cc]*model.vC[p,sc,n,ni,nf,cc] <= (1 - model.vNetworkInvest[p,ni,nf,cc])
        elif (st,n) in model.s2n and pIndBinSingleNode == 0 and (ni,nf,cc) not in model.lc:
            return model.vQfr[p,sc,n,ni,nf,cc] + (pLineB[ni,nf,cc]+pLineBsh[ni,nf,cc])*pLineTAP[ni,nf,cc]**2*model.vW[p,sc,n,ni] - pLineTAP[ni,nf,cc]*pLineG[ni,nf,cc]*model.vS[p,sc,n,ni,nf,cc] - pLineTAP[ni,nf,cc]*pLineB[ni,nf,cc]*model.vC[p,sc,n,ni,nf,cc] <= 0
        else:
            return Constraint.Skip
    model.eQfrUpperBound          = Constraint(model.ps, model.st, model.n, model.la, rule=eQfrUpperBound, doc='maximum flow by existing network capacity [p.u.]')

    # def eq1(model,p,sc,st,n,ni,nf,cc):
    #     if (st,n) in model.s2n:
    #         return model.vQfr[p,sc,n,ni,nf,cc] + (pLineB[ni,nf,cc]+pLineBsh[ni,nf,cc])*pLineTAP[ni,nf,cc]**2*model.vW[p,sc,n,ni] - pLineTAP[ni,nf,cc]*pLineG[ni,nf,cc]*model.vS[p,sc,n,ni,nf,cc] - pLineTAP[ni,nf,cc]*pLineB[ni,nf,cc]*model.vC[p,sc,n,ni,nf,cc] == 0
    #     else:
    #         return Constraint.Skip
    # model.eq1 = Constraint(model.ps, model.st, model.n, model.la, rule=eq1, doc='')
    #
    # def eq2(model,p,sc,st,n,ni,nf,cc):
    #     if (st,n) in model.s2n:
    #         return model.vQto[p,sc,n,ni,nf,cc] + (pLineB[ni,nf,cc]+pLineBsh[ni,nf,cc])                      *model.vW[p,sc,n,nf] + pLineTAP[ni,nf,cc]*pLineG[ni,nf,cc]*model.vS[p,sc,n,ni,nf,cc] - pLineTAP[ni,nf,cc]*pLineB[ni,nf,cc]*model.vC[p,sc,n,ni,nf,cc] == 0
    #     else:
    #         return Constraint.Skip
    # model.eq2 = Constraint(model.ps, model.st, model.n, model.la, rule=eq2, doc='')

    def eQtoLowerBound(model,p,sc,st,n,ni,nf,cc):
        if (st,n) in model.s2n and pIndBinSingleNode == 0 and (ni,nf,cc) in model.lc:
            return model.vQto[p,sc,n,ni,nf,cc] + (pLineB[ni,nf,cc]+pLineBsh[ni,nf,cc])                     *model.vW[p,sc,n,nf] + pLineTAP[ni,nf,cc]*pLineG[ni,nf,cc]*model.vS[p,sc,n,ni,nf,cc] - pLineTAP[ni,nf,cc]*pLineB[ni,nf,cc]*model.vC[p,sc,n,ni,nf,cc] >= (1 - model.vNetworkInvest[p,ni,nf,cc])
        elif (st,n) in model.s2n and pIndBinSingleNode == 0 and (ni,nf,cc) not in model.lc:
            return model.vQto[p,sc,n,ni,nf,cc] + (pLineB[ni,nf,cc]+pLineBsh[ni,nf,cc])                     *model.vW[p,sc,n,nf] + pLineTAP[ni,nf,cc]*pLineG[ni,nf,cc]*model.vS[p,sc,n,ni,nf,cc] - pLineTAP[ni,nf,cc]*pLineB[ni,nf,cc]*model.vC[p,sc,n,ni,nf,cc] >= 0
        else:
            return Constraint.Skip
    model.eQtoLowerBound          = Constraint(model.ps, model.st, model.n, model.la, rule=eQtoLowerBound, doc='maximum flow by existing network capacity [p.u.]')

    def eQtoUpperBound(model,p,sc,st,n,ni,nf,cc):
        if (st,n) in model.s2n and pIndBinSingleNode == 0 and (ni,nf,cc) in model.lc:
            return model.vQto[p,sc,n,ni,nf,cc] + (pLineB[ni,nf,cc]+pLineBsh[ni,nf,cc])                     *model.vW[p,sc,n,nf] + pLineTAP[ni,nf,cc]*pLineG[ni,nf,cc]*model.vS[p,sc,n,ni,nf,cc] - pLineTAP[ni,nf,cc]*pLineB[ni,nf,cc]*model.vC[p,sc,n,ni,nf,cc] <= (1 - model.vNetworkInvest[p,ni,nf,cc])
        elif (st,n) in model.s2n and pIndBinSingleNode == 0 and (ni,nf,cc) not in model.lc:
            return model.vQto[p,sc,n,ni,nf,cc] + (pLineB[ni,nf,cc]+pLineBsh[ni,nf,cc])                     *model.vW[p,sc,n,nf] + pLineTAP[ni,nf,cc]*pLineG[ni,nf,cc]*model.vS[p,sc,n,ni,nf,cc] - pLineTAP[ni,nf,cc]*pLineB[ni,nf,cc]*model.vC[p,sc,n,ni,nf,cc] <= 0
        else:
            return Constraint.Skip
    model.eQtoUpperBound          = Constraint(model.ps, model.st, model.n, model.la, rule=eQtoUpperBound, doc='maximum flow by existing network capacity [p.u.]')

    def eWLowerBound(model,p,sc,st,n,nd):
        if (st,n) in model.s2n:
            return model.vW[p,sc,n,nd] >=   0.95**2
        else:
            return Constraint.Skip
    model.eWLowerBound            = Constraint(model.ps, model.st, model.n, model.nd, rule=eWLowerBound,   doc='lower bound of the squared voltage magnitude [p.u.]')

    def eWUpperBound(model,p,sc,st,n,nd):
        if (st,n) in model.s2n:
            return model.vW[p,sc,n,nd] <=   1.05**2
        else:
            return Constraint.Skip
    model.eWUpperBound            = Constraint(model.ps, model.st, model.n, model.nd, rule=eWUpperBound,   doc='upper bound of the squared voltage magnitude [p.u.]')

    # def eLineCapacityFrUpperBound(model,p,sc,st,n,ni,nf,cc):
    #     if (st,n) in model.s2n:
    #         return model.vPfr[p,sc,n,ni,nf,cc]**2 + model.vQfr[p,sc,n,ni,nf,cc]**2 <= (pLineNTCFrw[ni,nf,cc]*1)**2
    #     else:
    #         return Constraint.Skip
    # model.eLineCapacityFrUpperBound = Constraint(model.ps, model.st, model.n, model.la, rule=eLineCapacityFrUpperBound, doc='maximum flow by a network capacity [p.u.]')
    #
    # def eLineCapacityToUpperBound(model,p,sc,st,n,ni,nf,cc):
    #     if (st,n) in model.s2n:
    #         return model.vPto[p,sc,n,ni,nf,cc]**2 + model.vQto[p,sc,n,ni,nf,cc]**2 <= (pLineNTCFrw[ni,nf,cc]*1)**2
    #     else:
    #         return Constraint.Skip
    # model.eLineCapacityToUpperBound = Constraint(model.ps, model.st, model.n, model.la, rule=eLineCapacityToUpperBound, doc='maximum flow by a network capacity [p.u.]')

    def eLineCapacityFr_LP(model,p,sc,st,n,ni,nf,cc):
        if (st, n) in model.s2n:
            return sum(model.pLineM[ni,nf,cc,l] * model.vDelta_Pfr[p,sc,n,ni,nf,cc,l] for l in model.L) + sum(model.pLineM[ni,nf,cc,l] * model.vDelta_Qfr[p,sc,n,ni,nf,cc,l] for l in model.L) <= (pLineNTCFrw[ni,nf,cc]*1)**2
    model.eLineCapacityFr_LP = Constraint(model.ps, model.st, model.n, model.laa, rule=eLineCapacityFr_LP)

    def eLineCapacityFr_LinearPfr1(model,p,sc,st,n,ni,nf,cc):
        if (st, n) in model.s2n:
            return model.vPfr_max[p,sc,n,ni,nf,cc] - model.vPfr_min[p,sc,n,ni,nf,cc] == model.vPfr[p,sc,n,ni,nf,cc]
    model.eLineCapacityFr_LinearPfr1 = Constraint(model.ps, model.st, model.n, model.laa, rule=eLineCapacityFr_LinearPfr1)

    def eLineCapacityFr_LinearPfr2(model,p,sc,st,n,ni,nf,cc):
        if (st, n) in model.s2n:
            return model.vPfr_max[p,sc,n,ni,nf,cc] + model.vPfr_min[p,sc,n,ni,nf,cc] == sum(model.vDelta_Pfr[p,sc,n,ni,nf,cc,l] for l in model.L)
    model.eLineCapacityFr_LinearPfr2 = Constraint(model.ps, model.st, model.n, model.laa, rule=eLineCapacityFr_LinearPfr2)

    def eLineCapacityFr_LinearQfr1(model,p,sc,st,n,ni,nf,cc):
        if (st, n) in model.s2n:
            return (model.vQfr_max[p,sc,n,ni,nf,cc] - model.vQfr_min[p,sc,n,ni,nf,cc] == model.vQfr[p,sc,n,ni,nf,cc])
    model.eLineCapacityFr_LinearQfr1 = Constraint(model.ps, model.st, model.n, model.laa, rule=eLineCapacityFr_LinearQfr1)

    def eLineCapacityFr_LinearQfr2(model,p,sc,st,n,ni,nf,cc):
        if (st, n) in model.s2n:
            return (model.vQfr_max[p,sc,n,ni,nf,cc] + model.vQfr_min[p,sc,n,ni,nf,cc] == sum(model.vDelta_Qfr[p,sc,n,ni,nf,cc,l] for l in model.L))
    model.eLineCapacityFr_LinearQfr2 = Constraint(model.ps, model.st, model.n, model.laa, rule=eLineCapacityFr_LinearQfr2)

    def eLineCapacityTo_LP(model,p,sc,st,n,ni,nf,cc):
        if (st, n) in model.s2n:
            return sum(model.pLineM[ni,nf,cc,l] * model.vDelta_Pto[p,sc,n,ni,nf,cc,l] for l in model.L) + sum(model.pLineM[ni,nf,cc,l] * model.vDelta_Qto[p,sc,n,ni,nf,cc,l] for l in model.L) <= (pLineNTCFrw[ni,nf,cc]*1)**2
    model.eLineCapacityTo_LP = Constraint(model.ps, model.st, model.n, model.laa, rule=eLineCapacityTo_LP)

    def eLineCapacityTo_LinearPTo1(model,p,sc,st,n,ni,nf,cc):
        if (st, n) in model.s2n:
            return model.vPto_max[p,sc,n,ni,nf,cc] - model.vPto_min[p,sc,n,ni,nf,cc] == model.vPto[p,sc,n,ni,nf,cc]
    model.eLineCapacityTo_LinearPTo1 = Constraint(model.ps, model.st, model.n, model.laa, rule=eLineCapacityTo_LinearPTo1)

    def eLineCapacityTo_LinearPTo2(model,p,sc,st,n,ni,nf,cc):
        if (st, n) in model.s2n:
            return model.vPto_max[p,sc,n,ni,nf,cc] + model.vPto_min[p,sc,n,ni,nf,cc] == sum(model.vDelta_Pto[p,sc,n,ni,nf,cc,l] for l in model.L)
    model.eLineCapacityTo_LinearPTo2 = Constraint(model.ps, model.st, model.n, model.laa, rule=eLineCapacityTo_LinearPTo2)

    def eLineCapacityTo_LinearQTo1(model,p,sc,st,n,ni,nf,cc):
        if (st, n) in model.s2n:
            return (model.vQto_max[p,sc,n,ni,nf,cc] - model.vQto_min[p,sc,n,ni,nf,cc] == model.vQto[p,sc,n,ni,nf,cc])
    model.eLineCapacityTo_LinearQTo1 = Constraint(model.ps, model.st, model.n, model.laa, rule=eLineCapacityTo_LinearQTo1)

    def eLineCapacityTo_LinearQTo2(model,p,sc,st,n,ni,nf,cc):
        if (st, n) in model.s2n:
            return (model.vQto_max[p,sc,n,ni,nf,cc] + model.vQto_min[p,sc,n,ni,nf,cc] == sum(model.vDelta_Qto[p,sc,n,ni,nf,cc,l] for l in model.L))
    model.eLineCapacityTo_LinearQTo2 = Constraint(model.ps, model.st, model.n, model.laa, rule=eLineCapacityTo_LinearQTo2)

    def eLineActivePowerLossesLowerBound(model,p,sc,st,n,ni,nf,cc):
        if (st,n) in model.s2n:
            return model.vPfr[p,sc,n,ni,nf,cc] + model.vPto[p,sc,n,ni,nf,cc] >= 0
        else:
            return Constraint.Skip
    model.eLineActivePowerLossesLowerBound = Constraint(model.ps, model.st, model.n, model.la, rule=eLineActivePowerLossesLowerBound, doc='lower bound of the active power losses [p.u.]')

    # def eLineConic1(model,p,sc,st,n,ni,nf,cc):
    #     if (st,n) in model.s2n:
    #         # return model.vC[p,sc,n,ni,nf,cc]**2 + model.vS[p,sc,n,ni,nf,cc]**2 <= model.vW[p,sc,n,ni]*model.vW[p,sc,n,nf]
    #         return model.vC[p,sc,n,ni,nf,cc]**2 + model.vS[p,sc,n,ni,nf,cc]**2 <= model.vW[p,sc,n,nf]
    #     else:
    #         return Constraint.Skip
    # model.eLineConic1 = Constraint(model.ps, model.st, model.n, model.la, rule=eLineConic1, doc='conic constraint 1 [p.u.]')

    def eLineConic1_LP(model,p,sc,st,n,ni,nf,cc):
        if (st, n) in model.s2n:
            return sum(model.pLineM[ni,nf,cc,l] * model.vDelta_S[p,sc,n,ni,nf,cc,l] for l in model.L) + sum(model.pLineM[ni,nf,cc,l] * model.vDelta_C[p,sc,n,ni,nf,cc,l] for l in model.L) <= model.vW[p,sc,n,nf]
    model.eLineConic1_LP = Constraint(model.ps, model.st, model.n, model.laa, rule=eLineConic1_LP)

    def eLineConic1_LinearS1(model,p,sc,st,n,ni,nf,cc):
        if (st, n) in model.s2n:
            return model.vS_max[p,sc,n,ni,nf,cc] - model.vS_min[p,sc,n,ni,nf,cc] == model.vS[p,sc,n,ni,nf,cc]
    model.eLineConic1_LinearS1 = Constraint(model.ps, model.st, model.n, model.laa, rule=eLineConic1_LinearS1)

    def eLineConic1_LinearS2(model,p,sc,st,n,ni,nf,cc):
        if (st, n) in model.s2n:
            return model.vS_max[p,sc,n,ni,nf,cc] + model.vS_min[p,sc,n,ni,nf,cc] == sum(model.vDelta_S[p,sc,n,ni,nf,cc,l] for l in model.L)
    model.eLineConic1_LinearS2 = Constraint(model.ps, model.st, model.n, model.laa, rule=eLineConic1_LinearS2)

    def eLineConic1_LinearC1(model,p,sc,st,n,ni,nf,cc):
        if (st, n) in model.s2n:
            return (model.vC_max[p,sc,n,ni,nf,cc] - model.vC_min[p,sc,n,ni,nf,cc] == model.vC[p,sc,n,ni,nf,cc])
    model.eLineConic1_LinearC1 = Constraint(model.ps, model.st, model.n, model.laa, rule=eLineConic1_LinearC1)

    def eLineConic1_LinearC2(model,p,sc,st,n,ni,nf,cc):
        if (st, n) in model.s2n:
            return (model.vC_max[p,sc,n,ni,nf,cc] + model.vC_min[p,sc,n,ni,nf,cc] == sum(model.vDelta_C[p,sc,n,ni,nf,cc,l] for l in model.L))
    model.eLineConic1_LinearC2 = Constraint(model.ps, model.st, model.n, model.laa, rule=eLineConic1_LinearC2)

    def eLineConic2(model,p,sc,st,n,ni,nf,cc):
        if (st,n) in model.s2n:
            return model.vW[p,sc,n,ni] + model.vW[p,sc,n,nf] - 2*model.vC[p,sc,n,ni,nf,cc] >= 0
        else:
            return Constraint.Skip
    model.eLineConic2 = Constraint(model.ps, model.st, model.n, model.la, rule=eLineConic2, doc='conic constraint 2 [p.u.]')

    def eExistingShuntGshb(model,p,sc,st,n,sh):
        if (st,n) in model.s2n:
            return model.vBusShuntP[p,sc,n,sh] - pGshb[sh]*model.vW[p,sc,n,pShuntToNode[sh]] == 0
        else:
            return Constraint.Skip
    model.eExistingShuntGshb = Constraint(model.ps, model.st, model.n, model.she, rule=eExistingShuntGshb, doc='shunt constraint [p.u.]')

    def eExistingShuntBshb(model,p,sc,st,n,sh):
        if (st,n) in model.s2n:
            return model.vBusShuntQ[p,sc,n,sh] - pBshb[sh]*model.vW[p,sc,n,pShuntToNode[sh]] == 0
        else:
            return Constraint.Skip
    model.eExistingShuntBshb = Constraint(model.ps, model.st, model.n, model.she, rule=eExistingShuntBshb, doc='shunt constraint [p.u.]')

    GeneratingNetTime = time.time() - StartTime
    StartTime         = time.time()
    print('Generating network    constraints     ... ', round(GeneratingNetTime), 's')

    def eGenFixedInvestment(model,p,gc):
        if pGenSensitivity[gc]:
            # return model.vGenerationInvest[p,gc] == pGenFxInvest[gc] * model.vGenSensi[p,gc]
            return model.vGenerationInvest[p,gc] == pGenFxInvest[gc]
        else:
            return Constraint.Skip
    model.eGenFixedInvestment       = Constraint(model.pgc, rule=eGenFixedInvestment, doc='Fixing the investment decision to a specific value [p.u.]')

    def eGenSensiGroup(model,p,gc,sg):
        if pGenSensitivity[gc] and (sg,gc) in model.sg2g:
            return model.vGenSensiGr[p,sg] == model.vGenSensi[p,gc]
        else:
            return Constraint.Skip
    model.eGenSensiGroup            = Constraint(model.pgc, model.gs, rule=eGenSensiGroup, doc='Coupling the sensitivities per group [p.u.]')

    def eGenSensiGroupValue(model,p,gc,sg):
        if pGenSensitivity[gc] and (sg, gc) in model.sg2g:
            return model.vGenSensiGr[p,sg] == pGenSensiGroupValue[gc]
        else:
            return Constraint.Skip
    model.eGenSensiGroupValue       = Constraint(model.pgc, model.gs, rule=eGenSensiGroupValue, doc='Fixing the variable [p.u.]')

    def eNetFixedInvestment(model,p,ni,nf,cc):
        if pIndBinSingleNode == 0 and pNetSensitivity[ni,nf,cc]:
            return model.vNetworkInvest[p,ni,nf,cc] == pNetFxInvest[ni,nf,cc]
            # return model.vNetworkInvest[p,ni,nf,cc] == pNetFxInvest[ni,nf,cc] * model.vNetSensi[p,ni,nf,cc]
            # return model.vNetworkInvest[p, ni, nf, cc] <= pNetFxInvest[ni, nf, cc]
            # return model.vNetworkInvest[p, ni, nf, cc] <= pNetFxInvest[ni, nf, cc] * 0.99999999
        else:
            return Constraint.Skip
    model.eNetFixedInvestment       = Constraint(model.plc, rule=eNetFixedInvestment, doc='Fixing the investment decision to a specific value [p.u.]')

    def eNetSensiGroup(model,p,ni,nf,cc,sg):
        if pNetSensitivity[ni,nf,cc] and (sg,ni,nf,cc) in model.sg2la:
            return model.vNetSensiGr[p,sg] == model.vNetSensi[p,ni,nf,cc]
        else:
            return Constraint.Skip
    model.eNetSensiGroup            = Constraint(model.plc, model.ns, rule=eNetSensiGroup, doc='Coupling the sensitivities per group [p.u.]')

    def eNetSensiGroupValue(model,p,sg):
        if sg != 'Gr0':
            return model.vNetSensiGr[p,sg] == sum(pNetSensiGroupValue[ni,nf,cc] for (ni,nf,cc) in model.lc if (sg,ni,nf,cc) in model.sg2la)/len([(ni,nf,cc) for (ni,nf,cc) in model.lc if (sg,ni,nf,cc) in model.sg2la])
        else:
            return Constraint.Skip
    model.eNetSensiGroupValue       = Constraint(model.p, model.ns, rule=eNetSensiGroupValue, doc='Fixing the variable [p.u.]')

    GeneratingEATime = time.time() - StartTime
    StartTime         = time.time()
    print('Econometric analysis  constraints     ... ', round(GeneratingEATime), 's')

    # #%% solving the problem
    # model.write(_path+'/openStarNet_'+CaseName+'.lp', io_options={'symbolic_solver_labels': True}) # create lp-format file
    # WritingLPTime = time.time() - StartTime
    # StartTime   = time.time()
    # print('Writing LP file                       ... ', round(WritingLPTime), 's')

    Solver = SolverFactory(SolverName)                                                   # select solver
    if SolverName == 'gurobi':
        Solver.options['LogFile'       ] = _path+'/openStarNet_'+CaseName+'.log'
        #Solver.options['IISFile'      ] = _path+'/openStarNet_'+CaseName+'.ilp'                   # should be uncommented to show results of IIS
        #Solver.options['Method'       ] = 2                                             # barrier method
        Solver.options['Method'        ] = 2                                                 # barrier method
        Solver.options['MIPFocus'      ] = 1
        Solver.options['Presolve'      ] = 2
        Solver.options['RINS'          ] = 100
        Solver.options['Crossover'     ] = 0
        # Solver.options['BarConvTol'    ] = 1e-9
        # Solver.options['BarQCPConvTol' ] = 0.025
        Solver.options['MIPGap'        ] = 0.01
        Solver.options['Threads'       ] = int((psutil.cpu_count(logical=True) + psutil.cpu_count(logical=False))/2)
        Solver.options['TimeLimit'     ] =    76000
        Solver.options['IterationLimit'] = 76000000
    idx = 0
    for var in model.component_data_objects(Var, active=False, descend_into=True):
        if not var.is_continuous():
            idx += 1
    if idx == 0:
        model.dual = Suffix(direction=Suffix.IMPORT)
        model.rc   = Suffix(direction=Suffix.IMPORT)
    SolverResults = Solver.solve(model, tee=True)                                        # tee=True displays the output of the solver
    print('Termination condition: ', SolverResults.solver.termination_condition)
    SolverResults.write()                                                                # summary of the solver results

    #%% fix values of binary variables to get dual variables and solve it again
    print('# ============================================================================= #')
    print('# ============================================================================= #')
    idx = 0
    for var in model.component_data_objects(Var, active=True, descend_into=True):
        if not var.is_continuous():
            print("fixing: " + str(var))
            var.fixed = True  # fix the current value
            idx += 1
    print("Number of fixed variables: ", idx)
    print('# ============================================================================= #')
    print('# ============================================================================= #')
    if idx != 0:
        if SolverName == 'gurobi':
            Solver.options['relax_integrality'] = 1                                          # introduced to show results of the dual variables
        model.dual = Suffix(direction=Suffix.IMPORT)
        model.rc   = Suffix(direction=Suffix.IMPORT)
        SolverResults = Solver.solve(model, tee=False)                                        # tee=True displays the output of the solver
        SolverResults.write()                                                                # summary of the solver results

    SolvingTime = time.time() - StartTime
    StartTime   = time.time()
    print('Solving                               ... ', round(SolvingTime), 's')

    print('Objective function value                  ', model.eTotalSCost.expr())

    # #%% outputting the generation operation
    # SysCost     = pd.Series(data=[                                                                                                             model.vTotalSCost()                                                                                                  ], index=[' ']  ).to_frame(name='Total          System Cost').stack()
    # GenInvCost  = pd.Series(data=[pDiscountFactor[p] * sum(pGenInvestCost[gc  ]                                                              * model.vGenerationInvest[p,gc  ]()  for gc      in model.gc                                     )     for p in model.p], index=model.p).to_frame(name='Generation Investment Cost').stack()
    # NetInvCost  = pd.Series(data=[pDiscountFactor[p] * sum(pNetFixedCost [lc  ]                                                              * model.vNetworkInvest   [p,lc  ]()  for lc      in model.lc                                     )     for p in model.p], index=model.p).to_frame(name='Network    Investment Cost').stack()
    # GenCost     = pd.Series(data=[pDiscountFactor[p] * sum(pScenProb     [p,sc]                                                              * model.vTotalGCost      [p,sc,n]()  for sc,n    in model.sc*model.n           if pScenProb[p,sc])     for p in model.p], index=model.p).to_frame(name='Generation  Operation Cost').stack()
    # ConCost     = pd.Series(data=[pDiscountFactor[p] * sum(pScenProb     [p,sc]                                                              * model.vTotalCCost      [p,sc,n]()  for sc,n    in model.sc*model.n           if pScenProb[p,sc])     for p in model.p], index=model.p).to_frame(name='Consumption Operation Cost').stack()
    # EmiCost     = pd.Series(data=[pDiscountFactor[p] * sum(pScenProb     [p,sc]                                                              * model.vTotalECost      [p,sc,n]()  for sc,n    in model.sc*model.n           if pScenProb[p,sc])     for p in model.p], index=model.p).to_frame(name='Emission              Cost').stack()
    # RelCost     = pd.Series(data=[pDiscountFactor[p] * sum(pScenProb     [p,sc]                                                              * model.vTotalRCost      [p,sc,n]()  for sc,n    in model.sc*model.n           if pScenProb[p,sc])     for p in model.p], index=model.p).to_frame(name='Reliability           Cost').stack()
    # CostSummary = pd.concat([SysCost, GenInvCost, NetInvCost, GenCost, ConCost, EmiCost, RelCost])
    # CostSummary = CostSummary.reset_index().rename(columns={'level_0': 'Period', 'level_1': 'Cost/Payment', 0: 'MEUR'})
    # CostSummary.to_csv(_path+'/3.Out/oT_Result_CostSummary_'+CaseName+'.csv', sep=',', index=False)
    #
    # WritingCostSummaryTime = time.time() - StartTime
    # StartTime              = time.time()
    # print('Writing         cost summary results  ... ', round(WritingCostSummaryTime), 's')

    return model

# def saving_results(DirName, CaseName, SolverName, model):
#
#     _path = os.path.join(DirName, CaseName)
#     StartTime = time.time()
#     print('Objective function value                  ', model.eTotalSCost.expr())
#
    # #%% outputting the generation operation
    # SysCost     = pd.Series(data=[                                                                                                             model.vTotalSCost()                                                                                                  ], index=[' ']  ).to_frame(name='Total          System Cost').stack()
    # GenInvCost  = pd.Series(data=[pDiscountFactor[p] * sum(pGenInvestCost[gc  ]                                                              * model.vGenerationInvest[p,gc  ]()  for gc      in model.gc                                     )     for p in model.p], index=model.p).to_frame(name='Generation Investment Cost').stack()
    # NetInvCost  = pd.Series(data=[pDiscountFactor[p] * sum(pNetFixedCost [lc  ]                                                              * model.vNetworkInvest   [p,lc  ]()  for lc      in model.lc                                     )     for p in model.p], index=model.p).to_frame(name='Network    Investment Cost').stack()
    # GenCost     = pd.Series(data=[pDiscountFactor[p] * sum(pScenProb     [p,sc]                                                              * model.vTotalGCost      [p,sc,n]()  for sc,n    in model.sc*model.n           if pScenProb[p,sc])     for p in model.p], index=model.p).to_frame(name='Generation  Operation Cost').stack()
    # ConCost     = pd.Series(data=[pDiscountFactor[p] * sum(pScenProb     [p,sc]                                                              * model.vTotalCCost      [p,sc,n]()  for sc,n    in model.sc*model.n           if pScenProb[p,sc])     for p in model.p], index=model.p).to_frame(name='Consumption Operation Cost').stack()
    # EmiCost     = pd.Series(data=[pDiscountFactor[p] * sum(pScenProb     [p,sc]                                                              * model.vTotalECost      [p,sc,n]()  for sc,n    in model.sc*model.n           if pScenProb[p,sc])     for p in model.p], index=model.p).to_frame(name='Emission              Cost').stack()
    # RelCost     = pd.Series(data=[pDiscountFactor[p] * sum(pScenProb     [p,sc]                                                              * model.vTotalRCost      [p,sc,n]()  for sc,n    in model.sc*model.n           if pScenProb[p,sc])     for p in model.p], index=model.p).to_frame(name='Reliability           Cost').stack()
    # CostSummary = pd.concat([SysCost, GenInvCost, NetInvCost, GenCost, ConCost, EmiCost, RelCost])
    # CostSummary = CostSummary.reset_index().rename(columns={'level_0': 'Period', 'level_1': 'Cost/Payment', 0: 'MEUR'})
    # CostSummary.to_csv(_path+'/3.Out/oT_Result_CostSummary_'+CaseName+'.csv', sep=',', index=False)
    #
    # WritingCostSummaryTime = time.time() - StartTime
    # StartTime              = time.time()
    # print('Writing         cost summary results  ... ', round(WritingCostSummaryTime), 's')
#
#     #%% outputting the investments
#     if len(model.pgc):
#         OutputResults = pd.Series(data=[model.vGenerationInvest[p,gc]() for p,gc in model.pgc], index=pd.MultiIndex.from_tuples(model.pgc))
#         OutputResults.to_frame(name='p.u.').rename_axis(['Period','Unit'], axis=0).reset_index().to_csv(_path+'/3.Out/oT_Result_GenerationInvestment_'+CaseName+'.csv', index=False, sep=',')
#
#     if len(model.plc):
#         OutputResults = pd.Series(data=[model.vNetworkInvest[p,ni,nf,cc]() for p,ni,nf,cc in model.plc], index=pd.MultiIndex.from_tuples(model.plc))
#         OutputResults.to_frame(name='p.u.').rename_axis(['Period','InitialNode','FinalNode','Circuit'], axis=0).reset_index().to_csv(_path+'/3.Out/oT_Result_NetworkInvestment_'+CaseName+'.csv', index=False, sep=',')
#
#     WritingInvResultsTime = time.time() - StartTime
#     StartTime              = time.time()
#     print('Writing           investment results  ... ', round(WritingInvResultsTime), 's')
#
#     #%%  Power balance per period, scenario, and load level
#     # incoming and outgoing lines (lin) (lout)
#     lin   = defaultdict(list)
#     lout  = defaultdict(list)
#     for ni,nf,cc in model.la:
#         lin  [nf].append((ni,cc))
#         lout [ni].append((nf,cc))
#
#     sPSNARND   = [(p,sc,n,ar,nd)    for p,sc,n,ar,nd    in model.psnar*model.nd if sum(1 for g in model.g if (nd,g) in model.n2g) + sum(1 for lout in lout[nd]) + sum(1 for ni,cc in lin[nd]) and (nd,ar) in model.ndar]
#     sPSNARNDGT = [(p,sc,n,ar,nd,gt) for p,sc,n,ar,nd,gt in sPSNARND*model.gt    if sum(1 for g in model.g if (gt,g) in model.t2g)                                                             and (nd,ar) in model.ndar]
#
#     OutputResults1     = pd.Series(data=[ sum(model.vTotalOutput   [p,sc,n,g       ]()*pLoadLevelDuration[n] for g  in model.g  if (nd,g ) in model.n2g and (gt,g ) in model.t2g) for p,sc,n,ar,nd,gt in sPSNARNDGT                        ], index=pd.Index(sPSNARNDGT)).to_frame(name='Generation'   ).reset_index().pivot_table(index=['level_0','level_1','level_2','level_3','level_4'], columns='level_5', values='Generation' , aggfunc=sum)
#     OutputResults2     = pd.Series(data=[-sum(model.vESSTotalCharge[p,sc,n,es      ]()*pLoadLevelDuration[n] for es in model.es if (nd,es) in model.n2g and (gt,es) in model.t2g) for p,sc,n,ar,nd,gt in sPSNARNDGT                        ], index=pd.Index(sPSNARNDGT)).to_frame(name='Consumption'  ).reset_index().pivot_table(index=['level_0','level_1','level_2','level_3','level_4'], columns='level_5', values='Consumption', aggfunc=sum)
#     OutputResults3     = pd.Series(data=[     model.vENS           [p,sc,n,nd      ]()*pLoadLevelDuration[n]                                                                      for p,sc,n,ar,nd    in sPSNARND                          ], index=pd.Index(sPSNARND  )).to_frame(name='ENS'          )
#     OutputResults4     = pd.Series(data=[-          pDemand        [p,sc,n,nd      ]  *pLoadLevelDuration[n]                                                                      for p,sc,n,ar,nd    in sPSNARND                          ], index=pd.Index(sPSNARND  )).to_frame(name='EnergyDemand' )
#     OutputResults5     = pd.Series(data=[-sum(model.vFlow          [p,sc,n,nd,lout ]()*pLoadLevelDuration[n] for lout  in lout [nd])                                              for p,sc,n,ar,nd    in sPSNARND                          ], index=pd.Index(sPSNARND  )).to_frame(name='EnergyFlowOut')
#     OutputResults6     = pd.Series(data=[ sum(model.vFlow          [p,sc,n,ni,nd,cc]()*pLoadLevelDuration[n] for ni,cc in lin  [nd])                                              for p,sc,n,ar,nd    in sPSNARND                          ], index=pd.Index(sPSNARND  )).to_frame(name='EnergyFlowIn' )
#     OutputResults  = pd.concat([OutputResults1, OutputResults2, OutputResults3, OutputResults4, OutputResults5, OutputResults6                                ], axis=1)
#
#     OutputResults.stack().rename_axis(['Period', 'Scenario', 'LoadLevel', 'Area', 'Node', 'Technology'], axis=0).reset_index().rename(columns={0: 'GWh'}, inplace=False).to_csv(_path+'/3.Out/oT_Result_BalanceEnergy_'+CaseName+'.csv', index=False, sep=',')
#
#     WritingEnergyBalanceTime = time.time() - StartTime
#     StartTime              = time.time()
#     print('Writing       energy balance results  ... ', round(WritingEnergyBalanceTime), 's')
#
#     #%% outputting the network operation
#     OutputResults = pd.Series(data=[model.vFlow[p,sc,n,ni,nf,cc]() for p,sc,n,ni,nf,cc in model.psnla], index=pd.Index(model.psnla))
#     OutputResults.to_frame(name='GWh').rename_axis(['Period', 'Scenario', 'LoadLevel', 'InitialNode', 'FinalNode', 'Circuit'], axis=0).reset_index().to_csv(_path+'/3.Out/oT_Result_NetworkFlowPerNode_'       +CaseName+'.csv', index=False, sep=',')
#
#     # tolerance to consider avoid division by 0
#     pEpsilon = 1e-6
#
#     OutputResults = pd.Series(data=[max(model.vFlow[p,sc,n,ni,nf,cc]()/(pLineNTCFrw[ni,nf,cc]+pEpsilon),-model.vFlow[p,sc,n,ni,nf,cc]()/(pLineNTCFrw[ni,nf,cc]+pEpsilon)) for p,sc,n,ni,nf,cc in model.psnla], index=pd.Index(model.psnla))
#     OutputResults.to_frame(name='GWh').rename_axis(['Period', 'Scenario', 'LoadLevel', 'InitialNode', 'FinalNode', 'Circuit'], axis=0).reset_index().to_csv(_path+'/3.Out/oT_Result_NetworkUtilizationPerNode_'+CaseName+'.csv', index=False, sep=',')
#
#     WritingNetOperTime = time.time() - StartTime
#     StartTime              = time.time()
#     print('Writing    network operation results  ... ', round(WritingNetOperTime), 's')
#
#     #%% outputting the generation operation
#     OutputResults = pd.Series(data=[model.vCommitment[p,sc,n,nr]() for p,sc,n,nr in model.psnnr], index=pd.MultiIndex.from_tuples(model.psnnr))
#     OutputResults.to_frame(name='p.u.').rename_axis(['Period','Scenario','LoadLevel','Unit'], axis=0).reset_index().to_csv(_path+'/3.Out/oT_Result_GenerationCommitment_'+CaseName+'.csv', index=False, sep=',')
#     OutputResults = pd.Series(data=[model.vStartUp   [p,sc,n,nr]() for p,sc,n,nr in model.psnnr], index=pd.MultiIndex.from_tuples(model.psnnr))
#     OutputResults.to_frame(name='p.u.').rename_axis(['Period','Scenario','LoadLevel','Unit'], axis=0).reset_index().to_csv(_path+'/3.Out/oT_Result_GenerationStartUp_'   +CaseName+'.csv', index=False, sep=',')
#     OutputResults = pd.Series(data=[model.vShutDown  [p,sc,n,nr]() for p,sc,n,nr in model.psnnr], index=pd.MultiIndex.from_tuples(model.psnnr))
#     OutputResults.to_frame(name='p.u.').rename_axis(['Period','Scenario','LoadLevel','Unit'], axis=0).reset_index().to_csv(_path+'/3.Out/oT_Result_GenerationShutDown_'  +CaseName+'.csv', index=False, sep=',')
#
#     OutputResults = pd.Series(data=[model.vTotalOutput[p,sc,n,g]()*pLoadLevelDuration[n] for p,sc,n,g in model.psng], index=pd.MultiIndex.from_tuples(model.psng))
#     OutputResults.to_frame(name='GWh' ).rename_axis(['Period','Scenario','LoadLevel','Unit'], axis=0).reset_index().to_csv(_path+'/3.Out/oT_Result_GenerationEnergy_'    +CaseName+'.csv', index=False, sep=',')
#     OutputResults = pd.Series(data=[sum(OutputResults[p,sc,n,g] for g in model.g if (gt,g) in model.t2g) for p,sc,n,gt in model.psngt], index=pd.MultiIndex.from_tuples(model.psngt))
#     OutputResults.to_frame(name='GWh' ).rename_axis(['Period','Scenario','LoadLevel','Technology'], axis=0).reset_index().to_csv(_path+'/3.Out/oT_Result_TechnologyEnergy_'+CaseName+'.csv', index=False, sep=',')
#
#     OutputResults = pd.Series(data=[model.vENS[p,sc,n,nd]()*pLoadLevelDuration[n] for p,sc,n,nd in model.psnnd], index=pd.MultiIndex.from_tuples(model.psnnd))
#     OutputResults.to_frame(name='GWh' ).rename_axis(['Period','Scenario','LoadLevel','Node'], axis=0).reset_index().to_csv(_path+'/3.Out/oT_Result_ENS_'                 +CaseName+'.csv', index=False, sep=',')
#
#     if len(model.r):
#         OutputResults = pd.Series(data=[(pMaxPower[p,sc,n,g]-model.vTotalOutput[p,sc,n,g]())*1e3 for p,sc,n,g in model.psnr], index=pd.MultiIndex.from_tuples(model.psnr))
#         OutputResults.to_frame(name='MW'  ).rename_axis(['Period','Scenario','LoadLevel','Unit'], axis=0).reset_index().to_csv(_path+'/3.Out/oT_Result_RESCurtailment_'      +CaseName+'.csv', index=False, sep=',')
#
#     OutputResults = pd.Series(data=[model.vTotalOutput[p,sc,n,nr]()*pCO2EmissionRate[nr]*1e3 for p,sc,n,nr in model.psn*model.t], index=pd.MultiIndex.from_tuples(model.psn*model.t))
#     OutputResults.to_frame(name='tCO2').rename_axis(['Period','Scenario','LoadLevel','Unit'], axis=0).reset_index().to_csv(_path+'/3.Out/oT_Result_GenerationEmission_'  +CaseName+'.csv', index=False, sep=',')
#
#     WritingGenOperTime = time.time() - StartTime
#     StartTime              = time.time()
#     print('Writing generation operation results  ... ', round(WritingGenOperTime), 's')
#
#     #%% outputting the ESS operation
#     if len(model.es):
#
#         OutputResults = pd.Series(data=[model.vESSTotalCharge   [p,sc,n,es]()*pLoadLevelDuration[n] for p,sc,n,es in model.psnes], index=pd.MultiIndex.from_tuples(model.psnes))
#         OutputResults.to_frame(name='GWh').rename_axis(['Period','Scenario','LoadLevel','Unit'], axis=0).reset_index().to_csv(_path+'/3.Out/oT_Result_ESSConsumptionEnergy_'+CaseName+'.csv', index=False, sep=',')
#         OutputResults = pd.Series(data=[sum(OutputResults[p,sc,n,es] for es in model.es if (gt,es) in model.t2g) for p,sc,n,gt in model.psngt], index=pd.MultiIndex.from_tuples(model.psngt))
#         OutputResults.to_frame(name='GWh').rename_axis(['Period','Scenario','LoadLevel','Technology'], axis=0).reset_index().to_csv(_path+'/3.Out/oT_Result_TechnologyConsumption_'+CaseName+'.csv', index=False, sep=',')
#
#         OutputResults = pd.Series(data=[model.vESSInventory[p,sc,n,es]()                               for p,sc,n,es in model.psnes], index=pd.MultiIndex.from_tuples(model.psnes))
#         OutputResults *= 1e3
#         OutputResults.to_frame(name='GWh').rename_axis(['Period','Scenario','LoadLevel','Unit'], axis=0).reset_index().to_csv(_path+'/3.Out/oT_Result_ESSInventory_'+CaseName+'.csv', index=False, sep=',')
#
#         OutputResults = pd.Series(data=[model.vESSSpillage [p,sc,n,es]()                               for p,sc,n,es in model.psnes], index=pd.MultiIndex.from_tuples(model.psnes))
#         OutputResults *= 1e3
#         OutputResults.to_frame(name='GWh').rename_axis(['Period','Scenario','LoadLevel','Unit'], axis=0).reset_index().to_csv(_path+'/3.Out/oT_Result_ESSSpillage_'+CaseName+'.csv', index=False, sep=',')
#
#     WritingESSOperTime = time.time() - StartTime
#     StartTime              = time.time()
#     print('Writing        ESS operation results  ... ', round(WritingESSOperTime), 's')
#
#     #%% outputting the SRMC
#     if SolverName == 'gurobi':
#         OutputResults = pd.Series(data=[model.dual[model.eBalance[p,sc,st,n,nd]]*1e3/pScenProb[p,sc]/pLoadLevelDuration[n] for p,sc,n,nd,st in model.psnnd*model.st if (st,n) in model.s2n], index=pd.MultiIndex.from_tuples(model.psnnd))
#         OutputResults.to_frame(name='SRMC').rename_axis(['Period','Scenario','LoadLevel','Node'], axis=0).reset_index().to_csv(_path+'/3.Out/oT_Result_SRMC_'+CaseName+'.csv', index=False, sep=',')
#
#     if sum(pGenFxInvest[gc] for gc in model.gc):
#         List1 = [(p,sc,st,n,gc) for p,sc,st,n,gc in model.ps*model.n*model.st*model.n*model.gc if (st,n) in model.s2n and pMaxPower[p,sc,n,gc]]
#         if len(List1):
#             OutputResults = pd.Series(data=[model.dual[model.eInstalGenCap[p,sc,st,n,gc]] for (p,sc,st,n,gc) in List1], index=pd.MultiIndex.from_tuples(list(List1)))
#             OutputResults.to_frame(name='p.u.').rename_axis(['Period','Scenario','Stage','LoadLevel','Unit'], axis=0).reset_index().to_csv(_path+'/3.Out/oT_Result_Dual_eGenFixedOperation_'+CaseName+'.csv', index=False, sep=',')
#
#             OutputResults = pd.Series(data=[model.dual[model.eGenFixedInvestment[p,gc]] for p,gc in model.pgc], index=pd.MultiIndex.from_tuples(model.pgc))
#             OutputResults.to_frame(name='p.u.').rename_axis(['Period','Unit'], axis=0).reset_index().to_csv(_path+'/3.Out/oT_Result_Dual_eGenFixedInvestment_'+CaseName+'.csv', index=False, sep=',')
#
#         List2 = [(p,gc,sg) for p,gc,sg in model.pgc*model.gs if pGenSensitivity[gc] and (sg,gc) in model.sg2g]
#         if len(List2):
#             OutputResults = pd.Series(data=[model.dual[model.eGenSensiGroup[p,gc,sg]] for p,gc,sg in List2], index=pd.MultiIndex.from_tuples(list(List2)))
#             OutputResults.to_frame(name='p.u.').rename_axis(['Period','Unit','Group'], axis=0).reset_index().to_csv(_path+'/3.Out/oT_Result_Dual_eGenSensitivityCoupling_'+CaseName+'.csv', index=False, sep=',')
#
#             OutputResults = pd.Series(data=[model.dual[model.eGenSensiGroupValue[p,gc,sg]] for p,gc,sg in List2], index=pd.MultiIndex.from_tuples(list(List2)))
#             OutputResults.to_frame(name='p.u.').rename_axis(['Period','Unit','Group'], axis=0).reset_index().to_csv(_path+'/3.Out/oT_Result_Dual_eGenSensitivityCouplingValue_'+CaseName+'.csv', index=False, sep=',')
#
#     List3 = [(p,sc,st,n,ni,nf,cc) for p,sc,st,n,ni,nf,cc in model.ps*model.st*model.n*model.lc if (st,n) in model.s2n and pIndBinSingleNode == 0]
#     if len(List3):
#         # OutputResults = pd.Series(data=[model.dual[model.eNetCapacity1[p,sc,st,n,ni,nf,cc]]/pPeriodProb[p,sc]/pLoadLevelDuration[n]*1e3 for (p,sc,st,n,ni,nf,cc) in List3], index=pd.MultiIndex.from_tuples(list(List3)))
#         OutputResults = pd.Series(data=[model.dual[model.eNetCapacity1[p,sc,st,n,ni,nf,cc]] for (p,sc,st,n,ni,nf,cc) in List3], index=pd.MultiIndex.from_tuples(list(List3)))
#         OutputResults.to_frame(name='EUR/MW').rename_axis(['Period','Scenario','Stage','LoadLevel','InitialNode','FinalNode','Circuit'], axis=0).reset_index().to_csv(_path+'/3.Out/oT_Result_Dual_eNetCapacityOperation_LowerBound_'+CaseName+'.csv', index=False, sep=',')
#
#         # OutputResults = pd.Series(data=[model.dual[model.eNetCapacity2[p,sc,st,n,ni,nf,cc]]/pPeriodProb[p,sc]/pLoadLevelDuration[n]*1e3 for (p,sc,st,n,ni,nf,cc) in List3], index=pd.MultiIndex.from_tuples(list(List3)))
#         OutputResults = pd.Series(data=[model.dual[model.eNetCapacity2[p,sc,st,n,ni,nf,cc]] for (p,sc,st,n,ni,nf,cc) in List3], index=pd.MultiIndex.from_tuples(list(List3)))
#         OutputResults.to_frame(name='EUR/MW').rename_axis(['Period','Scenario','Stage','LoadLevel','InitialNode','FinalNode','Circuit'], axis=0).reset_index().to_csv(_path+'/3.Out/oT_Result_Dual_eNetCapacityOperation_UpperBound_'+CaseName+'.csv', index=False, sep=',')
#
#     List4 = [(p,sc,st,n,ni,nf,cc) for p,sc,st,n,ni,nf,cc in model.ps*model.st*model.n*model.lca if (st,n) in model.s2n and pIndBinSingleNode == 0]
#     if len(List4):
#         OutputResults = pd.Series(data=[model.dual[model.eKirchhoff2ndLaw1[p,sc,st,n,ni,nf,cc]] for (p,sc,st,n,ni,nf,cc) in List4], index=pd.MultiIndex.from_tuples(list(List4)))
#         OutputResults.to_frame(name='p.u.').rename_axis(['Period','Scenario','Stage','LoadLevel','InitialNode','FinalNode','Circuit'], axis=0).reset_index().to_csv(_path+'/3.Out/oT_Result_Dual_eNetAngleDifference_LowerBound_'+CaseName+'.csv', index=False, sep=',')
#
#         OutputResults = pd.Series(data=[model.dual[model.eKirchhoff2ndLaw2[p,sc,st,n,ni,nf,cc]] for (p,sc,st,n,ni,nf,cc) in List4], index=pd.MultiIndex.from_tuples(list(List4)))
#         OutputResults.to_frame(name='p.u.').rename_axis(['Period','Scenario','Stage','LoadLevel','InitialNode','FinalNode','Circuit'], axis=0).reset_index().to_csv(_path+'/3.Out/oT_Result_Dual_eNetAngleDifference_UpperBound_'+CaseName+'.csv', index=False, sep=',')
#
#     if sum(pNetFxInvest[ni,nf,cc] for ni,nf,cc in model.lc):
#         if len(List3):
#             # OutputResults = pd.Series(data=[model.dual[model.eNetFixedInvestment[p,ni,nf,cc]]*1e3 for p,ni,nf,cc in model.plc], index=pd.MultiIndex.from_tuples(model.plc))
#             OutputResults = pd.Series(data=[model.dual[model.eNetFixedInvestment[p,ni,nf,cc]] for p,ni,nf,cc in model.plc], index=pd.MultiIndex.from_tuples(model.plc))
#             OutputResults.to_frame(name='EUR/MW').rename_axis(['Period','InitialNode','FinalNode','Circuit'], axis=0).reset_index().to_csv(_path+'/3.Out/oT_Result_Dual_eNetFixedInvestment_'+CaseName+'.csv', index=False, sep=',')
#
#         # List5 = [(p,ni,nf,cc,sg) for p,ni,nf,cc,sg in model.plc*model.ns if pNetSensitivity[ni,nf,cc] and (sg,ni,nf,cc) in model.sg2la]
#         # if len(List5):
#         #     OutputResults = pd.Series(data=[model.dual[model.eNetSensiGroup[p,ni,nf,cc,sg]]*1e3 for p,ni,nf,cc,sg in List5], index=pd.MultiIndex.from_tuples(list(List5)))
#         #     OutputResults.to_frame(name='EUR/MW').rename_axis(['Period','InitialNode','FinalNode','Circuit','Group'], axis=0).reset_index().to_csv(_path+'/3.Out/oT_Result_Dual_eNetSensitivityCoupling_'+CaseName+'.csv', index=False, sep=',')
#         # List6 = [(p,sg) for p,sg in model.p*model.ns if sg != 'Gr0']
#         # if len(List6):
#         #     OutputResults = pd.Series(data=[model.dual[model.eNetSensiGroupValue[p,sg]]*1e3 for p,sg in List6], index=pd.MultiIndex.from_tuples(list(List6)))
#         #     OutputResults.to_frame(name='EUR/MW').rename_axis(['Period','Group'], axis=0).reset_index().to_csv(_path+'/3.Out/oT_Result_Dual_eNetSensitivityCouplingValue_'+CaseName+'.csv', index=False, sep=',')
#
#     #%% Reduced cost for NetworkInvestment
#     ReducedCostActivation = pIndBinGenInvest*len(model.gc) + pIndBinNetInvest*len(model.lc) + pIndBinGenOperat*len(model.nr)
#     if len(model.lc) and not ReducedCostActivation and model.vNetworkInvest.is_variable_type():
#         OutputToFile = pd.Series(data=[model.rc[model.vNetworkInvest[p,ni,nf,cc]] for p,ni,nf,cc in model.plc], index=pd.Index(model.plc))
#         # OutputToFile.index.names = ['Period', 'InitialNode', 'FinalNode', 'Circuit']
#         OutputToFile.to_frame(name='MEUR').rename_axis(['Period','InitialNode','FinalNode','Circuit'], axis=0).to_csv(_path+'/3.Out/oT_Result_ReducedCost_NetworkInvestment_'+CaseName+'.csv', sep=',')
#
#     WritingEconomicTime = time.time() - StartTime
#     print('Writing             economic results  ... ', round(WritingEconomicTime), 's')
#     print('Total time                            ... ', round(ReadingDataTime + SettingUpDataTime + GeneratingOFTime + GeneratingInvTime + GeneratingBalanceTime + GeneratingESSTime + GeneratingUCTime + GeneratingNetTime + GeneratingEATime + WritingLPTime + SolvingTime + WritingCostSummaryTime + WritingInvResultsTime + WritingEnergyBalanceTime + WritingNetOperTime + WritingGenOperTime + WritingESSOperTime + WritingEconomicTime), 's')
#     print('\n #### Academic research license - for non-commercial use only #### \n')
#
#     return model

if __name__ == '__main__':
    main(openStarNet)
