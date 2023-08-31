# Librarie(s) necessaire(s)
import argparse      # parse command line options
import os            # path
import time          # count clock time
import numpy         as np
import pandas        as pd
from   oSN_Main_v2   import *
from   collections   import defaultdict
from   pyomo.environ import ConcreteModel, Set, Param, Var, Objective, minimize, Constraint, DataPortal, PositiveIntegers, NonNegativeIntegers, Boolean, NonNegativeReals, UnitInterval, PositiveReals, Any, Binary, Reals, Suffix


# Calling the main function
def main():
    args = parser.parse_args()
    # if args.dir is None:
    #     args.dir    = input('Input Dir    Name (Default {}): '.format(DIR))
    #     if args.dir == '':
    args.dir = DIR
    # if args.case is None:
    #     args.case   = input('Input Case   Name (Default {}): '.format(CASE))
    #     if args.case == '':
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

    # Reading and data processing
    base_model = data_processing(args.dir, args.case, base_model)

    dict_n    = [(n)      for (     n) in base_model.n  ]
    dict_psn  = [(p,sc,n) for (p,sc,n) in base_model.psn]

    print('Number elements in the set psn: ', len(dict_psn))

    # create the model
    oSN       = ConcreteModel()

    for (p,sc,n) in base_model.psn:
        # removing the sets from the base model
        base_model.del_component(base_model.n  )
        base_model.del_component(base_model.psn)

        # defining the sets
        base_model.n   = Set(initialize=dict_n  , ordered=True)


if __name__ == '__main__':
    t_start = time.time()
    main()
    total_time = time.time() - t_start
    print('########################################################')
    print('Total time                            ... ', round(total_time), 's')
    print('########################################################')