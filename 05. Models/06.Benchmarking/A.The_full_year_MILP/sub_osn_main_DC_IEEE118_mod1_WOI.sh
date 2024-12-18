#!/bin/bash -l


#SBATCH --cluster="genius"
#SBATCH --job-name="IEEE_mod1_full_MILP_DC_WOI"
#SBATCH --nodes="1"
#SBATCH --mail-user="kristof.phillips@kuleuven.be"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time="6:00:00"
#SBATCH --ntasks-per-node="36"
#SBATCH --account="lp_elect_gen_modeling"
#SBATCH --partition="batch"
    
source activate Jr23
echo "Activation OK"
cd $VSC_SCRATCH/JointResearch/05.Models/06.Benchmarking/A.The_full_year_MILP

echo "Starting runs"

python oSN_Main_DCOPF_WithoutInvestment.py --case IEEE118_mod1

