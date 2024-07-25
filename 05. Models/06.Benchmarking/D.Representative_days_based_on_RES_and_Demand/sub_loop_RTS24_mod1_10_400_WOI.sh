#!/bin/bash -l


#SBATCH --cluster="genius"
#SBATCH --job-name="RTS24_mod1_by_stages_loop_D_10-400_WOI"
#SBATCH --nodes="1"
#SBATCH --mail-user="kristof.phillips@kuleuven.be"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time="10:00:00"
#SBATCH --ntasks-per-node="36"
#SBATCH --account="lp_elect_gen_modeling"
#SBATCH --partition="batch"
      
source activate Jr23
echo "Activation OK"
cd $VSC_SCRATCH/JointResearch/05.Models/06.Benchmarking/D.Representative_days_based_on_RES_and_Demand

echo "Starting runs"

python loop_by_stages_WOI.py --case RTS24_mod1 --min_nb 10 --max_nb 400
