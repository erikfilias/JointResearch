#!/bin/bash -l


#SBATCH --cluster="genius"
#SBATCH --job-name="RTS24_mod1_by_stages_loop_M_10-250"
#SBATCH --nodes="1"
#SBATCH --mail-user="kristof.phillips@kuleuven.be"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time="12:00:00"
#SBATCH --ntasks-per-node="36"
#SBATCH --account="lp_elect_gen_modeling"
#SBATCH --partition="batch"
      
source activate Jr23
echo "Activation OK"
cd $VSC_SCRATCH/JointResearch/05.Models/06.Benchmarking/M.Random_forest

echo "Starting runs"

python loop_by_stages.py --case RTS24_mod1 --min_nb 10 --max_nb 250