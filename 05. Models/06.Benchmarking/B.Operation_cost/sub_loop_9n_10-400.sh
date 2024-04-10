#!/bin/bash -l


#SBATCH --cluster="genius"
#SBATCH --job-name="9n_by_stages_loop_B_10-400"
#SBATCH --nodes="1"
#SBATCH --mail-user="kristof.phillips@kuleuven.be"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time="6:00:00"
#SBATCH --ntasks-per-node="12"
#SBATCH --account="lp_elect_gen_modeling"
#SBATCH --partition="batch"
      
source activate Jr23
echo "Activation OK"
cd $VSC_SCRATCH/JointResearch/05.Models/06.Benchmarking/B.Operation_cost

echo "Starting runs"

python loop_by_stages.py --case 9n --min_nb 10 --max_nb 400

