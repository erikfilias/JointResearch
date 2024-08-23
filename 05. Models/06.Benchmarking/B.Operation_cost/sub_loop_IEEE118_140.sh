#!/bin/bash -l


#SBATCH --cluster="genius"
#SBATCH --job-name="IEEE118_by_stages_loop_B_140"
#SBATCH --nodes="1"
#SBATCH --mail-user="kristof.phillips@kuleuven.be"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time="168:00:00"
#SBATCH --ntasks-per-node="36"
#SBATCH --account="lp_elect_gen_modeling"
#SBATCH --partition="batch_long"
      
source activate Jr23
echo "Activation OK"
cd $VSC_SCRATCH/JointResearch/05.Models/06.Benchmarking/B.Operation_cost

echo "Starting runs"

python loop_by_stages.py --case IEEE118 --min_nb 140 --max_nb 140

