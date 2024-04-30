#!/bin/bash -l


#SBATCH --cluster="genius"
#SBATCH --job-name="118_BS_K_1000"
#SBATCH --nodes="1"
#SBATCH --mail-user="kristof.phillips@kuleuven.be"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time="72:00:00"
#SBATCH --ntasks-per-node="36"
#SBATCH --account="lp_elect_gen_modeling"
#SBATCH --partition="batch"
      
source activate Jr23
echo "Activation OK"
cd $VSC_SCRATCH/JointResearch/05.Models/06.Benchmarking/K.Investments_per_hour

echo "Starting runs"


python loop_by_stages.py --case IEEE118 --min_nb 1000 --max_nb 1000
