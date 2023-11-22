#!/bin/bash -l


#SBATCH --cluster="genius"
#SBATCH --job-name="118_by_stages_loop_D"
#SBATCH --nodes="1"
#SBATCH --mail-user="kristof.phillips@kuleuven.be"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time="72:00:00"
#SBATCH --ntasks-per-node="36"
#SBATCH --account="lp_elect_gen_modeling"
#SBATCH --partition="batch"
      
source activate Jr23
echo "Activation OK"
cd $VSC_SCRATCH/JointResearch/05.Models/06.Benchmarking/D.Representative_days_based_on_RES_and_Demand

echo "Starting runs"

python loop_by_stages.py --case IEEE118 --min_nb 10 --max_nb 250
