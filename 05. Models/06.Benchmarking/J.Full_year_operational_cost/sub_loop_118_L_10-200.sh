#!/bin/bash -l


#SBATCH --cluster="genius"
#SBATCH --job-name="118_FYOP_loop_L_10-200"
#SBATCH --nodes="1"
#SBATCH --mail-user="kristof.phillips@kuleuven.be"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time="36:00:00"
#SBATCH --ntasks-per-node="36"
#SBATCH --account="lp_elect_gen_modeling"
#SBATCH --partition="batch"
      
source activate Jr23
echo "Activation OK"
cd $VSC_SCRATCH/JointResearch/05.Models/06.Benchmarking/J.Full_year_operational_cost

echo "Starting runs"

python loop_operational_in_mem.py --case IEEE118 --origin_folder L --min_nb 10 --max_nb 200
