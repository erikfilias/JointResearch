#!/bin/bash -l


#SBATCH --cluster="genius"
#SBATCH --job-name="RTS24_mod2_FYOP_loop_E"
#SBATCH --nodes="1"
#SBATCH --mail-user="kristof.phillips@kuleuven.be"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time="10:00:00"
#SBATCH --ntasks-per-node="36"
#SBATCH --account="lp_elect_gen_modeling"
#SBATCH --partition="batch"
      
source activate Jr23
echo "Activation OK"
cd $VSC_SCRATCH/JointResearch/05.Models/06.Benchmarking/J.Full_year_operational_cost

echo "Starting runs"

python loop_operational_in_mem.py --case RTS24_mod2 --origin_folder E --min_nb 10 --max_nb 400
