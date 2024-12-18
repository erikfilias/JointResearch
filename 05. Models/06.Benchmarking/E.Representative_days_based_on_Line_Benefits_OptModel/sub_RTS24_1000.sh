#!/bin/bash -l


#SBATCH --cluster="genius"
#SBATCH --job-name="RTS24_by_stages_E_1000"
#SBATCH --nodes="1"
#SBATCH --mail-user="kristof.phillips@kuleuven.be"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time="72:00:00"
#SBATCH --ntasks-per-node="36"
#SBATCH --account="lp_elect_gen_modeling"
#SBATCH --partition="batch"
      
source activate Jr23
echo "Activation OK"
cd $VSC_SCRATCH/JointResearch/05.Models/06.Benchmarking/E.Representative_days_based_on_Line_Benefits_OptModel

echo "Starting runs"

python loop_by_stages.py --case RTS24 --min_nb 1000 --max_nb 1000


