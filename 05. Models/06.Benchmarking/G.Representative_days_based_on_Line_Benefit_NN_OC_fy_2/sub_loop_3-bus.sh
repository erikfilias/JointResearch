#!/bin/bash -l


#SBATCH --cluster="genius"
#SBATCH --job-name="3-bus_by_stages_loop_G"
#SBATCH --nodes="1"
#SBATCH --mail-user="kristof.phillips@kuleuven.be"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time="36:00:00"
#SBATCH --ntasks-per-node="36"
#SBATCH --account="lp_elect_gen_modeling"
#SBATCH --partition="batch"
      
source activate Jr23
echo "Activation OK"
cd $VSC_SCRATCH/JointResearch/05.Models/06.Benchmarking/G.Representative_days_based_on_Line_Benefit_NN_OC_fy_2

echo "Starting runs"

python loop_by_stages.py --case 3-bus