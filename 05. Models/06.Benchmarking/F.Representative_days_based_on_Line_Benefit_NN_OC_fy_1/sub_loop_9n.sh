#!/bin/bash -l


#SBATCH --cluster="genius"
#SBATCH --job-name="9n_by_stages_loop_F"
#SBATCH --nodes="1"
#SBATCH --mail-user="kristof.phillips@kuleuven.be"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time="36:00:00"
#SBATCH --ntasks-per-node="36"
#SBATCH --account="lp_elect_gen_modeling"
#SBATCH --partition="batch"
      
source activate Jr23
echo "Activation OK"
cd $VSC_SCRATCH/JointResearch/05.Models/06.Benchmarking/F.Representative_days_based_on_Line_Benefit_NN_OC_fy_1

echo "Starting runs"

python loop_by_stages.py --case 9n