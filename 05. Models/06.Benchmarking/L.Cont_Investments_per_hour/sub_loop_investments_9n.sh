#!/bin/bash -l


#SBATCH --cluster="genius"
#SBATCH --job-name="9n_IPHC"
#SBATCH --nodes="1"
#SBATCH --mail-user="kristof.phillips@kuleuven.be"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time="4:00:00"
#SBATCH --ntasks-per-node="36"
#SBATCH --account="lp_elect_gen_modeling"
#SBATCH --partition="batch"
      
source activate Jr23
echo "Activation OK"
cd $VSC_SCRATCH/JointResearch/05.Models/06.Benchmarking/L.Cont_Investments_per_hour

echo "Starting runs"

python loop_hourly_investments.py --case 9n
