#!/bin/bash -l


#SBATCH --cluster="genius"
#SBATCH --job-name="3-bus_fy_se"
#SBATCH --nodes="1"
#SBATCH --mail-user="kristof.phillips@kuleuven.be"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time="24:00:00"
#SBATCH --ntasks-per-node="36"
#SBATCH --account="lp_elect_gen_modeling"
#SBATCH --partition="batch"
      
source activate Jr23
echo "Activation OK"
cd $VSC_SCRATCH/JointResearch/05.Models/06.Benchmarking/E.Representative_days_based_on_Line_Benefits_OptModel

echo "Starting runs"

python TrainingDataGenerator_DCOPF.py --case 3-bus
