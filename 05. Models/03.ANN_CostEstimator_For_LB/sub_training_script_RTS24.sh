#!/bin/bash -l


#SBATCH --cluster="genius"
#SBATCH --job-name="training_hyper"
#SBATCH --nodes="1"
#SBATCH --mail-user="kristof.phillips@kuleuven.be"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time="36:00:00"
#SBATCH --ntasks-per-node="12"
#SBATCH --account="lp_elect_gen_modeling"
#SBATCH --partition="batch"
    
source activate Jr23
echo "Activation OK"
cd $VSC_SCRATCH/JointResearch/05.Models/03.ANN_CostEstimator_For_LB

echo "Starting runs"

python hyperparams_training_script_v3.py --case RTS24

