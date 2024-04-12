#!/bin/bash -l


#SBATCH --cluster="genius"
#SBATCH --job-name="118_mod1_fy_se_p4"
#SBATCH --nodes="1"
#SBATCH --mail-user="kristof.phillips@kuleuven.be"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time="72:00:00"
#SBATCH --ntasks-per-node="36"
#SBATCH --account="lp_elect_gen_modeling"
#SBATCH --partition="batch_long"
      
source activate Jr23
echo "Activation OK"
cd $VSC_SCRATCH/JointResearch/05.Models/06.Benchmarking/E.Representative_days_based_on_Line_Benefits_OptModel

echo "Starting runs"

python TrainingDataGenerator_DCOPF_part_i_out_j.py --case IEEE118_mod1 --i 4 --j 5
