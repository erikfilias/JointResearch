#!/bin/bash -l

#SBATCH --cluster="genius"
#SBATCH --job-name="9n_FYOP_loop_All"
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

# List of origin folders to iterate over
origin_folders=("B" "D" "E" "K" "L")

# Loop over each origin folder
for origin_folder in "${origin_folders[@]}"; do
    echo "Running for origin_folder: $origin_folder"
    python loop_operational_in_mem.py --case 9n --origin_folder "$origin_folder" --min_nb 10 --max_nb 400
done
