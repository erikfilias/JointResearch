
echo "Starting runs"

# List of origin folders to iterate over
origin_folders=("B" "D" "E" "K" "L")

# Loop over each origin folder
for origin_folder in "${origin_folders[@]}"; do
    echo "Running for origin_folder: $origin_folder"
    python loop_operational_in_mem.py --case 9n_mod1 --origin_folder "$origin_folder" --min_nb 10 --max_nb 400
done
