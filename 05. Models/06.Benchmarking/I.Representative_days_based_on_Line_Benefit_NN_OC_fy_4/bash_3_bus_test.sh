#!/bin/bash

# Get the directory of the script
script_dir="$(dirname "$0")"

# Define the string you want to check for
search_string="3-bus_ByStages_nc2"

#add python to path variables 

PATH+= :C:\Workdir\Programs\Miniconda\python.exe


# Check if the script's directory exists
if [ -d "$script_dir" ]; then
    # Loop through the items in the script's directory
    for item in "$script_dir"/*; do
        # Check if the item is a directory
        if [ -d "$item" ]; then
            # Get the directory name
            dir_name="$(basename "$item")"
            
            # Check if the directory name contains the search string
            if [[ "$dir_name" == *"$search_string"* ]]; then
                # Print the directory name
                echo "Directory: $item"
                python --version
                #python oSN_Main_v2.py --case "3-bus" --solver gurobi --dir script_dir
            fi
        fi
    done
else
    echo "Directory not found: $script_dir"
fi