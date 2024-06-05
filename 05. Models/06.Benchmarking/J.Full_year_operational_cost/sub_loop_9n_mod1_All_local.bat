@echo off
echo Starting runs

rem List of origin folders to iterate over
set origin_folders=B D E K L

rem Loop over each origin folder
for %%i in (%origin_folders%) do (
    echo Running for origin_folder: %%i
    python loop_operational_in_mem.py --case 9n_mod1 --origin_folder %%i --min_nb 10 --max_nb 400
)
