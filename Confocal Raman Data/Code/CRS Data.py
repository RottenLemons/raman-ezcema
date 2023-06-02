import glob
import subprocess

# path to folder containing all the sub-folders
path = r"./"

# loop through all sub-folders
for folder in glob.glob(path + "*/"):
    # path to csv files in the current sub-folder
    csv_path = folder + "*.csv"

    # loop through all csv files in the current sub-folder
    for file in glob.glob(csv_path):
        # path to the script that processes the csv files
        script_path = r"Unmixing_with_water_and_average_components.py"
        # run the script with the current csv file as argument
        subprocess.run(["python", script_path, file])
