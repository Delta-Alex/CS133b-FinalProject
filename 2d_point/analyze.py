from fileinput import filename
import pandas as pd
import os

directory = os.getcwd()
rrt_types = ["rrt_reg", "rrt_march", "rrt_var", "rrt_star", "rrt_bi"]
rrt_names = {
    "rrt_reg"   : "RRT",
    "rrt_march" : "RRT March",
    "rrt_var"   : "RRT with variable step size",
    "rrt_star"  : "RRT*",
    "rrt_bi"    : "Bidirectional RRT"
}

def get_csv_filenames():
    csv_files = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"): 
            #print(os.path.join(directory, filename))
            #print(filename)
            csv_files.append(filename)
    return csv_files

def get_details(grid, csv_files, metric):
    print("-" * 50)
    print(grid.upper())
    for file in csv_files:
        if grid in file:
            parts = file.split("_")
            rrt_type = parts[0] + "_" + parts[1]
            rrt_name = rrt_names[rrt_type]

            if metric.lower()  == "mean":
                df = pd.read_csv(file)
                nodes = df['Nodes'].mean()
                steps = df['Steps'].mean()
                dist = df['Dist'].mean()
                time =df['Time'].mean()
                output = "{}, Mean Nodes: {}, Mean Steps:{}, Mean Dist: {}, Mean Time: {}".format(rrt_name, nodes, steps, dist, time)
                print(output)

            elif metric.lower() == "std":
                df = pd.read_csv(file)
                nodes = df['Nodes'].std()
                steps = df['Steps'].std()
                dist = df['Dist'].std()
                time =df['Time'].std()
                output = "{}, Std Nodes: {}, Std Steps:{}, Std Dist: {}, Std Time: {}".format(rrt_name, nodes, steps, dist, time)
                print(output)

            elif metric.lower() == "range":
                df = pd.read_csv(file)
                nodes_min, nodes_max = df['Nodes'].min(), df['Nodes'].max()
                steps_min, steps_max = df['Steps'].min(), df['Steps'].max()
                dist_min, dist_max = df['Dist'].min(), df['Dist'].max()
                time_min, time_max = df['Time'].min(), df['Time'].max()
                output = "{}, Nodes: ({}, {}), Steps: ({}, {}), Dist: ({},{}), Time: ({}, {})".format(rrt_name, nodes_min, nodes_max, steps_min, steps_max, dist_min, dist_max, time_min, time_max)
                print(output)
                pass

            elif metric.lower() == "median":
                df = pd.read_csv(file)
                nodes = df['Nodes'].median()
                steps = df['Steps'].median()
                dist = df['Dist'].median()
                time = df['Time'].median()
                output = "{}, Median Nodes: {}, Median Steps:{}, Median Dist: {}, Median Time: {}".format(rrt_name, nodes, steps, dist, time)
                print(output)
            
            else:
                pass

if __name__ == "__main__":
    files = get_csv_filenames()
    get_details("grid1", files, "median")


