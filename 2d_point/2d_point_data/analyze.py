from fileinput import filename
import pandas as pd
import os

directory = os.getcwd()
rrt_types = ["rrt_reg", "rrt_march", "rrt_var", "rrt_star", "rrt_bi"]
#rrt_names = {
#    "rrt_reg"   : "RRT",
#    "rrt_march" : "RRT March",
#    "rrt_var"   : "RRT with variable step size",
#    "rrt_star"  : "RRT*",
#    "rrt_bi"    : "Bidirectional RRT"
#}

rrt_names = {
    "rrt_reg"   : "rrt_reg",
    "rrt_march" : "rrt_march",
    "rrt_var"   : "rrt_var",
    "rrt_star"  : "rrt_star",
    "rrt_bi"    : "rrt_bi"
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
    rrt_lst = []
    mean_lst = []
    std_lst = []
    range_lst = []
    median_lst = []
    for file in csv_files:
        if grid in file:
            parts = file.split("_")
            rrt_type = parts[0] + "_" + parts[1]
            rrt_name = rrt_names[rrt_type]

            df = pd.read_csv(file)
            mean = df[metric].mean()
            std = df[metric].std()
            min = round(df[metric].min(),3)
            max = round(df[metric].max(),3)
            median = df[metric].median()

            range_str = "({},{})".format(min,max)
            rrt_lst.append(rrt_name)
            mean_lst.append(round(mean,3))
            std_lst.append(round(std,3))
            range_lst.append(range_str)
            median_lst.append(round(median,3))

    # create a dictionary with the three lists
    dict = {'Algorithm': rrt_lst, 'Mean': mean_lst, 'Std': std_lst, 'Range': range_lst, 'Median': median_lst}  
        
    # create a Pandas DataFrame from the dictionary
    df = pd.DataFrame(dict) 

    # write the DataFrame to a CSV file
    filename = "output/{}_{}_2d_point.csv".format(grid, metric)
    df.to_csv(filename, index = False)

if __name__ == "__main__":
    files = get_csv_filenames()
    #Nodes, Steps, Dist, Time
    grids = ["grid1", "grid2", "grid3"]
    metrics = ["Nodes", "Steps", "Dist", "Time"]
    for grid in grids:
        for metric in metrics:
            get_details(grid, files, metric)


