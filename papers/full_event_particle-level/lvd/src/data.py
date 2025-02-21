import numpy as np
import pandas as pd

# filepath dictionaries
mcpath_filepath = {
    "mc0.4-0.5_pom": "/home/tkj9ep/dev/fox-diffusion/papers/full_event_particle-level/lvd/data/mc0.4-0.5_pom.dat",
    "mc0.5-0.6_pom": "/home/tkj9ep/dev/fox-diffusion/papers/full_event_particle-level/lvd/data/mc0.5-0.6_pom.dat",
    "mc0.6-0.7_pom": "/home/tkj9ep/dev/fox-diffusion/papers/full_event_particle-level/lvd/data/mc0.6-0.7_pom.dat",
    "mc0.7-0.8_pom": "/home/tkj9ep/dev/fox-diffusion/papers/full_event_particle-level/lvd/data/mc0.7-0.8_pom.dat",
    "mc0.8-0.9_pom": "/home/tkj9ep/dev/fox-diffusion/papers/full_event_particle-level/lvd/data/mc0.8-0.9_pom.dat",
    "mc0.9-1.0_pom": "/home/tkj9ep/dev/fox-diffusion/papers/full_event_particle-level/lvd/data/mc0.9-1.0_pom.dat"
}

def main():
    filepath = mcpath_filepath["mc0.4-0.5_pom"]

    results, cols = convert_df_to_np(filepath)
    eda_dict = eda(results)

    print(eda_dict["mean"])
    print(eda_dict["std"])

    return 0

def display_file(filepath):
    with open(filepath, 'r') as file:
        for line in file:
            print(line.strip())


def convert_df_to_np(filepath):
    df = pd.read_csv(filepath, delimiter = '\t')
    results = []
    cols = df.columns.values.tolist()[0].split() # gets the columns

    for i in range(10):
        entry = df.iloc[i].tolist()[0].split()
        int_list = [float(item) for item in entry]
        results.append(int_list)

    # perform EDA

    return np.array(results), np.array(cols)

def eda(df_np):
    eda_dict = {}

    eda_dict["mean"] = np.mean(df_np, axis = 0)
    eda_dict["std"] = np.std(df_np, axis = 0)


    return eda_dict

main()