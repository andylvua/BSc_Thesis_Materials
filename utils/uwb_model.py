import numpy as np
import pandas as pd
import ast
import csv

def uwb_to_df(file_path: str):
    if file_path.endswith('.pkl'):
        print(f"Loading data from pickle file: {file_path}")
        df = pd.read_pickle(file_path)
        return df
    
    with open(file_path, mode='r') as file:
        data = file.readlines()

        data = [x for x in data if x[0] == '{']
        data = [ast.literal_eval(x) for x in data]

        df = pd.DataFrame(data)

        def to_complex_array(arr):
            np_arr = np.array(arr)
            reshaped = np_arr.reshape(-1, 2)
            complex_arr = reshaped[:, 0] + 1j * reshaped[:, 1]
            return complex_arr

        df['CIR'] = df['CIR'].apply(lambda x: to_complex_array(x))

    return df


def read_beacons_coordinates(file_path: str):
    beacons = {}
    with open(file_path, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            x = float(row['x'])
            y = float(row['y'])
            z = float(row['z'])
            id_val = str(row['id'])
            beacons[id_val] = np.array((x, y, z))
    return beacons


class UWBModel:
    def __init__(self, data_path: str = "./Data/UWB_Data/data.txt",
                 coord_path: str = "./Data/UWB_Data/beacons_coord.csv",
                 idx_from: int = 0, 
                 idx_to: int = -1):
        self.dist_bias = 0 # cm

        self.df = uwb_to_df(data_path)

        if not data_path.endswith('.pkl'):
            self.df = self.df[idx_from:idx_to]

            self.df = self.df[self.df['s'] == 0]

        self.t = np.array(self.df["t"]) / 1e6

        self.auto_dist = np.array(self.df["D"] - self.dist_bias) / 100

        self.ids = np.array(self.df["a"])

        self.beacons = read_beacons_coordinates(coord_path)

        self.cir = self.df["CIR"].values

        print(f"Loaded experimental data. Duration: {self.t[-1]} s")
    
    def get_dist(self, t_i):
        return self.auto_dist[t_i]

    def get_origin(self, t_i):
        return self.beacons[self.ids[t_i]]

    def get_id(self, t_i):
        return self.ids[t_i]

    def get_data(self):
        return self.df
    
    def get_cir(self, t_i):
        return self.cir[t_i]
    
    def to_pickle(self, file_path: str):
        self.df.to_pickle(file_path)

    @property
    def origins(self):
        return self.beacons
