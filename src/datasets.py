import pandas as pd
import os


def read_data_from_csv(path, sep=";"):
    return pd.read_csv(path, sep=sep)

def read_massspec_data(path):
    extension = os.path.splitext(path)[1].lower()
    if extension == ".txt":
        header = pd.read_table(path, sep='\t', nrows=1, header=None)
        colnames = header.loc[:, 1:].values[0].tolist()
        colnames = ['time'] + colnames
        df = pd.read_table(path, sep='\t', skiprows=[0, 1, 2, 3, 4], index_col=False,
                           header=None)
        df = df.loc[:, 0:15]
        df.columns = colnames

    elif extension == ".csv":
        df = pd.read_csv(path, sep=";")
    else:
        print("Unknown file type")
        df = None
    return df

def get_unique_sessions(drift_data):
    return drift_data.Session.unique()

def get_session_dict(drift_data):
    return drift_data.groupby("Session")["Label"].apply(list).to_dict()

class Datasets(object):
    def __init__(self):
        self.drift_data = None
        self.calibration_data = None
        self.data_file_names = None
        self.data_files = None
        self.unique_sessions = None
        self.session_dict = None
        self.labeled_datasets = None

    def read_drift_data(self, path):
        self.drift_data = read_data_from_csv(path)
        self.session_dict = get_session_dict(self.drift_data)
        self.unique_sessions = self.session_dict.keys()

    def get_drift_data(self):
        if self.drift_data is not None:
            return self.drift_data
        else:
            raise ValueError("Drift data not loaded")

    def read_calibration_data(self, path):
        self.calibration_data = read_data_from_csv(path)

    def get_calibration_data(self):
        if self.calibration_data is not None:
            return self.calibration_data
        else:
            raise ValueError("Calibration data not loaded")

    def read_datasets(self, path):
        self.data_file_names = os.listdir(path)
        self.data_file_names = [f for f in self.data_file_names if os.path.splitext(f)[1].lower() in [".txt", ".csv"]]
        self.data_files = {f.split(".")[0]:read_massspec_data(os.path.join(path, f)) for f in self.data_file_names}

    def get_datasets(self):
        return self.data_files

    def get_dataset(self, filename):
        return self.data_files[filename]

    def get_sessions(self):
        if self.unique_sessions is not None:
            return self.unique_sessions
        else:
            raise ValueError("Drift data not loaded")

    def get_session_dict(self):
        if self.session_dict is not None:
            return self.session_dict
        else:
            raise("Drift data not loaded")


    def get_labeled_datasets(self):
        return self.labeled_datasets

    def get_labeled_dataset(self, filename):
        return self.labeled_datasets[filename]