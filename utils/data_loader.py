import pandas as pd

class DataLoader:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir

    def load_all(self):
        nodes = pd.read_csv(f"{self.data_dir}/nodes.csv")
        segments = pd.read_csv(f"{self.data_dir}/segments.csv")
        streets = pd.read_csv(f"{self.data_dir}/streets.csv")
        statuses = pd.read_csv(f"{self.data_dir}/segment_status.csv")
        train = pd.read_csv(f"{self.data_dir}/train.csv")
        return nodes, segments, streets, statuses, train
