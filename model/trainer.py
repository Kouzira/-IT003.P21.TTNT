import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

class TrafficModelTrainer:
    def __init__(self, train_df):
        self.train_df = train_df
        self.label_encoder = LabelEncoder()
        self.street_encoder = LabelEncoder()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def preprocess(self):
        df = self.train_df.copy()
        df["is_weekend"] = df["weekday"] >= 5

        df["street_type"] = df["street_type"].fillna("unknown")
        df["street_type_encoded"] = self.street_encoder.fit_transform(df["street_type"])
        df["LOS"] = self.label_encoder.fit_transform(df["LOS"])

        self.train_df = df

    def train(self):
        self.preprocess()
        X = self.train_df[[
            "weekday", "is_weekend", "length", "max_velocity", "street_type_encoded"
        ]]
        y = self.train_df["LOS"]
        self.model.fit(X, y)

    def save(self, path="model/traffic_model.pkl"):
        joblib.dump({
            "model": self.model,
            "label_encoder": self.label_encoder,
            "street_encoder": self.street_encoder
        }, path)
