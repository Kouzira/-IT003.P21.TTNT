import joblib
import pandas as pd

class ETAPredictor:
    def __init__(self, model_path="model/traffic_model.pkl"):
        self.status_to_speed = {0: 60, 1: 50, 2: 40, 3: 30, 4: 20, 5: 10}
        data = joblib.load(model_path)
        self.model = data["model"]
        self.label_encoder = data["label_encoder"]
        self.street_encoder = data["street_encoder"]

    def kmh_to_mps(self, kmh):
        return kmh * 1000 / 3600

    def predict_eta(self, path, G, segments_df, streets_df,
                    hour, dayofweek, is_weekend, is_peak_hour):
        total_sec = 0
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            if not G.has_edge(u, v):
                continue

            segment_id = G[u][v]["segment_id"]
            segment_info = segments_df[segments_df["_id"] == segment_id].iloc[0]
            street_info = streets_df[streets_df["_id"] == segment_info["street_id"]].iloc[0]
            street_type = street_info["type"] if pd.notnull(street_info["type"]) else "unknown"

            if street_type in self.street_encoder.classes_:
                street_type_encoded = self.street_encoder.transform([street_type])[0]
            else:
                street_type_encoded = 0

            features = pd.DataFrame([[
                hour, dayofweek, int(is_weekend), int(is_peak_hour),
                segment_info["length"], segment_info["max_velocity"], street_type_encoded
            ]], columns=[
                "hour", "dayofweek", "is_weekend", "is_peak_hour",
                "length", "max_velocity", "street_type_encoded"
            ])

            status_pred = self.model.predict(features)[0]
            speed_mps = self.kmh_to_mps(self.status_to_speed.get(status_pred, 10))
            total_sec += segment_info["length"] / speed_mps

        return total_sec
