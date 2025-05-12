from utils.data_loader import DataLoader
from model.trainer import TrafficModelTrainer

loader = DataLoader("data")
_, _, _, _, train_df = loader.load_all()

trainer = TrafficModelTrainer(train_df)
trainer.train()
trainer.save("model/traffic_model.pkl")
print("Model đã được lưu.")