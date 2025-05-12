import streamlit as st
import pandas as pd
import networkx as nx
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# === Load dữ liệu ===
nodes_df = pd.read_csv("data/nodes.csv")
segments_df = pd.read_csv("data/segments.csv")
streets_df = pd.read_csv("data/streets.csv")
train_df = pd.read_csv("data/train.csv")

# Tiền xử lý
train_df["timestamp"] = pd.to_datetime(train_df["date"])
train_df["hour"] = train_df["timestamp"].dt.hour
train_df["dayofweek"] = train_df["timestamp"].dt.weekday
train_df["is_weekend"] = train_df["dayofweek"] >= 5
train_df["is_peak_hour"] = train_df["hour"].apply(lambda h: 7 <= h <= 9 or 16 <= h <= 19)
train_df["street_type"] = train_df["street_type"].fillna("unknown")

# Mã hóa street_type
le = LabelEncoder()
train_df["street_type_encoded"] = le.fit_transform(train_df["street_type"])

# Huấn luyện model
feature_names = [
    "hour", "dayofweek", "is_weekend", "is_peak_hour",
    "length", "max_velocity", "street_type_encoded"
]

X = train_df[feature_names]
y = train_df["LOS"]
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Tạo đồ thị ===
G = nx.DiGraph()
for _, row in segments_df.iterrows():
    G.add_edge(row["s_node_id"], row["e_node_id"], weight=row["length"], segment_id=row["_id"])

# Node -> Tọa độ
node_coords = {
    row['_id']: (row['lat'], row['long'])
    for _, row in nodes_df.iterrows()
}

status_to_speed = {
    0: 60, 1: 50, 2: 40, 3: 30, 4: 20, 5: 10
}

def kmh_to_mps(kmh):
    return kmh * 1000 / 3600

# Tìm đường đi ngắn nhất
def find_shortest_path(start_node, end_node):
    try:
        path = nx.dijkstra_path(G, source=start_node, target=end_node, weight="weight")
        total_distance = nx.dijkstra_path_length(G, source=start_node, target=end_node, weight="weight")
        return path, total_distance
    except nx.NetworkXNoPath:
        return None, float("inf")

# Ước lượng ETA
def estimate_eta(path, hour, dayofweek, is_weekend, is_peak_hour):
    total_time_sec = 0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        if G.has_edge(u, v):
            segment_id = G[u][v]['segment_id']
            segment_info = segments_df[segments_df['_id'] == segment_id].iloc[0]
            street_info = streets_df[streets_df['_id'] == segment_info['street_id']].iloc[0]

            street_type = street_info["type"] if pd.notnull(street_info["type"]) else "unknown"
            if street_type in le.classes_:
                street_type_encoded = le.transform([street_type])[0]
            else:
                street_type_encoded = 0

            features_df = pd.DataFrame([[hour, dayofweek, int(is_weekend), int(is_peak_hour),
                                         segment_info["length"], segment_info["max_velocity"],
                                         street_type_encoded]], columns=feature_names)

            predicted_status = model.predict(features_df)[0]
            predicted_speed_kmh = status_to_speed.get(predicted_status, 10)
            predicted_speed_mps = kmh_to_mps(predicted_speed_kmh)

            time_sec = segment_info["length"] / predicted_speed_mps
            total_time_sec += time_sec
    return total_time_sec

# Vẽ đường đi
def visualize_path(path):
    coords = [node_coords[n] for n in path if n in node_coords]
    if not coords:
        return None
    start_lat, start_lon = coords[0]
    m = folium.Map(location=[start_lat, start_lon], zoom_start=15)
    folium.PolyLine(locations=coords, color='blue', weight=5).add_to(m)
    folium.Marker(location=coords[0], popup="Start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker(location=coords[-1], popup="End", icon=folium.Icon(color="red")).add_to(m)
    return m

# === Giao diện Streamlit ===
st.title("🚦 Ước lượng ETA và Đường đi ngắn nhất")

start_node = st.number_input("Nhập ID Node bắt đầu:", value=373543511)
end_node = st.number_input("Nhập ID Node kết thúc:", value=696860119)
hour = st.slider("Giờ trong ngày (0-23):", 0, 23, 8)
dayofweek = st.selectbox("Thứ trong tuần (0=Thứ 2, 6=Chủ Nhật):", list(range(7)), index=1)
is_weekend = st.checkbox("Là cuối tuần?", value=False)
is_peak_hour = st.checkbox("Là giờ cao điểm?", value=True)

if st.button("🚗 Tính đường đi và ETA"):
    path, distance = find_shortest_path(start_node, end_node)
    if path:
        st.success(f"Đã tìm thấy đường đi {len(path)} điểm với tổng chiều dài {distance:.2f} mét")
        eta_sec = estimate_eta(path, hour, dayofweek, is_weekend, is_peak_hour)
        st.info(f"ETA ước tính: {eta_sec / 60:.2f} phút")
        st_folium(visualize_path(path), width=800, height=500)
    else:
        st.error("Không tìm được đường đi giữa hai node.")
