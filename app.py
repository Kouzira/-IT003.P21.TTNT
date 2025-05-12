import streamlit as st
from utils.data_loader import DataLoader
from utils.graph_utils import GraphBuilder
from utils.map_utils import visualize_path
from model.predictor import ETAPredictor

# Load dữ liệu
loader = DataLoader("data")
nodes_df, segments_df, streets_df, _, _ = loader.load_all()
node_coords = {row['_id']: (row['lat'], row['long']) for _, row in nodes_df.iterrows()}

# Tạo đồ thị và model
graph = GraphBuilder(segments_df)
predictor = ETAPredictor()

# Giao diện người dùng
st.title("Dự đoán thời gian di chuyển (ETA)")

start_node = st.number_input("Node bắt đầu", value=373543511)
end_node = st.number_input("Node kết thúc", value=696860119)

hour = st.slider("Giờ", 0, 23, 8)
dayofweek = st.selectbox("Ngày trong tuần", list(range(7)), format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x])
is_weekend = st.checkbox("Cuối tuần", value=False)
is_peak_hour = st.checkbox("Giờ cao điểm", value=True)

if st.button("Tìm đường và dự đoán ETA"):
    path, dist = graph.find_shortest_path(start_node, end_node)
    if path:
        st.success(f"Tìm thấy đường đi ({len(path)} node), độ dài {dist:.2f}m")

        eta_sec = predictor.predict_eta(path, graph.G, segments_df, streets_df,
                                        hour, dayofweek, is_weekend, is_peak_hour)
        st.info(f"ETA dự đoán: {eta_sec/60:.2f} phút")

        # Hiển thị bản đồ
        m = visualize_path(path, node_coords)
        if m:
            from streamlit_folium import folium_static
            folium_static(m)
    else:
        st.error("Không tìm được đường đi.")
