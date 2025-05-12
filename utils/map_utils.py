import folium
from streamlit_folium import folium_static

def visualize_path(path, node_coords):
    coordinates = [node_coords[n] for n in path if n in node_coords]

    if not coordinates:
        return None

    m = folium.Map(location=coordinates[0], zoom_start=15)
    folium.PolyLine(locations=coordinates, color='blue', weight=5).add_to(m)
    folium.Marker(location=coordinates[0], popup="Start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker(location=coordinates[-1], popup="End", icon=folium.Icon(color="red")).add_to(m)

    return m
