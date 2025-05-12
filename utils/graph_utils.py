import networkx as nx

class GraphBuilder:
    def __init__(self, segments_df):
        self.G = nx.DiGraph()
        self.segments_df = segments_df
        self.build_graph()

    def build_graph(self):
        for _, row in self.segments_df.iterrows():
            self.G.add_edge(row["s_node_id"], row["e_node_id"],
                            weight=row["length"],
                            segment_id=row["_id"])

    def find_shortest_path(self, start_node, end_node):
        try:
            path = nx.dijkstra_path(self.G, source=start_node, target=end_node, weight="weight")
            distance = nx.dijkstra_path_length(self.G, source=start_node, target=end_node, weight="weight")
            return path, distance
        except nx.NetworkXNoPath:
            return None, float("inf")
