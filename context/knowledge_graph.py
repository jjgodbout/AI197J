from utils.query_handler import execute_sql

class KnowledgeGraphManager:
    def __init__(self, db_type="snowflake"):
        self.db_type = db_type

    def create_knowledge_graph(self, graph_id, graph_name, user_email):
        query = f"""
        INSERT INTO colby.ai197J.knowledge_graphs (graph_id, graph_name, user_email)
        VALUES ('{graph_id}', '{graph_name}', '{user_email}')
        """
        execute_sql(query, self.db_type)

    def add_node(self, node_id, node_label, graph_id):
        query = f"""
        INSERT INTO colby.ai197J.nodes (node_id, node_label, graph_id)
        VALUES ('{node_id}', '{node_label}', '{graph_id}')
        """
        execute_sql(query, self.db_type)

    def add_edges(self, edges, graph_id):
        """
        Adds multiple edges to the graph.
        :param edges: List of dictionaries with edge_id, source_node, target_node, and edge_label.
        :param graph_id: The ID of the graph to which the edges belong.
        """
        values = ", ".join([
            f"('{edge['edge_id']}', '{edge['source_node']}', '{edge['target_node']}', '{edge['edge_label']}', '{graph_id}')"
            for edge in edges
        ])
        query = f"""
        INSERT INTO colby.ai197J.edges (edge_id, source_node, target_node, edge_label, graph_id)
        VALUES {values}
        """
        execute_sql(query, self.db_type)

    def get_user_graphs(self, user_email):
        query = f"""
        SELECT graph_id, graph_name FROM colby.ai197J.knowledge_graphs
        WHERE user_email = '{user_email}'
        """
        result = execute_sql(query, self.db_type)
        return [{"graph_id": row[0], "graph_name": row[1]} for row in result]

    def get_graph_nodes(self, graph_id):
        query = f"""
        SELECT node_id, node_label FROM colby.ai197J.nodes
        WHERE graph_id = '{graph_id}'
        """
        result = execute_sql(query, self.db_type)
        return [{"node_id": row[0], "node_label": row[1]} for row in result]

    def get_graph_edges(self, graph_id):
        query = f"""
        SELECT edge_id, source_node, target_node, edge_label FROM colby.ai197J.edges
        WHERE graph_id = '{graph_id}'
        """
        result = execute_sql(query, self.db_type)
        return [
            {"edge_id": row[0], "source_node": row[1], "target_node": row[2], "edge_label": row[3]}
            for row in result
        ]

    def delete_node(self, node_id, graph_id):
        query = f"""
        DELETE FROM colby.ai197J.nodes
        WHERE node_id = '{node_id}' AND graph_id = '{graph_id}'
        """
        execute_sql(query, self.db_type)

    def delete_edge(self, edge_id):
        query = f"""
        DELETE FROM colby.ai197J.edges
        WHERE edge_id = '{edge_id}'
        """
        execute_sql(query, self.db_type)
