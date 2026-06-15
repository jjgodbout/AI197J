import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
from context.knowledge_graph import KnowledgeGraphManager

# Initialize the KnowledgeGraphManager
kg_manager = KnowledgeGraphManager()

import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
from context.knowledge_graph import KnowledgeGraphManager

# Initialize the KnowledgeGraphManager
kg_manager = KnowledgeGraphManager()


def filter_graph_data(nodes, edges, node_search, selected_node_ids, selected_edge_labels):
    """Filter nodes and edges based on search and selections."""
    # Filter nodes
    filtered_nodes = []
    for node in nodes:
        if (node_search.lower() in node["node_label"].lower() or
            node_search.lower() in node["node_id"].lower() or
            not node_search) and (
                node["node_id"] in selected_node_ids or not selected_node_ids):
            filtered_nodes.append(node)

    # Get set of filtered node IDs for edge filtering
    filtered_node_ids = {node["node_id"] for node in filtered_nodes}

    # Filter edges
    filtered_edges = []
    for edge in edges:
        if (edge["source_node"] in filtered_node_ids and
                edge["target_node"] in filtered_node_ids and
                (edge["edge_label"] in selected_edge_labels or not selected_edge_labels)):
            filtered_edges.append(edge)

    return filtered_nodes, filtered_edges

def render_graph_management_tab():
    st.header("Manage Knowledge Graphs")

    # Retrieve user email from session
    user_email = st.session_state.get("username")
    if not user_email:
        st.error("Please log in to manage graphs.")
        return

    # Fetch user's knowledge graphs
    user_graphs = kg_manager.get_user_graphs(user_email)
    graph_options = {graph["graph_name"]: graph["graph_id"] for graph in user_graphs}

    # Layout: Three Columns
    col1, col2, col3 = st.columns(3)

    # Column 1: Graph Selection, Creation, and Data Display
    with col1:
        st.subheader("Select or Create Graph")

        # Add Create New Graph section
        with st.expander("Create New Graph", expanded=False):
            new_graph_name = st.text_input("Graph Name", key="new_graph_name")
            if st.button("Create Graph"):
                if new_graph_name:
                    # Generate a unique graph ID
                    new_graph_id = f"graph_{hash(new_graph_name + user_email)}"
                    try:
                        kg_manager.create_knowledge_graph(new_graph_id, new_graph_name, user_email)
                        st.success(f"Graph '{new_graph_name}' created successfully!")
                        st.rerun()  # Refresh the page to show the new graph
                    except Exception as e:
                        st.error(f"Error creating graph: {str(e)}")
                else:
                    st.error("Please enter a graph name.")

        # Existing graph selection
        if not user_graphs:
            st.info("No graphs found. Create a new graph above.")
        else:
            selected_graph_name = st.selectbox("Select Knowledge Graph", options=list(graph_options.keys()))
            selected_graph_id = graph_options[selected_graph_name]

            with st.expander("Graph Data (JSON Format)", expanded=False):
                nodes = kg_manager.get_graph_nodes(selected_graph_id)
                edges = kg_manager.get_graph_edges(selected_graph_id)
                graph_data = {"nodes": nodes, "edges": edges}
                st.json(graph_data)

    # Column 2: Node Management
    with col2:
        st.subheader("Manage Nodes")
        with st.expander("Add Node", expanded=False):
            if user_graphs:
                node_label = st.text_input("Node Label", key="node_label")
                if st.button("Create Node"):
                    if node_label:
                        node_id = f"node_{hash(node_label)}"
                        kg_manager.add_node(node_id, node_label, selected_graph_id)
                        st.success(f"Node '{node_label}' added to '{selected_graph_name}'.")
                        st.rerun()
                    else:
                        st.error("Node Label is required.")

        with st.expander("Delete Node", expanded=False):
            if user_graphs:
                node_id_to_delete = st.text_input("Node ID to delete", key="delete_node")
                if st.button("Delete Node", key="delete_node_btn"):
                    kg_manager.delete_node(node_id_to_delete, selected_graph_id)
                    st.success(f"Node '{node_id_to_delete}' deleted.")
                    st.rerun()

    # Column 3: Edge Management
    with col3:
        st.subheader("Manage Edges")
        with st.expander("Add Multiple Edges", expanded=False):
            if user_graphs:
                # Populate dropdowns for source and target nodes
                node_data = kg_manager.get_graph_nodes(selected_graph_id)
                label_to_id = {node["node_label"]: node["node_id"] for node in node_data}

                source_nodes = st.multiselect(
                    "Source Nodes", options=list(label_to_id.keys()), key="source_node_dropdown"
                )
                target_node = st.selectbox(
                    "Target Node", options=list(label_to_id.keys()), key="target_node_dropdown"
                )
                edge_label = st.text_input("Edge Label", key="edge_label")

                if st.button("Add Edges", key="add_edges_btn"):
                    if source_nodes and target_node and edge_label:
                        edges = [
                            {
                                "edge_id": f"edge_{hash(label_to_id[source] + label_to_id[target_node])}",
                                "source_node": label_to_id[source],
                                "target_node": label_to_id[target_node],
                                "edge_label": edge_label,
                            }
                            for source in source_nodes
                        ]
                        kg_manager.add_edges(edges, selected_graph_id)
                        st.success(
                            f"Edges added from '{', '.join(source_nodes)}' to '{target_node}' with label '{edge_label}'."
                        )
                        st.rerun()
                    else:
                        st.error("Please select at least one source node, one target node, and provide an edge label.")

        with st.expander("Delete Edge", expanded=False):
            if user_graphs:
                edge_id_to_delete = st.text_input("Edge ID to delete", key="delete_edge")
                if st.button("Delete Edge", key="delete_edge_btn"):
                    kg_manager.delete_edge(edge_id_to_delete)
                    st.success(f"Edge '{edge_id_to_delete}' deleted.")
                    st.rerun()

    # Visualization Section with Filters
    st.subheader("Visualize Graph")
    if user_graphs and selected_graph_id:
        # Create filter sidebar
        st.sidebar.header("Graph Filters")

        # Node search filter
        node_search = st.sidebar.text_input("Search Nodes",
                                            placeholder="Enter node label or ID...")

        # Node selection filter
        all_node_labels = {node["node_label"]: node["node_id"] for node in nodes}
        selected_node_labels = st.sidebar.multiselect(
            "Filter Nodes",
            options=list(all_node_labels.keys()),
            default=list(all_node_labels.keys())
        )
        selected_node_ids = [all_node_labels[label] for label in selected_node_labels]

        # Edge type filter
        all_edge_labels = list(set(edge["edge_label"] for edge in edges))
        selected_edge_labels = st.sidebar.multiselect(
            "Filter Edge Types",
            options=all_edge_labels,
            default=all_edge_labels
        )

        # Apply filters
        filtered_nodes, filtered_edges = filter_graph_data(
            nodes, edges, node_search, selected_node_ids, selected_edge_labels
        )

        # Display filter statistics
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**Visible Nodes:** {len(filtered_nodes)}/{len(nodes)}")
        st.sidebar.markdown(f"**Visible Edges:** {len(filtered_edges)}/{len(edges)}")

        # Create graph visualization with filtered data
        graph_nodes = [
            Node(id=node["node_id"],
                 label=node["node_label"],
                 size=25)  # You can customize node appearance here
            for node in filtered_nodes
        ]

        graph_edges = [
            Edge(source=edge["source_node"],
                 target=edge["target_node"],
                 label=edge["edge_label"])
            for edge in filtered_edges
        ]

        # Configure graph appearance
        config = Config(
            height=700,
            width=1400,
            directed=True,
            physics=True,
            hierarchical=False,
            nodeHighlightBehavior=True,
            highlightColor="#F7A7A6",
        )

        # Render the graph
        agraph(nodes=graph_nodes, edges=graph_edges, config=config)
    else:
        st.info("Select a graph to visualize or create a new one.")
