from typing import List, Dict, Optional
import uuid
import json
from context.analyzer import TextAnalyzer
from context.knowledge_graph import KnowledgeGraphManager
from utils.query_handler import execute_sql


class DocumentGraphAnalyzer:
    def __init__(self, page_group_size: int = 10, db_type: str = "snowflake"):
        """
        Initialize DocumentGraphAnalyzer with TextAnalyzer and KnowledgeGraphManager

        Args:
            page_group_size: Size of page groups for text analysis
            db_type: Database type for knowledge graph storage
        """
        self.text_analyzer = TextAnalyzer(page_group_size=page_group_size)
        self.graph_manager = KnowledgeGraphManager(db_type=db_type)

    def _generate_id(self) -> str:
        """Generate a unique identifier"""
        return str(uuid.uuid4())

    def _parse_completion_result(self, completion_result: Dict) -> Dict:
        """
        Parse the completion result into structured node and edge data

        Args:
            completion_result: Raw completion result from text analysis

        Returns:
            Dict containing parsed nodes and edges
        """
        try:
            parts = completion_result.get('analysis', {}).get('parts', [])
            nodes = []
            edges = []

            for part in parts:
                result = part.get('result')
                if isinstance(result, str):
                    try:
                        parsed = json.loads(result)
                        if isinstance(parsed, dict):
                            if 'nodes' in parsed:
                                nodes.extend(parsed['nodes'])
                            if 'edges' in parsed:
                                edges.extend(parsed['edges'])
                    except json.JSONDecodeError:
                        continue

            return {'nodes': nodes, 'edges': edges}
        except Exception as e:
            print(f"Error parsing completion result: {str(e)}")
            return {'nodes': [], 'edges': []}

    def create_document_graph(
            self,
            doc_id: str,
            graph_name: str,
            user_email: str,
            node_types: List[str],
            edge_types: List[str],
            custom_prompts: Optional[Dict[str, str]] = None
    ) -> Dict:
        """
        Create a knowledge graph from document analysis

        Args:
            doc_id: Document identifier
            graph_name: Name for the new knowledge graph
            user_email: User email for graph ownership
            node_types: List of node types to extract
            edge_types: List of edge types to extract
            custom_prompts: Optional custom prompts for specific node or edge types

        Returns:
            Dict containing graph creation results
        """
        try:
            # Generate graph ID
            graph_id = self._generate_id()

            # Create base graph
            self.graph_manager.create_knowledge_graph(graph_id, graph_name, user_email)

            # Construct analysis prompt
            base_prompt = f"""
            Analyze the following text and extract a knowledge graph with these components:

            Node types to identify: {', '.join(node_types)}
            Edge types to identify: {', '.join(edge_types)}

            For each section, return a JSON object with:
            - 'nodes': list of identified entities with their types
            - 'edges': list of relationships between entities

            Format:
            {{
                "nodes": [
                    {{"id": "unique_id", "label": "entity_name", "type": "node_type"}}
                ],
                "edges": [
                    {{"source": "source_node_id", "target": "target_node_id", "type": "edge_type"}}
                ]
            }}
            """

            # Add any custom prompts
            if custom_prompts:
                for node_type, prompt in custom_prompts.items():
                    base_prompt += f"\nFor {node_type}, specifically: {prompt}"

            # Perform text analysis
            completion_result = self.text_analyzer.complete_analysis(
                doc_id=doc_id,
                prompt=base_prompt,
                model='mixtral-8x7b'
            )

            # Parse results
            parsed_results = self._parse_completion_result(completion_result)

            # Add nodes and edges to graph
            for node in parsed_results['nodes']:
                node_id = node.get('id', self._generate_id())
                self.graph_manager.add_node(
                    node_id=node_id,
                    node_label=f"{node.get('label')} [{node.get('type')}]",
                    graph_id=graph_id
                )

            # Process edges
            edges_to_add = []
            for edge in parsed_results['edges']:
                edge_id = self._generate_id()
                edges_to_add.append({
                    'edge_id': edge_id,
                    'source_node': edge['source'],
                    'target_node': edge['target'],
                    'edge_label': edge['type']
                })

            if edges_to_add:
                self.graph_manager.add_edges(edges_to_add, graph_id)

            # Return creation results
            return {
                'status': 'success',
                'graph_id': graph_id,
                'graph_name': graph_name,
                'node_count': len(parsed_results['nodes']),
                'edge_count': len(parsed_results['edges']),
                'document_id': doc_id
            }

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'graph_id': None
            }

    def enhance_existing_graph(
            self,
            doc_id: str,
            graph_id: str,
            node_types: List[str],
            edge_types: List[str],
            custom_prompts: Optional[Dict[str, str]] = None
    ) -> Dict:
        """
        Enhance an existing knowledge graph with additional document analysis

        Args:
            doc_id: Document identifier
            graph_id: Existing graph identifier
            node_types: Additional node types to extract
            edge_types: Additional edge types to extract
            custom_prompts: Optional custom prompts for specific types

        Returns:
            Dict containing enhancement results
        """
        try:
            # Get existing nodes
            existing_nodes = self.graph_manager.get_graph_nodes(graph_id)
            existing_node_ids = {node['node_id'] for node in existing_nodes}

            # Create analysis prompt that considers existing nodes
            enhance_prompt = f"""
            Analyze the following text and enhance the existing knowledge graph.

            Additional node types to identify: {', '.join(node_types)}
            Additional edge types to identify: {', '.join(edge_types)}

            Existing nodes: {[node['node_label'] for node in existing_nodes]}

            Return JSON with new nodes and edges, including connections to existing nodes where relevant.
            """

            if custom_prompts:
                for node_type, prompt in custom_prompts.items():
                    enhance_prompt += f"\nFor {node_type}, specifically: {prompt}"

            # Perform analysis
            completion_result = self.text_analyzer.complete_analysis(
                doc_id=doc_id,
                prompt=enhance_prompt,
                model='mixtral-8x7b'
            )

            # Parse and add new elements
            parsed_results = self._parse_completion_result(completion_result)

            # Add new nodes
            new_nodes = []
            for node in parsed_results['nodes']:
                node_id = node.get('id', self._generate_id())
                if node_id not in existing_node_ids:
                    self.graph_manager.add_node(
                        node_id=node_id,
                        node_label=f"{node.get('label')} [{node.get('type')}]",
                        graph_id=graph_id
                    )
                    new_nodes.append(node)

            # Add new edges
            new_edges = []
            edges_to_add = []
            for edge in parsed_results['edges']:
                edge_id = self._generate_id()
                edges_to_add.append({
                    'edge_id': edge_id,
                    'source_node': edge['source'],
                    'target_node': edge['target'],
                    'edge_label': edge['type']
                })
                new_edges.append(edge)

            if edges_to_add:
                self.graph_manager.add_edges(edges_to_add, graph_id)

            return {
                'status': 'success',
                'graph_id': graph_id,
                'new_nodes_added': len(new_nodes),
                'new_edges_added': len(new_edges),
                'document_id': doc_id
            }

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'graph_id': graph_id
            }


class KnowledgeGraphEdgeDefinitions:
    """
    Defines edge types and their properties for a knowledge graph connecting
    Person, Location, Organization, Food, and Activity nodes
    """

    @staticmethod
    def get_edge_definitions() -> Dict[str, Dict[str, List[Dict[str, str]]]]:
        return {
            "Person": {
                "Person": [
                    {"type": "knows", "description": "Personal or professional acquaintance"},
                    {"type": "related_to", "description": "Family or genetic relationship"},
                    {"type": "collaborates_with", "description": "Works together on projects/activities"},
                    {"type": "reports_to", "description": "Professional reporting relationship"}
                ],
                "Location": [
                    {"type": "lives_in", "description": "Primary residence"},
                    {"type": "visited", "description": "Temporary visit or stay"},
                    {"type": "born_in", "description": "Place of birth"},
                    {"type": "worked_in", "description": "Professional location history"}
                ],
                "Organization": [
                    {"type": "works_for", "description": "Employment relationship"},
                    {"type": "founded", "description": "Created or established"},
                    {"type": "member_of", "description": "Membership or affiliation"},
                    {"type": "invested_in", "description": "Financial investment relationship"}
                ],
                "Food": [
                    {"type": "likes", "description": "Food preference"},
                    {"type": "allergic_to", "description": "Food allergy"},
                    {"type": "prepares", "description": "Cooking or preparation relationship"},
                    {"type": "sells", "description": "Commercial food relationship"}
                ],
                "Activity": [
                    {"type": "participates_in", "description": "Active involvement"},
                    {"type": "organizes", "description": "Leadership or organization role"},
                    {"type": "teaches", "description": "Instructional relationship"},
                    {"type": "interested_in", "description": "Personal interest or hobby"}
                ]
            },
            "Location": {
                "Location": [
                    {"type": "contains", "description": "Geographic containment"},
                    {"type": "adjacent_to", "description": "Physical proximity"},
                    {"type": "connected_to", "description": "Transportation or access link"},
                    {"type": "part_of", "description": "Administrative or geographic hierarchy"}
                ],
                "Organization": [
                    {"type": "hosts", "description": "Physical location relationship"},
                    {"type": "supplies", "description": "Resource or service provision"},
                    {"type": "regulates", "description": "Jurisdictional authority"}
                ],
                "Food": [
                    {"type": "produces", "description": "Food production location"},
                    {"type": "known_for", "description": "Cultural or traditional food association"},
                    {"type": "exports", "description": "Food trade relationship"}
                ],
                "Activity": [
                    {"type": "suitable_for", "description": "Activity compatibility"},
                    {"type": "hosts_activity", "description": "Regular activity venue"},
                    {"type": "restricts", "description": "Activity limitations or regulations"}
                ]
            },
            "Organization": {
                "Organization": [
                    {"type": "partners_with", "description": "Business partnership"},
                    {"type": "competes_with", "description": "Market competition"},
                    {"type": "subsidiary_of", "description": "Corporate ownership"},
                    {"type": "collaborates_with", "description": "Joint venture or project"}
                ],
                "Food": [
                    {"type": "produces", "description": "Food manufacturing"},
                    {"type": "distributes", "description": "Food distribution"},
                    {"type": "certifies", "description": "Food safety/quality certification"},
                    {"type": "researches", "description": "Food research and development"}
                ],
                "Activity": [
                    {"type": "sponsors", "description": "Financial or resource support"},
                    {"type": "organizes", "description": "Event or activity management"},
                    {"type": "promotes", "description": "Marketing or advocacy"},
                    {"type": "regulates", "description": "Activity oversight"}
                ]
            },
            "Food": {
                "Food": [
                    {"type": "pairs_with", "description": "Culinary pairing"},
                    {"type": "contains", "description": "Ingredient relationship"},
                    {"type": "substitutes", "description": "Alternative ingredient"},
                    {"type": "complements", "description": "Enhancement relationship"}
                ],
                "Activity": [
                    {"type": "required_for", "description": "Essential food for activity"},
                    {"type": "enhanced_by", "description": "Performance enhancement"},
                    {"type": "featured_in", "description": "Activity-specific food role"}
                ]
            },
            "Activity": {
                "Activity": [
                    {"type": "prerequisite_for", "description": "Required prior activity"},
                    {"type": "part_of", "description": "Component relationship"},
                    {"type": "leads_to", "description": "Sequential relationship"},
                    {"type": "conflicts_with", "description": "Incompatible activities"}
                ]
            }
        }

    @staticmethod
    def get_edge_types() -> List[str]:
        """
        Returns a flat list of all unique edge types
        """
        edges = set()
        definitions = KnowledgeGraphEdgeDefinitions.get_edge_definitions()

        for source in definitions:
            for target in definitions[source]:
                for edge in definitions[source][target]:
                    edges.add(edge["type"])

        return sorted(list(edges))

    @staticmethod
    def get_edges_for_nodes(source_type: str, target_type: str) -> List[Dict[str, str]]:
        """
        Returns valid edge types between specific node types

        Args:
            source_type: Type of the source node
            target_type: Type of the target node

        Returns:
            List of valid edge definitions between the node types
        """
        definitions = KnowledgeGraphEdgeDefinitions.get_edge_definitions()

        # Check direct relationship
        if source_type in definitions and target_type in definitions[source_type]:
            return definitions[source_type][target_type]

        # Check inverse relationship
        if target_type in definitions and source_type in definitions[target_type]:
            # Return inverse relationships
            inverse_edges = []
            for edge in definitions[target_type][source_type]:
                inverse_type = f"inverse_{edge['type']}"
                inverse_desc = f"Inverse of: {edge['description']}"
                inverse_edges.append({"type": inverse_type, "description": inverse_desc})
            return inverse_edges

        return []
