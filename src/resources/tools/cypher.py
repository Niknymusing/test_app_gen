"""
Tool that can be used to execute Cypher queries on a graph Neo4j database
"""

from langchain_core.tools import tool
from typing import Any, List, Dict
import json
from src.resources.vector_database_clients.neo4j_client import Neo4jClient
from langchain_core.embeddings.embeddings import Embeddings

_neo4j_client: Neo4jClient = None
_embedding_model = None


def set_neo4j_client_for_tools(client: Neo4jClient):
    """Helper to reuse the initialized Neo4j client from the FastAPI's dependency injection layer"""
    global _neo4j_client
    _neo4j_client = client


def set_embedding_model_for_tools(embedding_model: Embeddings):
    """Helper used to setup the initialized embedding model from the FastAPI's dependency injection layer"""
    global _embedding_model
    _embedding_model = embedding_model


@tool("execute_cypher_query", parse_docstring=True)
async def execute_cypher_query(cypher_query: str, user_query_text: str = None) -> str:
    """
    Executes a Cypher query on the Neo4j database and returns the results as JSON.

    Args:
        cypher_query: Cypher query string, may include an embedding placeholder `$embedding`.
        user_query_text: Optional text to embed and replace `$embedding` for semantic queries.

    Returns:
        JSON string of query results or an error message.

    Raises:
        ValueError: If cypher_query is empty.
    """
    global _neo4j_client, _embedding_model

    if _neo4j_client is None:
        return json.dumps({"error": "Neo4j client not initialized"})

    if not cypher_query.strip():
        raise ValueError("No Cypher query provided.")

    try:
        # If user_query_text is provided, embed it and inject into Cypher query
        if user_query_text:
            # Embed the user query text
            embedding_vector = await _embedding_model.aembed_query(text=user_query_text)
            # Format embedding vector as Cypher list string
            embedding_str = "[" + ",".join(f"{x:.8f}" for x in embedding_vector) + "]"
            # Replace placeholder (e.g. $embedding) in the cypher_query with actual embedding string
            cypher_query = cypher_query.replace("$embedding", embedding_str)

        # Execute the Cypher query once
        results: List[Dict[str, Any]] = _neo4j_client.graph_db.query(cypher_query)
        return json.dumps(results, indent=2, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": f"Failed to execute query: {str(e)}"})
