from langchain_qdrant import QdrantVectorStore
from langchain_groq import ChatGroq
from langgraph.graph import MessagesState

from typing_extensions import List

class GraphState(MessagesState):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        final_answer: LLM generation
        recommend_docs: list of documents
    """
    groq_llama3_1_70b: ChatGroq
    product_qdrant: QdrantVectorStore
    policy_qdrant: QdrantVectorStore
    query: str
    final_answer: str
    recommend_products: List[str]
    supervisor_route_choice: str
    cust_id: str