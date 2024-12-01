from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from dotenv import load_dotenv
import os

from langchain_qdrant import QdrantVectorStore
from langchain_qdrant import FastEmbedSparse, RetrievalMode
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from flower_agents.flower_agents import FlowerAgents

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


sparse_embeddings = FastEmbedSparse(
    model_name="Qdrant/bm25"
)

model_name = "jinaai/jina-embeddings-v3"
model_kwargs = {'device': 'cuda', 'trust_remote_code':True}
encode_kwargs = {'normalize_embeddings': False}
jina_embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

embeddings = jina_embeddings

product_qdrant = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    sparse_embedding=sparse_embeddings,
    url=QDRANT_URL,
    prefer_grpc=True,
    api_key=QDRANT_API_KEY,
    collection_name="flowers",
    retrieval_mode=RetrievalMode.HYBRID,
)


policy_qdrant = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    sparse_embedding=sparse_embeddings,
    url=QDRANT_URL,
    prefer_grpc=True,
    api_key=QDRANT_API_KEY,
    collection_name="flowers_policy",
    retrieval_mode=RetrievalMode.HYBRID,
)

groq_llama3_1_70b = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0,
    api_key=GROQ_API_KEY
)

app = FastAPI()

# Define the Pydantic model
class QueryState(BaseModel):
    user_query: str
    cust_id: str


@app.post("/chatResponse")
async def chat_response(request: QueryState):
    # Access the incoming JSON
    user_query = request.user_query
    cust_id = request.cust_id
    state = {
        "query":user_query,
        "cust_id":cust_id,
        "product_qdrant":product_qdrant,
        "policy_qdrant":policy_qdrant,
        "groq_llama3_1_70b":groq_llama3_1_70b,
        "final_answer":"None",
        "recommend_products":"None",
        "supervisor_route_choice":"None",
    }

    flower_agents = FlowerAgents()

    # Construct a response
    result = flower_agents(state)
    
    # Return a JSON response
    return {
        "final_answer": result['final_answer'],
        "recommend_products": result['recommend_products'],
        "supervisor_route_choice": result['supervisor_route_choice']
    }





''' 
        cust_id = "C001"
        user_query = user_prompt
        state = {
            "query":user_query,
            "cust_id":cust_id,
            "product_qdrant":product_qdrant,
            "policy_qdrant":policy_qdrant,
            "groq_llama3_1_70b":groq_llama3_1_70b
        }

'''