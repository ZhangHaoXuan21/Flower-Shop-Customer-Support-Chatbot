from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from typing import List

from typing import Literal
from pydantic import BaseModel, Field
from flower_agents.utils.database import add_item_to_cart_by_name


# Supervisor Node

def supervisor_agent(state):
    query = state['query']
    groq_llama3_1_70b = state['groq_llama3_1_70b']

    # Data model
    class SupervisorRoute(BaseModel):
        """Route a user query to the most relevant datasource."""

        route: Literal["product_recommendation_agent", "policy_agent", "cart_agent", "apology_agent"] = Field(
            description="Given a user question choose to route it to vector_store_agent or apology_agent. ",
        )


    structured_supervisor_llm = groq_llama3_1_70b.with_structured_output(SupervisorRoute)

    # Prompt
    system = """You are an expert at delegating a user query to an appropriate agent.
    if the user query involved product recommendation:
        delegate to "product_recommendation_agent"
    if the user query involve policy question:
        delegate to "policy_agent"
    if the user query involve adding items to cart:
        delegate to "cart_agent"
    if the user query are not related to product recommendation, policy, cart:
        delegate to "apology_agent"
    """
    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )

    supervisor_agent = route_prompt | structured_supervisor_llm

    supervisor_choice = supervisor_agent.invoke(query)

    return {"supervisor_route_choice": supervisor_choice.route}

# Recommendation Agent
def product_recommendation_agent(state):
    query = state['query']
    product_qdrant = state['product_qdrant']
    groq_llama3_1_70b = state['groq_llama3_1_70b']

    # -----------------------------------------------------------------
    # 1. Retrieval
    # -----------------------------------------------------------------

    compressor = FlashrankRerank(
        model="ms-marco-MiniLM-L-12-v2",
        top_n=5
    )

    product_hybrid_rerank_qdrant_retriever = product_qdrant.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": 20},
    )
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=product_hybrid_rerank_qdrant_retriever
    )

    recommend_docs = compression_retriever.invoke(query)

    # Format the docs
    recommend_docs_context = "\n\n".join(doc.page_content for doc in recommend_docs)

    # -----------------------------------------------------------------
    # 2. Recommendation Text Generation
    # -----------------------------------------------------------------
    # Prompt
    recommend_template = """
    You are a persuasive product recommendation assistant.  
    Your goal is to recommend products in a compelling and confident manner based on the context and user’s needs.  
    Use strong, persuasive language to highlight why this product is the perfect choice.  
    If the context does not provide enough information to recommend a product, suggest reaching out for personalized assistance.  

    Provide the recommendation in the following structured format:  
    1.**Product Name**: [Insert product name]  
    **Reason to Buy**: [Provide a persuasive and compelling reason why this product is the best choice for the user]

    User Need: {user_need}  
    Context: {context}  
    Answer:

    """

    recommend_prompt = ChatPromptTemplate.from_template(recommend_template)


    llm = groq_llama3_1_70b

    rag_chain = (
        recommend_prompt
        | llm
        | StrOutputParser()
    )

    recommendation_text =  rag_chain.invoke({"user_need": query, "context": recommend_docs_context})

    # -----------------------------------------------------------------
    # 3. Extract Product Names
    # -----------------------------------------------------------------

    # Data model
    class ProductNames(BaseModel):
        """Route a user query to the most relevant datasource."""

        names: List[str] = Field(
            description="Product names from the recommendation text ",
        )

    product_names_template = """
    Extract the Product Names from Recommendation Text.
    Recommendation Text: {recommendation_text}  
    Product Names:
    """

    product_names_llm = groq_llama3_1_70b.with_structured_output(ProductNames)

    product_name_prompt = ChatPromptTemplate.from_template(product_names_template)

    product_name_chain = product_name_prompt | product_names_llm

    product_names = product_name_chain.invoke({'recommendation_text': recommendation_text})

    return {'final_answer': recommendation_text, "recommend_products":product_names.names}

def cart_agent(state):
    query = state['query']
    cust_id = state['cust_id']
    groq_llama3_1_70b = state['groq_llama3_1_70b']

    class CartItem(BaseModel):
        """Represents an item in the shopping cart."""
        product_name: str
        product_quantity: int

    class CartData(BaseModel):
        """Extract useful cart data from user query."""
        cart_data: List[CartItem]
        complete_data: Literal['yes', 'no']
        reason: str

    structured_cart_llm = groq_llama3_1_70b.with_structured_output(CartData)

    # Prompt
    system = """You are a highly intelligent assistant trained to extract structured data from user input.  
    Your task is to extract a list of products and their quantities in the user's shopping cart from the query provided.  
    Ensure the data follows the required structure exactly as described.  
    If no cart data is mentioned, respond with an empty list.

    class CartItem(BaseModel):
        product_name: str
        product_quantity: int

    class CartData(BaseModel):
        cart_data: List[CartItem]

    ### User Query:
    {user_query}

    ### Extracted Cart Data:

    """
    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{user_query}"),
        ]
    )

    cart_agent = route_prompt | structured_cart_llm

    cart_items = cart_agent.invoke(query)

    process_message = ""

    if cart_items.complete_data == "yes":
        for cart_item in cart_items.cart_data:
            process_data = add_item_to_cart_by_name(cust_id, cart_item.product_name, cart_item.product_quantity)
            process_message += f"{process_data['process_message']}\n"

    else:
        process_message = cart_items.reason
    
    return {'final_answer': process_message}

# Policy Agent
def policy_agent(state):
    query = state['query']
    policy_qdrant = state['policy_qdrant']
    groq_llama3_1_70b = state['groq_llama3_1_70b']

    # -----------------------------------------------------------------
    # 1. Retrieval
    # -----------------------------------------------------------------
    policy_hybrid_rerank_qdrant_retriever = policy_qdrant.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": 5},
    )

    policy_docs = policy_hybrid_rerank_qdrant_retriever.invoke(query)

    # Format the docs
    policy_docs_context = "\n\n".join(doc.page_content for doc in policy_docs)

    # -----------------------------------------------------------------
    # 2. Generation
    # -----------------------------------------------------------------
    # Prompt

    policy_template = """
    You are a policy assistant.  
    Your role is to provide clear, confident, and informative answers based on the policy documents provided.  
    Use the policy details to formulate precise answers that are aligned with the company's guidelines and regulations.  
    Ensure your response is concise, easy to understand, and authoritative.  
    Do not mention that you are referring to the context to answer the question.

    policy_query: {policy_query}  
    Policy Context: {policy_docs_context}  
    Answer:
    """

    policy_prompt = ChatPromptTemplate.from_template(policy_template)

    llm = groq_llama3_1_70b

    rag_chain = (
        policy_prompt
        | llm
        | StrOutputParser()
    )

    policy_text =  rag_chain.invoke({"policy_query": query, "policy_docs_context": policy_docs_context})

    return {'final_answer': policy_text}


# Apology Agent
def apology_agent(state):
    query = state['query']
    groq_llama3_1_70b = state['groq_llama3_1_70b']

    # Prompt
    template = """
    The query are not related to product recommendation, policy and cart management.
    Do not handle the query and apologize to the user.

    Query: {query} 
    Apology:
    """

    prompt = ChatPromptTemplate.from_template(template)

    llm = groq_llama3_1_70b
    
    apology_chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    apology_text =  apology_chain.invoke({"query": query})

    return {
        "final_answer": apology_text
    }