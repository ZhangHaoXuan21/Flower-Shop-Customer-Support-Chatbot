from flower_agents.utils.state import GraphState
from flower_agents.utils.nodes import supervisor_agent, product_recommendation_agent, cart_agent, policy_agent, apology_agent
from flower_agents.utils.conditional_edges import supervisor_route

from IPython.display import Image, display
from langgraph.graph import END, StateGraph


class FlowerAgents:
    def __init__(self):
        # Initialise the graph
        workflow = StateGraph(GraphState)

        # Add nodes
        workflow.add_node("supervisor_agent", supervisor_agent)
        workflow.add_node("product_recommendation_agent", product_recommendation_agent)
        workflow.add_node("cart_agent", cart_agent)
        workflow.add_node("policy_agent", policy_agent)
        workflow.add_node("apology_agent", apology_agent)


        # Add Edges
        workflow.set_entry_point("supervisor_agent")
        workflow.add_conditional_edges(
            source="supervisor_agent",
            path=supervisor_route,
            path_map={
                "product_recommendation_agent": "product_recommendation_agent",
                "apology_agent": "apology_agent",
                "cart_agent": "cart_agent",
                "policy_agent": "policy_agent"
            }
        )
        workflow.add_edge("product_recommendation_agent", END)
        workflow.add_edge("cart_agent", END)
        workflow.add_edge("policy_agent", END)
        workflow.add_edge("apology_agent", END)

        self._flower_graph = workflow.compile()   
        
    def __call__(self, state):
        result_data = self._flower_graph.invoke(state)
        return result_data




