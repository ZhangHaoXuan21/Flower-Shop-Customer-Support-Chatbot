def supervisor_route(state):
    supervisor_route_choice = state['supervisor_route_choice']

    if supervisor_route_choice == "product_recommendation_agent":
        return "product_recommendation_agent"
    elif supervisor_route_choice == "apology_agent":
        return "apology_agent"
    elif supervisor_route_choice == "cart_agent":
        return "cart_agent"
    elif supervisor_route_choice == "policy_agent":
        return "policy_agent"