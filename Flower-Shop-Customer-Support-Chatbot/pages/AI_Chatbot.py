import streamlit as st
import requests
import sqlite3

def get_products_by_names(database, product_names):
    """
    Retrieves product IDs and product names for a specified list of product names.
    
    Parameters:
        database (str): Path to the SQLite database file.
        product_names (list of str): A list of product names to filter.
    
    Returns:
        list of dict: A list of dictionaries containing ProductId, ProductName, and Price.
    """
    query = """
    SELECT ProductId, ProductName, Price
    FROM Product
    WHERE ProductName IN ({})
    """.format(", ".join("?" for _ in product_names))
    
    try:
        with sqlite3.connect(database) as conn:
            cursor = conn.cursor()
            cursor.execute(query, product_names)
            rows = cursor.fetchall()
            
            # Convert rows to a list of dictionaries
            products = [
                {"ProductId": row[0], "ProductName": row[1], "Price": row[2]}
                for row in rows
            ]
            return products

    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
        return []

# Initialize the Streamlit app
st.title("AI Flower Chatbot 🤖🌸")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if user_query := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_query)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        cust_id = "C001"
        # Define the URL of the API
        url = "http://127.0.0.1:8000/chatResponse"  # Replace with your server URL if deployed

        # Define the JSON payload
        payload = {
            "cust_id": cust_id,
            "user_query": user_query
        }

        # Make the POST request
        response = requests.post(url, json=payload).json()

        if response['supervisor_route_choice'] == "product_recommendation_agent":
            database = "flowers.db"
            product_names = response["recommend_products"]
            product_data = get_products_by_names(database, product_names)

            # Dynamically create a new container for product display
            product_container = st.container()
            with product_container:
                for item in product_data:
                    col1, col2 = st.columns([1, 1])  # Adjust column width ratio as needed
                    product_name = item['ProductName']
                    product_id = item['ProductId']
                    image_url = f"flowers_photo/{product_id}_{product_name}.png"
                    price = item['Price']

                    with col1:
                        st.image(image_url, width=200)  # Adjust width for larger images
                    with col2:
                        st.subheader(product_name)
                        st.write(f"Price: {price}")

        answer = response['final_answer']

        # Display the assistant's answer
        st.markdown(answer)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})
