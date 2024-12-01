import sqlite3
import streamlit as st

# Cart items page
st.title("Shopping Cart 🛒")

def get_cart_items(database, customer_id):
    """
    Retrieves all cart items for a given customer ID.
    
    Parameters:
        database (str): Path to the SQLite database file.
        customer_id (str): The ID of the customer whose cart items are to be retrieved.
    
    Returns:
        list of dict: A list of dictionaries containing cart item details.
    """
    query = """
    SELECT 
        Cartlist.ProductId, 
        Product.ProductName, 
        Product.Price, 
        Cartlist.Quantity, 
        Cartlist.TotalPrice
    FROM 
        Cartlist
    INNER JOIN 
        Product 
    ON 
        Cartlist.ProductId = Product.ProductId
    WHERE 
        Cartlist.CustomerId = ?
    """
    
    try:
        with sqlite3.connect(database) as conn:
            cursor = conn.cursor()
            cursor.execute(query, (customer_id,))
            rows = cursor.fetchall()
            
            # Convert rows to a list of dictionaries for better readability
            cart_items = [
                {
                    "ProductId": row[0],
                    "ProductName": row[1],
                    "Price": row[2],
                    "Quantity": row[3],
                    "TotalPrice": row[4],
                }
                for row in rows
            ]
            return cart_items

    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
        return []


cart_items = get_cart_items(database="flowers.db", customer_id="C001")

if cart_items != []:
    # Display cart items
    for item in cart_items:
        col1, col2 = st.columns([1, 1])  # Adjust column width ratio as needed

        product_name = item['ProductName']
        product_id = item['ProductId']
        image_url = f"flowers_photo/{product_id}_{product_name}.png"
        price=item['Price']
        quantity=item['Quantity']
        total_price=item['TotalPrice']

        with col1:
            st.image(image_url, width=200)  # Adjust width for larger images
        with col2:
            st.subheader(product_name)
            st.write(f"Price: {price}")
            st.write(f"Quantity: {quantity}")
            st.write(f"Total Price: {total_price}")
        st.markdown("---")