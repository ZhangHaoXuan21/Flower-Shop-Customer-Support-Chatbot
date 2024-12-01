import sqlite3

def add_item_to_cart_by_name(customer_id, product_name, quantity):
    try:
        with sqlite3.connect('flowers.db') as conn:
            cursor = conn.cursor()
            
            # Get product price and ID
            cursor.execute("SELECT ProductId, Price FROM Product WHERE ProductName = ?", (product_name,))
            product = cursor.fetchone()
            
            if product is None:
                raise ValueError("Product not found")
            
            product_id, price = product
            total_price = price * quantity
            
            # Add or update item in cart
            query = """
            INSERT INTO Cartlist (CustomerId, ProductId, Quantity, TotalPrice)
            VALUES (
                ?,
                ?,
                ?,
                ?
            )
            ON CONFLICT(CustomerId, ProductId)
            DO UPDATE SET
                Quantity = Quantity + excluded.Quantity,
                TotalPrice = (Quantity + excluded.Quantity) * (SELECT Price FROM Product WHERE ProductId = excluded.ProductId);
            """
            cursor.execute(query, (customer_id, product_id, quantity, total_price))
            conn.commit()

            return {
                "process_message":f"{product_name} added to cart successfully.",
                "completed":True
            }
        
    except sqlite3.Error as e:
        print(f"Error: {e}")
        return {
                "process_message":f"Fail adding {product_name} to cart",
                "completed":False
            }
    except ValueError as ve:
        print(ve)
        return {
                "process_message":f"Fail adding {product_name} to cart",
                "completed":False
            }

