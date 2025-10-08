import streamlit as st
import pandas as pd
import joblib
import re
import sqlite3
import ollama  # Local Ollama client

# -------------------------------
# Load Prediction Model
# -------------------------------
@st.cache_resource
def load_model(path="House_Prediction_Model.pkl"):
    return joblib.load(path)

model = load_model()

# -------------------------------
# Connect to Database
# -------------------------------
@st.cache_resource
def get_db_connection():
    return sqlite3.connect("real_estate.db", check_same_thread=False)

conn = get_db_connection()

# -------------------------------
# Helper Functions
# -------------------------------
def predict_price(location, sqft, bath, bhk, model):
    """Predict house price using ML model."""
    input_data = pd.DataFrame([[location, sqft, bath, bhk]],
                              columns=['location', 'total_sqft', 'bath', 'bhk'])
    return model.predict(input_data)[0]

def query_database(sql_query):
    """Run SQL query and return dataframe."""
    try:
        df = pd.read_sql_query(sql_query, conn)
        return df, None
    except Exception as e:
        return None, str(e)

def check_property_in_database(location, sqft, bath, bhk):
    """Check if exact property exists in database and return price."""
    try:
        # Query to find exact match
        query = """
        SELECT location, total_sqft, bath, bhk, price 
        FROM properties 
        WHERE location LIKE ? 
        AND total_sqft = ? 
        AND bath = ? 
        AND bhk = ?
        """
        cursor = conn.cursor()
        cursor.execute(query, (f"%{location}%", sqft, bath, bhk))
        result = cursor.fetchone()
        
        if result:
            return True, result[4]  # Return True and the price
        return False, None
    except Exception as e:
        st.error(f"Database error: {e}")
        return False, None

# -------------------------------
# Local Ollama Chat
# -------------------------------
MODEL_NAME = "deepseek-r1:1.5b"

def ask_llama(messages):
    """Send chat history to local Ollama model."""
    response = ollama.chat(model=MODEL_NAME, messages=messages)
    return response["message"]["content"].strip()

# -------------------------------
# System Prompt for SQL-Aware LLM
# -------------------------------
SYSTEM_PROMPT = """
You are a Real Estate Assistant chatbot. 
You can access a local SQLite database named real_estate.db with a table called 'properties' containing these columns:
- location (TEXT): Property location/area name
- total_sqft (REAL): Total square feet area
- bath (REAL): Number of bathrooms
- bhk (REAL): Number of bedrooms (BHK)
- price (REAL): Property price in lakhs

IMPORTANT SQL SYNTAX RULES:
- The table name is 'properties'
- Use LIKE operator with wildcards for text matching: location LIKE '%Jayanagar%'
- Column names: location, total_sqft, bath, bhk, price
- Always use semicolon at the end of SQL queries
- For numerical comparisons, use =, <, >, <=, >= operators

When a user asks a question:
1. Generate appropriate SQL query to fetch the requested data
2. Always put your SQL query between ``` markers
3. Keep explanations concise

Example 1:
User: "Show me properties in Jayanagar"
SQL Query:
```
SELECT location, total_sqft, bath, bhk, price FROM properties 
WHERE location LIKE '%Jayanagar%';
```

Example 2:
User: "What's the average price in JP Nagar?"
SQL Query:
```
SELECT AVG(price) AS avg_price FROM properties 
WHERE location LIKE '%JP Nagar%';
```

Example 3:
User: "List 2 BHK properties under 150 lakhs"
SQL Query:
```
SELECT location, total_sqft, bath, bhk, price FROM properties 
WHERE bhk = 2 AND price < 150 
ORDER BY price;
```

Example 4:
User: "Top 5 most expensive properties"
SQL Query:
```
SELECT location, total_sqft, bath, bhk, price FROM properties 
ORDER BY price DESC LIMIT 5;
```

Example 5:
User: "Tell me properties in Laggere below 50 lacs"
SQL Query:
```SELECT location, total_sqft, bath, bhk, price FROM properties 
WHERE location LIKE '%Laggere%' AND price < 50;
"""

# -------------------------------
# Extract House Details from Text
# -------------------------------
def extract_details(user_input):
    """Extract location, sqft, bath, bhk from user text."""
    sqft_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:sqft|square\s*feet|sq\s*feet)", user_input.lower())
    sqft = float(sqft_match.group(1)) if sqft_match else None

    bhk_match = re.search(r"(\d+)\s*(?:bhk|bedroom)", user_input.lower())
    bhk = int(bhk_match.group(1)) if bhk_match else None

    bath_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:bath|bathroom)", user_input.lower())
    bath = float(bath_match.group(1)) if bath_match else None

    # Extract location by removing numbers and property-related keywords
    location = re.sub(
        r"\d+(?:\.\d+)?|\bsqft\b|\bsquare\s*feet\b|\bsq\s*feet\b|\bbhk\b|\bbath\b|\bbathroom\b|\bbedroom\b|\bprice\b|\bof\b|\ba\b|\bin\b|\bfor\b|\bwith\b",
        "",
        user_input,
        flags=re.I
    ).strip(" ,.")

    return location, sqft, bath, bhk

# -------------------------------
# Check if query is for price prediction
# -------------------------------
def is_prediction_query(user_input):
    """Check if user wants a price prediction for specific property."""
    prediction_keywords = [
        "price of", "estimate", "predict", "cost of", "how much",
        "what's the price", "what is the price", "price for"
    ]
    return any(kw in user_input.lower() for kw in prediction_keywords)

# -------------------------------
# Extract SQL from LLM Response
# -------------------------------
def extract_sql(text):
    """Extract SQL query from LLM response."""
    # Look for SQL in code blocks
    sql_match = re.search(r"```(?:sql)?\s*(SELECT.*?)```", text, re.I | re.S)
    if sql_match:
        return sql_match.group(1).strip()
    
    # Look for raw SELECT statement
    sql_match = re.search(r"(SELECT\s+.*?;)", text, re.I | re.S)
    if sql_match:
        return sql_match.group(1).strip()
    
    return None

# -------------------------------
# Hybrid Query Handler (Model + DB)
# -------------------------------
def handle_query(user_input):
    """Route query to ML model or DB + LLM based on query type."""
    
    # Step 1: Check if this is a PRICE PREDICTION request
    if is_prediction_query(user_input):
        location, sqft, bath, bhk = extract_details(user_input)
        
        # Check if we have enough details for prediction
        if location and sqft and bhk:
            # If bath is not provided, use default heuristic
            if bath is None:
                bath = float(max(1, bhk - 1))
            
            # Step 2: Check if EXACT property exists in database
            exists, db_price = check_property_in_database(location, sqft, bath, bhk)
            
            if exists:
                # Property found in database - return actual price
                return {
                    "type": "text",
                    "content": f"""🏡 **Property Found in Database!**

**Property Details:**
- **Location:** {location}
- **Area:** {sqft} sqft
- **Bedrooms:** {bhk} BHK
- **Bathrooms:** {bath}

💰 **Actual Price: ₹{db_price:.2f} lacs**

✅ This property exists in our database."""
                }
            else:
                # Property NOT in database - use ML model to predict
                try:
                    predicted_price = predict_price(location, sqft, bath, bhk, model)
                    return {
                        "type": "text",
                        "content": f"""🏡 **ML Model Prediction**

**Property Details:**
- **Location:** {location}
- **Area:** {sqft} sqft
- **Bedrooms:** {bhk} BHK
- **Bathrooms:** {bath}

💰 **Predicted Price: ₹{predicted_price:.2f} lacs**

ℹ️ This property is not in our database. Price estimated using ML model."""
                    }
                except Exception as e:
                    return {"type": "text", "content": f"⚠️ Error while predicting: {e}"}
        else:
            return {
                "type": "text",
                "content": f"⚠️ Please provide complete property details: location, square feet, and number of bedrooms (BHK). Example: 'What's the price of 2750 sqft, 4 BHK, 4 bath property in Jayanagar?'"
            }
    
    # Step 3: NOT a price prediction query - use database to fetch information
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input}
    ]
    
    try:
        llm_output = ask_llama(messages)
    except Exception as e:
        return {"type": "text", "content": f"⚠️ Error communicating with LLM: {e}"}
    
    # Extract and execute SQL query
    sql_query = extract_sql(llm_output)
    
    if sql_query:
        df, error = query_database(sql_query)
        
        if error:
            return {
                "type": "text",
                "content": f"⚠️ **SQL Error:**\n{error}\n\n**Generated SQL:**\n```sql\n{sql_query}\n```\n\n**LLM Response:**\n{llm_output}"
            }
        
        if df is not None and not df.empty:
            # Extract explanation from LLM output (remove SQL part)
            explanation = re.sub(r"```(?:sql)?.*?```", "", llm_output, flags=re.S).strip()
            
            return {
                "type": "dataframe",
                "dataframe": df,
                "sql_query": sql_query,
                "explanation": explanation if explanation else "Here are the results from your query:"
            }
        else:
            return {
                "type": "text",
                "content": f"ℹ️ No matching records found in the database.\n\n**SQL Query Used:**\n```sql\n{sql_query}\n```"
            }
    
    # Fallback if no SQL detected
    return {
        "type": "text",
        "content": f"🤖 {llm_output}\n\nℹ️ No SQL query was generated. Please try rephrasing your question."
    }

# -------------------------------
# Streamlit Chatbot UI
# -------------------------------
# Set page config to prevent duplicate rendering
st.set_page_config(page_title="Smart Real Estate Chatbot", page_icon="🏠", layout="wide")

st.title("🏠 **Smart Real Estate Chatbot**")
st.markdown("Ask me about property prices, listings, or get predictions!")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("type") == "dataframe":
            st.markdown("### Properties->")
            st.dataframe(msg["dataframe"], use_container_width=True)
            st.markdown(f"\n**Summary:** {msg['explanation']}")
        else:
            st.markdown(msg["content"])

user_input = st.chat_input("Ask about house prices or properties...")

if user_input:
    st.session_state.messages.append({"role": "user", "type": "text", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            bot_reply = handle_query(user_input)
        
        if bot_reply["type"] == "dataframe":
            st.markdown("### 📊 Query Results")
            st.dataframe(bot_reply["dataframe"], use_container_width=True)
            st.markdown(f"\n**Summary:** {bot_reply['explanation']}")
        else:
            st.markdown(bot_reply["content"])
    
    st.session_state.messages.append({"role": "assistant", **bot_reply})