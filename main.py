import streamlit as st
import pandas as pd
import joblib
import re
import sqlite3
import json
from authentication import authenticate, show_user_profile, get_current_user
from google import genai
from dotenv import load_dotenv
import os

# Load variables from .env into environment
load_dotenv()

# Retrieve the API key
api_key = os.getenv("GEMINI_API_KEY")

# Ensure the key was loaded
if not api_key:
    raise ValueError("Missing GEMINI_API_KEY! Please check your .env file.")

# Create the client with the key
client = genai.Client(api_key=api_key)

# Create a chat session
chat = client.chats.create(model="gemini-2.5-flash")

# -------------------------------
# Page Config - MUST BE FIRST
# -------------------------------
st.set_page_config(page_title="Smart Real Estate Chatbot", page_icon="🏠", layout="wide")

# -------------------------------
# Authentication Check
# -------------------------------
if not authenticate():
    st.stop()

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
    print(f"[DEBUG] Checking database: location={location}, area={sqft}, bathrooms={bath}, bhk={bhk}")
    try:
        location = location.strip()
        cursor = conn.cursor()
        query = """
        SELECT location, total_sqft, bath, bhk, price 
        FROM properties 
        WHERE location LIKE ? 
        AND CAST(total_sqft AS INTEGER) = ?
        AND CAST(bath AS INTEGER) = ?
        AND CAST(bhk AS INTEGER) = ?
        """
        cursor.execute(query, (f"%{location}%", int(sqft), int(bath), int(bhk)))
        result = cursor.fetchone()
        
        if result:
            print(f"[DEBUG] ✓ Property found in database. Price: {result[4]} lacs")
            return True, result[4]
        print(f"[DEBUG] ✗ Property not in database. Will use ML prediction.")
        return False, None
    except Exception as e:
        print(f"[ERROR] Database error: {e}")
        return False, None

def format_dataframe_as_text(df, max_properties=10):
    """Convert dataframe to natural language chat format."""
    if df.empty:
        return "I couldn't find any properties matching your criteria."
    
    count = len(df)
    properties_to_show = df.head(max_properties)
    
    # Start with count
    if count == 1:
        response = "I found 1 property for you:\n\n"
    elif count <= max_properties:
        response = f"I found {count} properties for you:\n\n"
    else:
        response = f"I found {count} properties! Here are the top {max_properties}:\n\n"
    
    # Format each property
    for idx, row in properties_to_show.iterrows():
        location = row.get('location', 'N/A')
        sqft = row.get('total_sqft', 'N/A')
        bhk = int(row.get('bhk', 0)) if pd.notna(row.get('bhk')) else 'N/A'
        bath = int(row.get('bath', 0)) if pd.notna(row.get('bath')) else 'N/A'
        price = row.get('price', 'N/A')
        
        # Format price
        if price != 'N/A':
            price_str = f"₹{float(price):.2f} Lakhs"
        else:
            price_str = "Price not available"
        
        response += f"🏠 {bhk} BHK in {location}\n"
        response += f"   Area: {sqft} sqft | Bathrooms: {bath}\n"
        response += f"   Price: {price_str}\n\n"
    
    if count > max_properties:
        response += f"...and {count - max_properties} more properties matching your criteria."
    
    return response.strip()

# -------------------------------
# USING GEMINI 2.5 FLASH MODEL
# -------------------------------

def ask_gemini(user_message: str):
    """Send a plain string message to Gemini chat."""
    try:
        response = chat.send_message(user_message)
        content = response.text.strip()
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'\n\s*\n', '\n\n', content).strip()
        return content
    except Exception as e:
        print(f"[ERROR] Gemini API error: {e}")
        return f"⚠️ Error: {e}"

# -------------------------------
# Query Classification System Prompt
# -------------------------------

CLASSIFIER_PROMPT = """You are a query classifier for a Real Estate chatbot. 
Analyze the user's question and classify it into ONE of these categories:

1. **generic**: General conversation, greetings, small talk (hi, hello, how are you, good morning, thank you, etc.)
2. **price_prediction**: User wants to know the price/cost/estimate of a SPECIFIC property with details like location, sqft, BHK
3. **property_related**: User wants to search/list/query properties from database (show properties, list 2BHK, average price in area, etc.)

IMPORTANT: 
- If user asks "what's the price of X property" or "how much does X cost" with specific details → price_prediction
- If user asks "show me properties" or "list properties under X" → property_related
- If user just greets or chats → generic

Respond ONLY with a JSON object in this exact format:
{
    "category": "generic" OR "price_prediction" OR "property_related",
    "reasoning": "brief explanation"
}

Examples:

User: "Hi, how are you?"
{
    "category": "generic",
    "reasoning": "Simple greeting"
}

User: "What's the price of a 2750 sqft 4 BHK property in Jayanagar?"
{
    "category": "price_prediction",
    "reasoning": "User wants price estimate for specific property with location, sqft, and BHK details"
}

User: "Show me all 2 BHK properties under 100 lakhs"
{
    "category": "property_related",
    "reasoning": "User wants to search/list properties from database based on criteria"
}

Now classify the following user query:
"""

# -------------------------------
# SQL Generation System Prompt
# -------------------------------
SQL_SYSTEM_PROMPT = """
You are a Real Estate Assistant that generates SQL queries.
You have access to a SQLite database with a table called 'properties' containing:
- location (TEXT): Property location/area name
- total_sqft (REAL): Total square feet area
- bath (REAL): Number of bathrooms
- bhk (REAL): Number of bedrooms (BHK)
- price (REAL): Property price in lakhs

IMPORTANT SQL SYNTAX RULES:
- Table name: 'properties'
- Use LIKE with wildcards for text: location LIKE '%Jayanagar%'
- Column names: location, total_sqft, bath, bhk, price
- Always end queries with semicolon
- For numbers use: =, <, >, <=, >=

Generate the SQL query and put it between ``` markers.

Example 1:
User: "Show me properties in Jayanagar"
```
SELECT location, total_sqft, bath, bhk, price FROM properties 
WHERE location LIKE '%Jayanagar%';
```

Example 2:
User: "List 2 BHK properties under 150 lakhs"
```
SELECT location, total_sqft, bath, bhk, price FROM properties 
WHERE bhk = 2 AND price < 150 
ORDER BY price;
```
"""

# -------------------------------
# Extract House Details from Text
# -------------------------------

def extract_details(user_input):
    """Extract location, sqft, bath, bhk from user input using Gemini."""
    
    prompt = (
        f"You are an expert at extracting structured details from text. "
        f"Extract location, sqft, bath, bhk from: {user_input}\n\n"
        "Respond ONLY with a valid JSON object (no markdown, no code blocks, no extra text) in this exact format:\n"
        '{"location": "area_name", "sqft": 1200, "bath": 2, "bhk": 3}\n\n'
        "Example input: 'tell me price of property in Abbigere having area 2500 sqft, 6 bathrooms and 6 bhk'\n"
        'Example output: {"location": "Abbigere", "sqft": 2500, "bath": 6, "bhk": 6}\n\n'
        "If any field is missing, use null for that field. Return ONLY the JSON object, nothing else."
    )
    
    try:
        response = ask_gemini(prompt)
        print(f"[DEBUG] Raw Gemini response: {response}")
        
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response.strip()
        
        print(f"[DEBUG] Extracted JSON string: {json_str}")
        
        data = json.loads(json_str)
        
        location = data.get("location", "").strip() if data.get("location") else ""
        sqft = int(data.get("sqft", 0)) if data.get("sqft") else 0
        bath = int(data.get("bath")) if data.get("bath") is not None else None
        bhk = int(data.get("bhk", 0)) if data.get("bhk") else 0
        
        print(f"[DEBUG] Extracted details - location: {location}, sqft: {sqft}, bath: {bath}, bhk: {bhk}")
        return location, sqft, bath, bhk
        
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON parsing error: {e}")
        print(f"[ERROR] Failed to parse: {response}")
        return "", 0, None, 0
    except Exception as e:
        print(f"[ERROR] Error extracting details: {e}")
        return "", 0, None, 0

# -------------------------------
# Classify Query using LLM
# -------------------------------
def classify_query(user_input):
    """Use LLM to classify the query type."""
    try:
        llm_output = ask_gemini(f"{CLASSIFIER_PROMPT}\n\nUser: {user_input}")
        
        json_match = re.search(r'\{.*?\}', llm_output, re.DOTALL)
        if json_match:
            classification = json.loads(json_match.group())
            category = classification.get("category", "generic")
            reasoning = classification.get("reasoning", "")
            print(f"[DEBUG] Query classified as: {category} - {reasoning}")
            return category, reasoning
        
        llm_lower = llm_output.lower()
        if "price_prediction" in llm_lower:
            print(f"[DEBUG] Query classified as: price_prediction (fallback)")
            return "price_prediction", "Detected from response"
        elif "property_related" in llm_lower:
            print(f"[DEBUG] Query classified as: property_related (fallback)")
            return "property_related", "Detected from response"
        else:
            print(f"[DEBUG] Query classified as: generic (fallback)")
            return "generic", "Default classification"
            
    except Exception as e:
        print(f"[WARNING] Classification error: {e}. Defaulting to generic.")
        return "generic", f"Error: {e}"

# -------------------------------
# Extract SQL from LLM Response
# -------------------------------
def extract_sql(text):
    """Extract SQL query from LLM response."""
    sql_match = re.search(r"```(?:sql)?\s*(SELECT.*?)```", text, re.I | re.S)
    if sql_match:
        return sql_match.group(1).strip()
    
    sql_match = re.search(r"(SELECT\s+.*?;)", text, re.I | re.S)
    if sql_match:
        return sql_match.group(1).strip()
    
    return None

# -------------------------------
# Handle Generic Queries
# -------------------------------
def handle_generic_query(user_input):
    """Handle generic conversational queries."""
    try:
        prompt = f"You are a friendly Real Estate Assistant. Respond warmly and professionally to general conversation. Keep responses concise and helpful.\n\nUser: {user_input}"
        response = ask_gemini(prompt)
        return {"type": "text", "content": response}
    except Exception as e:
        print(f"[ERROR] Error in generic query handler: {e}")
        return {"type": "text", "content": f"⚠️ I apologize, but I'm having trouble processing your request right now. Please try again."}

# -------------------------------
# Handle Price Prediction Queries
# -------------------------------
def handle_price_prediction(user_input):
    """Handle price prediction queries - CHAT FORMAT ONLY."""
    location, sqft, bath, bhk = extract_details(user_input)
    
    if not (location and sqft and bhk):
        return {
            "type": "text",
            "content": "I need a few more details to give you an accurate price estimate. Could you tell me the location, square feet area, and number of bedrooms (BHK)? For example: 'What's the price of a 2000 sqft, 3 BHK property in Jayanagar?'"
        }
    
    if bath is None:
        bath = float(max(1, bhk - 1))
    
    exists, db_price = check_property_in_database(location, sqft, bath, bhk)
    
    if exists:
        response = f"Great! I found a property matching your description in {location}. "
        response += f"It's a {bhk} BHK with {int(bath)} bathrooms and {sqft} sqft of space. "
        response += f"The price for this property is ₹{db_price:.2f} Lakhs. "
        response += "This estimate is based on similar properties in our database."
        return {"type": "text", "content": response}
    else:
        try:
            predicted_price = predict_price(location, sqft, bath, bhk, model)
            response = f"Based on current market trends, a {bhk} BHK property in {location} "
            response += f"with {sqft} sqft and {int(bath)} bathrooms would be priced around "
            response += f"₹{predicted_price:.2f} Lakhs. "
            return {"type": "text", "content": response}
        except Exception as e:
            print(f"[ERROR] Error while predicting: {e}")
            return {"type": "text", "content": f"⚠️ I'm sorry, I couldn't generate a price estimate at this time. Please ensure the location is valid and try again."}

# -------------------------------
# Handle Property-Related Queries
# -------------------------------
def handle_property_query(user_input):
    """Handle database queries - CONVERT TO CHAT FORMAT."""
    
    prompt = f"{SQL_SYSTEM_PROMPT}\n\nUser: {user_input}"
    try:
        llm_output = ask_gemini(prompt)
    except Exception as e:
        print(f"[ERROR] Error communicating with LLM: {e}")
        return {"type": "text", "content": f"⚠️ I'm having trouble processing your request. Please try rephrasing your question."}
    
    sql_query = extract_sql(llm_output)
    
    if not sql_query:
        return {"type": "text", "content": f"I understand you're looking for property information. Could you please rephrase your question? For example: 'Show me 2 BHK properties under 100 lakhs' or 'List properties in Jayanagar'."}
    
    print(f"[DEBUG] Generated SQL: {sql_query}")
    
    df, error = query_database(sql_query)
    
    if error:
        print(f"[ERROR] SQL execution error: {error}")
        return {"type": "text", "content": f"⚠️ I encountered an error while searching for properties. Please try rephrasing your question."}
    
    if df is not None and not df.empty:
        # Convert dataframe to chat text
        chat_response = format_dataframe_as_text(df)
        return {"type": "text", "content": chat_response}
    else:
        return {"type": "text", "content": f"I couldn't find any properties matching your criteria. Try adjusting your search parameters or exploring different locations."}

# -------------------------------
# Main Query Router
# -------------------------------
def handle_query(user_input):
    """Route query based on LLM classification."""
    
    category, reasoning = classify_query(user_input)
    
    if category == "generic":
        return handle_generic_query(user_input)
    elif category == "price_prediction":
        return handle_price_prediction(user_input)
    elif category == "property_related":
        return handle_property_query(user_input)
    else:
        print(f"[WARNING] Unknown category: {category}")
        return {"type": "text", "content": "I'm not sure how to help with that. Could you please rephrase your question? I can help you with property price estimates, property searches, and general real estate questions."}

# -------------------------------
# Streamlit Chatbot UI
# -------------------------------

show_user_profile()

# Custom CSS - SIMPLIFIED FOR CHAT ONLY
st.markdown("""
<style>
    .main {
        background-color: #f5f7fa;
        padding: 0 !important;
    }
    
    .stApp {
        background-color: #f5f7fa !important;
        opacity: 1 !important;
        transition: none !important;
    }
    
    .main .block-container {
        opacity: 1 !important;
        transition: none !important;
    }
    
    .element-container {
        opacity: 1 !important;
        transition: none !important;
    }
    
    .stMarkdown {
        animation: none !important;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    button[kind="header"] {
        display: block !important;
    }
    
    section[data-testid="stSidebar"] button[kind="header"] {
        background-color: transparent;
        border: none;
        color: #1e3c72;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e1e8ed;
    }
    
    .header-bar {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 2rem;
        box-shadow: 0 2px 15px rgba(0,0,0,0.1);
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 999;
        backdrop-filter: blur(10px);
    }
    
    .header-title {
        font-size: 1.4rem;
        font-weight: 600;
        margin: 0;
        text-align: center;
        letter-spacing: 0.5px;
    }
    
    .chat-container {
        max-width: 900px;
        margin: 0 auto;
        padding: 90px 1.5rem 140px 1.5rem;
        min-height: calc(100vh - 80px);
    }
    
    .message {
        margin-bottom: 1.5rem;
        display: flex;
        animation: slideIn 0.3s ease-out;
    }
    
    @keyframes slideIn {
        from { 
            opacity: 0; 
            transform: translateY(20px);
        }
        to { 
            opacity: 1; 
            transform: translateY(0);
        }
    }
    
    .message.bot {
        justify-content: flex-start;
    }
    
    .message.bot .bubble {
        background-color: #ffffff;
        color: #2d3748;
        border-radius: 18px 18px 18px 4px;
        padding: 1.1rem 1.3rem;
        max-width: 75%;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
        line-height: 1.6;
        white-space: pre-line;
    }
    
    .message.user {
        justify-content: flex-end;
    }
    
    .message.user .bubble {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 18px 18px 4px 18px;
        padding: 1.1rem 1.3rem;
        max-width: 75%;
        box-shadow: 0 3px 12px rgba(102, 126, 234, 0.4);
        line-height: 1.6;
    }
    
    .stChatInput {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: linear-gradient(to top, #ffffff 0%, #ffffff 90%, rgba(255,255,255,0) 100%);
        padding: 1.5rem 2rem 1.5rem 2rem;
        box-shadow: 0 -3px 15px rgba(0,0,0,0.08);
        z-index: 1000;
    }
    
    .stChatInput input {
        border-radius: 25px !important;
        border: 2px solid #e2e8f0 !important;
        padding: 0.8rem 1.2rem !important;
        font-size: 0.95rem !important;
    }
    
    .stChatInput input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-bar">
    <div class="header-title">🏠 Smart Real Estate Assistant</div>
</div>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Display all messages as simple chat bubbles
for idx, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="message user">
            <div class="bubble">{msg['content']}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # All bot responses are now simple text
        st.markdown(f"""
        <div class="message bot">
            <div class="bubble">{msg['content']}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

user_input = st.chat_input("Ask me about properties, prices, or locations...")

if user_input:
    st.session_state.messages.append({
        "role": "user", 
        "type": "text", 
        "content": user_input
    })
    st.rerun()

if (st.session_state.messages and 
    st.session_state.messages[-1]["role"] == "user" and
    not st.session_state.get("processing", False)):
    
    st.session_state.processing = True
    last_user_msg = st.session_state.messages[-1]["content"]
    
    with st.spinner("🤔 Analyzing your query..."):
        bot_reply = handle_query(last_user_msg)
    
    # All responses are now stored as simple text
    st.session_state.messages.append({
        "role": "assistant",
        "type": "text",
        "content": bot_reply["content"]
    })
    
    st.session_state.processing = False
    st.rerun()