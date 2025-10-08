import streamlit as st
import pandas as pd
import joblib
import requests
import re
import json

# -------------------------------
# Load Prediction Model
# -------------------------------
@st.cache_resource
def load_model(path="House_Prediction_Model.pkl"):
    return joblib.load(path)

model = load_model()

def predict_price(location, sqft, bath, bhk, model):
    """Predict house price based on user input."""
    input_data = pd.DataFrame([[location, sqft, bath, bhk]],
                              columns=['location', 'total_sqft', 'bath', 'bhk'])
    return model.predict(input_data)[0]

# -------------------------------
# Ollama Chat (stream-safe)
# -------------------------------
OLLAMA_API = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.2:1b"

def ask_llama(messages):
    """Send chat history to Ollama and get response (handles streaming)."""
    response = requests.post(
        OLLAMA_API,
        json={"model": MODEL_NAME, "messages": messages},
        stream=True
    )

    full_reply = ""
    for line in response.iter_lines():
        if line:
            try:
                data = json.loads(line.decode("utf-8"))
                if "message" in data and "content" in data["message"]:
                    full_reply += data["message"]["content"]
            except json.JSONDecodeError:
                continue
    return full_reply.strip()

# -------------------------------
# Extract House Details from Text
# -------------------------------
def extract_details(user_input):
    """Extract location, sqft, bath, bhk from user text."""
    # Find sqft
    sqft_match = re.search(r"(\d+)\s*(?:sqft|square\s*feet|sq\s*feet)", user_input.lower())
    sqft = float(sqft_match.group(1)) if sqft_match else None

    # Find bhk
    bhk_match = re.search(r"(\d+)\s*bhk", user_input.lower())
    bhk = int(bhk_match.group(1)) if bhk_match else None

    # Find bath
    bath_match = re.search(r"(\d+)\s*bath", user_input.lower())
    bath = int(bath_match.group(1)) if bath_match else None

    # Location = remove numbers + keywords
    location = re.sub(r"\d+|\bsqft\b|\bsquare feet\b|\bsq feet\b|\bbhk\b|\bbath\b", "", user_input, flags=re.I).strip(" ,.")

    return location, sqft, bath, bhk

# -------------------------------
# Streamlit Chatbot UI
# -------------------------------
st.title("**House Price Chatbot**")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "You are a helpful assistant that predicts house prices. "
                                      "If the user provides location, total_sqft, bath, and bhk, "
                                      "use the internal prediction model to give the price. "}
    ]

# Show history
for msg in st.session_state.messages[1:]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask about house prices...")

if user_input:
    # Add user input
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Try extracting details
    location, sqft, bath, bhk = extract_details(user_input)

    if location and sqft and bath and bhk:
        try:
            price = predict_price(location, sqft, bath, bhk, model)
            bot_reply = f"🏡 Estimated house price for **{location}** ({sqft} sqft, {bath} bath, {bhk} BHK) is: **₹ {price:,.2f}**"
        except Exception as e:
            bot_reply = f"⚠️ Error while predicting: {e}"
    else:
        # Normal chat with Llama
        bot_reply = ask_llama(st.session_state.messages)

    # Add bot reply
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant"):
        st.markdown(bot_reply)
