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

# Call the model
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain how AI works in a in 100 words."
)

# Print the output
print(response.text)
