

# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define constants
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
FAISS_FILE_PATH = os.getenv("FAISS_FILE_PATH")


# Debugging: Print variables to check
print(f"GEMINI_API_KEY: {GEMINI_API_KEY}")
print(f"FAISS_FILE_PATH: {FAISS_FILE_PATH}")


if not GEMINI_API_KEY:
    raise ValueError("🚨 ERROR: GEMINI_API_KEY is missing! Please check your .env file.")

