import streamlit as st
import groq
import os
import json
import time
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment variable
groq_api_key = os.getenv('GROQ_API_KEY')

# Initialize the Groq client with the API key
client = groq.Groq(api_key=groq_api_key)

def analyze_dataset(df):
    analysis = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_summary": df.describe(include='all').to_dict(),
        "categorical_summary": {col: df[col].value_counts().to_dict() for col in df.select_dtypes(include=['object', 'category']).columns},
        "correlation": df.select_dtypes(include=[np.number]).corr().to_dict()
    }
    return analysis

def generate_code_for_query(prompt, analysis):
    messages = [
        {"role": "system", "content": """You are a Python code generator that helps users answer business intelligence queries based on a dataset. Given the dataset analysis, provide Python code that the user can run to answer their specific query. Ensure the code is clear and well-commented. Your response should be in JSON format."""},
        {"role": "user", "content": f"Dataset Analysis:\n{json.dumps(analysis, indent=2)}\n\nQuestion: {prompt}"}
    ]
    
    response = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=messages,
        max_tokens=300,
        temperature=0.2,
        response_format={"type": "json_object"}
    )
    
    return response.choices[0].message.content

def main():
    # Check if the API key is available
    if not groq_api_key:
        st.error("Groq API key not found. Please make sure you have a .env file with GROQ_API_KEY set.")
        return

    st.set_page_config(page_title="BI Analyst", page_icon="📊", layout="wide")
    
    st.title("BI Analyst: Analyzing CSV Data and Generating Python Code")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8')  # Attempt to read with UTF-8 encoding
        except UnicodeDecodeError:
            df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')  # Fallback to ISO-8859-1 encoding
        except Exception as e:
            st.error(f"Error reading the CSV file: {e}")
            return
        
        st.write("Data Preview:")
        st.dataframe(df.head())
        
        analysis = analyze_dataset(df)
        
        user_query = st.text_input("Enter your business question:", placeholder="e.g., What is the total revenue for each product category?")
        
        if user_query:
            st.write("Generating Python code to answer your query...")
            code = generate_code_for_query(user_query, analysis)
            st.code(code, language='python')

if __name__ == "__main__":
    main()