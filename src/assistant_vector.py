import sys
import os
from rag_app import RAGPGVector
from src.embedder import Embedder
import psycopg
from dotenv import load_dotenv
from openai import OpenAI
from langsmith.wrappers import wrap_openai



def create_assistant():
    DB_HOST = os.getenv("POSTGRES_HOST") or os.getenv("DB_HOST") or "postgres"
    DB_PORT = os.getenv("POSTGRES_PORT", "5432")
    DB_NAME = os.getenv("POSTGRES_DB", "coffe-review")
    DB_USER = os.getenv("POSTGRES_USER", "user").strip()
    DB_PASS = os.getenv("POSTGRES_PASSWORD", "pswd").strip()

    conn = psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )
    load_dotenv()  # Load environment variables from .env file
    model = Embedder()

    return RAGPGVector(
        embedder=model,
        conn=conn,
        llm_client=wrap_openai(OpenAI())
    )

if __name__ == "__main__":
    assistant = create_assistant()
    query = "best coffee from Indonesia?"
    
    if len(sys.argv) > 1:
        query = sys.argv[1]
    
    answer = assistant.rag(query)
    print(answer.output_text)