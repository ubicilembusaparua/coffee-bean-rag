import sys
from rag_app import RAGPGVector
from src.embedder import Embedder
import psycopg
from dotenv import load_dotenv
from openai import OpenAI
from langsmith.wrappers import wrap_openai



def create_assistant():
    conn = psycopg.connect(
        "postgresql://user:pswd@127.0.0.1:5432/coffe-review"
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