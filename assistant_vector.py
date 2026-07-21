from rag_app import RAGPGVector
from sentence_transformers import SentenceTransformer
import psycopg
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
from openai import OpenAI
import sys

conn = psycopg.connect(
    "postgresql://user:pswd@127.0.0.1:5432/coffe-review"
)

def create_assistant():
    model = SentenceTransformer('all-MiniLM-L6-v2')

    return RAGPGVector(
        embedder=model,
        conn=conn,
        llm_client=OpenAI()
    )

if __name__ == "__main__":
    assistant = create_assistant()
    query = "best coffee from Indonesia?"
    
    if len(sys.argv) > 1:
        query = sys.argv[1]
    
    answer = assistant.rag(query)
    print(answer.output_text)