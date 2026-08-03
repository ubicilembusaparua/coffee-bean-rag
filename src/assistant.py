import sys

from dotenv import load_dotenv
from openai import OpenAI
from rag_app import RAGBase
from scripts.ingest_data import load_data, build_index

def create_assistant():
    load_dotenv()  # Load environment variables from .env file
    
    docs = load_data()
    index = build_index(docs)

    return RAGBase(
        index=index,
        llm_client=OpenAI()
    )

if __name__ == "__main__":
    assistant = create_assistant()
    query = "What is the flavor profile of the coffee from Ethiopia?"
    
    if len(sys.argv) > 1:
        query = sys.argv[1]
    
    answer = assistant.rag(query)
    print(answer.output_text)

