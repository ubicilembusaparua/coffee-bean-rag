#!/usr/bin/env python
import os
import sys
from pathlib import Path
import psycopg
import numpy as np
from tqdm.auto import tqdm
import pandas as pd

# Setup imports relative to project root
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


from src.embedder import Embedder

# Read environment variables set by Docker / Compose
DB_HOST = os.getenv("POSTGRES_HOST", "postgres")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "coffe-review")
DB_USER = os.getenv("POSTGRES_USER", "user").strip()
DB_PASS = os.getenv("POSTGRES_PASSWORD", "pswd").strip()

from pathlib import Path
import pandas as pd

def load_data():
    # Resolve the absolute path to project_root/data/coffee_analysis.csv
    data_path = Path(__file__).resolve().parent.parent / "data" / "coffee_analysis.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at expected location: {data_path}")

    df = pd.read_csv(data_path)
    
    cols = df.columns.tolist()
    df.dropna(inplace=True)
    df[['100g_USD', 'rating']] = df[['100g_USD', 'rating']].astype(str)
    docs = df.to_dict(orient='records')

    for idx, item in enumerate(docs):
        item['id'] = idx

    return docs


def vec_to_str(vector):
    return "[" + ",".join(str(x) for x in vector) + "]"


def main():
    print(f"Connecting to PostgreSQL at {DB_HOST}:{DB_PORT}...")
    conn = psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
    )

    with conn.cursor() as cur:
        # 1. Enable extension & create table
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS coffee_reviews (
                id SERIAL PRIMARY KEY,
                name TEXT,
                origin_1 TEXT,
                origin_2 TEXT,
                desc_1 TEXT,
                desc_2 TEXT,
                desc_3 TEXT,
                roast TEXT,
                rating TEXT,
                loc_country TEXT,
                embedding vector(384)
            );
        """
        )
        conn.commit()

        # 2. Skip if table already populated
        cur.execute("SELECT COUNT(*) FROM coffee_reviews;")
        count = cur.fetchone()[0]
        if count > 0:
            print(f"Table 'coffee_reviews' already contains {count} records. Skipping ingestion.")
            conn.close()
            return

    # 3. Load model and data
    print("Loading data and model...")
    model = Embedder()
    data = load_data()

    texts = [
        f"{doc.get('name', '')} {doc.get('origin_1', '')} {doc.get('origin_2', '')} "
        f"{doc.get('desc_1', '')} {doc.get('desc_2', '')} {doc.get('desc_3', '')} "
        f"{doc.get('roast', '')} {doc.get('rating', '')} {doc.get('loc_country', '')}"
        for doc in data
    ]

    # 4. Generate Embeddings
    print("Generating embeddings...")
    batch_size = 50
    X = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i : i + batch_size]
        batch_vectors = model.encode_batch(batch)
        X.extend(batch_vectors)

    # 5. Batch Insert Data
    print("Inserting records into database...")
    records = [
        (
            doc.get("name"),
            doc.get("origin_1"),
            doc.get("origin_2"),
            doc.get("desc_1"),
            doc.get("desc_2"),
            doc.get("desc_3"),
            doc.get("roast"),
            doc.get("rating"),
            doc.get("loc_country"),
            vec_to_str(vec),
        )
        for doc, vec in zip(data, X)
    ]

    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO coffee_reviews (
                name, origin_1, origin_2, desc_1, desc_2, desc_3, 
                roast, rating, loc_country, embedding
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
            """,
            records,
        )
        conn.commit()

        # 6. Create HNSW index on the correct table
        print("Creating HNSW vector index...")
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS coffee_reviews_embedding_idx 
            ON coffee_reviews USING hnsw (embedding vector_cosine_ops);
        """
        )
        conn.commit()

    conn.close()
    print("Ingestion and indexing complete successfully.")


if __name__ == "__main__":
    main()