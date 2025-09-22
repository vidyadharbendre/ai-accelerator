# ================================================================
# RAG Implementation with LlamaIndex and LanceDB
# ================================================================
#
# This script demonstrates a complete RAG (Retrieval Augmented Generation)
# implementation using LlamaIndex and LanceDB.
#
# We explore three different approaches:
#   1. Vector Search Only – Fast retrieval without LLM generation
#   2. HuggingFace API Integration – Cloud-based LLM with authentication
#   3. Local LLM with Ollama – Complete local solution
#
# Overview:
# - Data loading and preparation from HuggingFace datasets
# - Vector store setup with LanceDB
# - Embedding generation with HuggingFace models
# - Three different query approaches with increasing complexity
# - Utility functions for table exploration and optimization
# ================================================================


# ================================================================
# 1. Install Required Dependencies
# ================================================================
#
# Install all required packages
# !pip install llama-index llama-index-vector-stores-lancedb \
#   llama-index-embeddings-huggingface llama-index-llms-huggingface-api \
#   lancedb datasets -q
#
# Additional packages for local LLM and utilities
# !pip install llama-index-llms-ollama requests -q
# ================================================================


# ================================================================
# 2. Import Libraries and Setup
# ================================================================
import os
import lancedb
import subprocess
import requests
import time
import asyncio
from pathlib import Path
from datasets import load_dataset

# LlamaIndex core components
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline

# Embedding and vector store
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.lancedb import LanceDBVectorStore

# LLM integrations
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.llms.ollama import Ollama

# Async support for notebooks
import nest_asyncio
nest_asyncio.apply()

print("All libraries imported successfully")


# ================================================================
# 3. Data Preparation and Loading
# ================================================================
def prepare_data(num_samples=100):
    """
    Load dataset and create document files
    """
    print(f"Loading {num_samples} personas from dataset...")

    dataset = load_dataset("dvilasuero/finepersonas-v0.1-tiny", split="train")
    Path("data").mkdir(parents=True, exist_ok=True)

    documents = []
    for i, persona in enumerate(dataset.select(range(min(num_samples, len(dataset))))):
        doc = Document(
            text=persona["persona"],
            metadata={
                "persona_id": i,
                "source": "finepersonas-dataset"
            }
        )
        documents.append(doc)

        with open(Path("data") / f"persona_{i}.txt", "w", encoding="utf-8") as f:
            f.write(persona["persona"])

    print(f"Prepared {len(documents)} documents")
    return documents


# ================================================================
# 4. LanceDB Vector Store Setup
# ================================================================
def setup_lancedb_store(table_name="personas_rag"):
    """
    Initialize LanceDB and create/connect to a table
    """
    print("Setting up LanceDB connection...")
    db = lancedb.connect("./lancedb_data")
    print(f"Connected to LanceDB, table: {table_name}")
    return db, table_name


# ================================================================
# 5. Vector Embeddings and Ingestion Pipeline
# ================================================================
async def create_and_populate_index(documents, db, table_name):
    """
    Create ingestion pipeline and populate LanceDB with embeddings
    """
    print("Creating embedding model and ingestion pipeline...")

    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )

    vector_store = LanceDBVectorStore(
        uri="./lancedb_data",
        table_name=table_name,
        mode="overwrite"
    )

    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=500, chunk_overlap=50),
            embed_model,
        ],
        vector_store=vector_store,
    )

    print("Processing documents and creating embeddings...")
    nodes = await pipeline.arun(documents=documents)
    print(f"Successfully processed {len(nodes)} text chunks")

    return vector_store, embed_model


# ================================================================
# 6. Option 1: Vector Search Only (No LLM)
# ================================================================
def perform_vector_search(db, table_name, query_text, embed_model, top_k=5):
    query_embedding = embed_model.get_text_embedding(query_text)
    table = db.open_table(table_name)
    results = table.search(query_embedding).limit(top_k*2).to_pandas().drop_duplicates(subset=["persona_id"]).head(top_k)   
    return results


def test_vector_search(db, table_name, embed_model):
    print("Testing Vector Search (No LLM needed)")
    print("=" * 50)

    queries = [
        "technology and artificial intelligence expert",
        "teacher educator professor",
        "environment climate sustainability",
        "art culture heritage creative"
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 30)

        results = perform_vector_search(db, table_name, query, embed_model, top_k=3)
        seen = set()
        for idx, row in results.iterrows():
            score = row.get('_distance', 'N/A')
            text = row.get('text', 'N/A')

            if text not in seen:
                seen.add(text)
                print(f"Result (Score: {score_str}): {text[:200]}...")

            if isinstance(score, (int, float)):
                score_str = f"{score:.3f}"
            else:
                score_str = str(score)

            print(f"\nResult {idx + 1} (Score: {score_str}):")
            print(f"{text[:200]}...")


# ================================================================
# Entry point for async execution
# ================================================================
# import asyncio

# async def main():
#     documents = prepare_data(num_samples=100)
#     db, table_name = setup_lancedb_store()
#     vector_store, embed_model = await create_and_populate_index(documents, db, table_name)

#     # Call test function with correct parameters
#     test_vector_search(db, table_name, embed_model)

#     # Optional future tests
#     # await test_huggingface_rag()
#     # await test_local_llm_rag()

# ================================================================
# Entry point for async execution
# ================================================================

# if __name__ == "__main__":
#     asyncio.run(main())
# import asyncio

async def main(run_vector=True, run_hf=False, run_ollama=False):
    # Step 1: Prepare data
    documents = prepare_data(num_samples=100)

    # Step 2: Setup LanceDB
    db, table_name = setup_lancedb_store()

    # Step 3: Create embeddings and populate vector store
    vector_store, embed_model = await create_and_populate_index(documents, db, table_name)

    # Step 4: Conditionally run test functions
    if run_vector:
        test_vector_search(db, table_name, embed_model)

    if run_hf:
        await test_huggingface_rag(vector_store, embed_model)

    if run_ollama:
        await test_local_llm_rag(vector_store, embed_model)

if __name__ == "__main__":
    asyncio.run(main(run_vector=True, run_hf=False, run_ollama=False))




