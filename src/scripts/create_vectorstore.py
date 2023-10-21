"""Script to generate embeddings store
"""
import argparse
import json
import os
import openai

from llama_index.embeddings import OpenAIEmbedding, LangchainEmbedding

from langchain.embeddings import HuggingFaceEmbeddings
from config import LOGGER, MAIN_DIR, DATA_DIR
from custom_storage import generate_vectorindex, load_vectorindex

def get_argument_parser():
    """Argument Parser

    Returns:
        args: argument dictionary
    """
    parser = argparse.ArgumentParser("Embedding Store Creation")
    parser.add_argument(
        "--embed_store", "-e", type=str, default="chroma",
        help="simple|chroma|faiss|pinecone|weavoate"
    )
    parser.add_argument(
        "--source", "-i", default=None, type=str,
        help="path to document source folder",
    )
    parser.add_argument(
        "--outputs", "-o", default=None, type=str,
        help="output directory to store embeddings",
    )
    parser.add_argument(
        "--model", "-m", type=str, default="openai",
        help="path to embedding model",
    )
    parser.add_argument(
        "--emb_size", "-d", type=int, default=1536,
        help="Dimension of embedding vector",
    )
    parser.add_argument(
        "--chunk_size", "-s", type=int, default=1024,
        help="chunk size to split documents",
    )
    parser.add_argument(
        "--chunk_overlap", "-v", type=int, default=20,
        help="overlap size between chunks",
    )
    parser.add_argument(
        "--index_name", "-n", type=str, default=None,
        help="Name of index",
    )
    parser.add_argument(
        "--exclude_pages", "-x", type=str, default=None,
        help="Path to json file containing information of excluded pages",
    )
    args = parser.parse_args()
    return args

def main():
    args = get_argument_parser()
    print(
        "Successfully created experiment with settings:\n{}".format(
            "\n".join([f"{k}:{v}" for k, v in vars(args).items()])
        )
    )
    
    emb_store_type = args.embed_store.lower()
    emb_directory = args.outputs
    emb_size = args.emb_size
    source_directory = args.source
    emb_model = args.model
    chunk_size = args.chunk_size
    chunk_overlap = args.chunk_overlap
    index_name = args.index_name
    exclude_pages_path = args.exclude_pages
    
    if emb_model.lower() == "openai":
        emb_model_name = "openai"
        with open(os.path.join(MAIN_DIR, "auth", "api_keys.json"), "r") as f:
            keys = json.load(f)
        os.environ["OPENAI_API_KEY"] = keys["OPENAI_API_KEY"]
        openai.api_key = keys["OPENAI_API_KEY"]
        embeddings = OpenAIEmbedding()
        LOGGER.info("Creating Vectorstore with OpenAI Embeddings")
    
    else:
        emb_model_name = emb_model.split("/")[-1]
        lc_embeddings = HuggingFaceEmbeddings(model_name=emb_model)
        embeddings = LangchainEmbedding(lc_embeddings)
        LOGGER.info("Creating Vectorstore with Sentence Transformer Embeddings")
        
    if not emb_directory:
        emb_directory = os.path.join(
            DATA_DIR, "emb_store", emb_store_type, f"{emb_model_name}_{chunk_size}_{chunk_overlap}"
        )
        if not os.path.exists(emb_directory):
            os.makedirs(emb_directory, exist_ok=True)

    if emb_store_type == "pinecone":
        with open(MAIN_DIR, "auth", "api_keys.json", "r") as f:
            api_keys = json.load(f)
        pinecone_api_key = api_keys["PINECONE_API"]["KEY"]
        pinecone_env = api_keys["PINECONE_API"]["ENV"]

    else:
        pinecone_api_key, pinecone_env = None, None

    if exclude_pages_path:
        assert exclude_pages_path.endswith(".json"), "Incorrect File Extension"
        with open(exclude_pages_path, "r") as f:
            exclude_pages = json.load(f)
            
    else:
        exclude_pages = None
            
    LOGGER.info("--- Start generating vectorstore ---")
    generate_vectorindex(
        embeddings=embeddings,
        emb_size=emb_size, 
        source_directory=source_directory,
        output_directory=emb_directory,
        emb_store_type=emb_store_type,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        exclude_pages=exclude_pages,
        index_name=index_name,
        pinecone_api_key=pinecone_api_key,
        pinecone_env=pinecone_env
    )

    # Test generated vector index:
    vector_index = load_vectorindex(
        db_directory=emb_directory,
        emb_store_type=emb_store_type,
        index_name=index_name,
        pinecone_api_key=pinecone_api_key,
        pinecone_env=pinecone_env
    )
    
    query_engine = vector_index.as_query_engine(similarity_top_k=2)
    sample_query = "Test Query"
    sample_response = query_engine.query(sample_query)
    assert len(sample_response.source_nodes) == 2, "Incorrect number of retrieved nodes"

if __name__ == "__main__":
    main()

# python ./src/scripts/create_vectorstore.py -e faiss -i ./data/document_sources -m openai -d 1536 -s 512 -v 20 -n vector_idx -x ./data/exclude_pages.json