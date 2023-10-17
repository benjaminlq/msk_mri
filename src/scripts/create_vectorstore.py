"""Script to generate embeddings store
"""
import argparse
import json
import os
from shutil import rmtree
from typing import Optional, List, Dict, Literal
from logging import Logger

from llama_index import ServiceContext
from llama_index.storage import StorageContext
from llama_index.indices.vector_store import VectorStoreIndex
from llama_index.embeddings.base import BaseEmbedding
from llama_index import SimpleDirectoryReader
from llama_index.schema import Document
from llama_index import load_index_from_storage
from llama_index.embeddings import OpenAIEmbedding, LangchainEmbedding

from langchain.embeddings import HuggingFaceEmbeddings
from config import LOGGER, MAIN_DIR, DATA_DIR
from datetime import datetime

def filter_by_pages(
    doc_list: List[Document],
    exclude_info: Dict[str, str]
) -> List[Document]:
    filtered_list = []
    for doc in doc_list:
        file_name = doc.metadata["file_name"]
        page = doc.metadata["page_label"]
        if file_name not in exclude_info.keys():
            filtered_list.append(doc)
            continue
        if page not in exclude_info[file_name]:
            filtered_list.append(doc)

    return filtered_list

def generate_vectorindex(
    embeddings: BaseEmbedding,
    emb_size: int,
    source_directory: Optional[str] = None,
    output_directory: Optional[str] = None,
    emb_store_type: Literal["simple", "faiss", "pinecone", "chroma"] = "faiss", # To do: Add Weaviate
    chunk_size: int = 1024,
    chunk_overlap: int = 20,
    exclude_pages: Optional[Dict[str, str]] = None,
    logger: Logger = LOGGER,
    index_name: Optional[str] = None,
    pinecone_api_key: Optional[str] = None,
    pinecone_env: Optional[str] = None
) -> VectorStoreIndex:
    
    if os.path.exists(output_directory):
        rmtree(output_directory)
        
    os.makedirs(output_directory, exist_ok=True)

    logger.info(f"Loading documents from {source_directory}")
    documents = SimpleDirectoryReader(source_directory).load_data()
    
    logger.info(f"Loaded {len(documents)} documents from {source_directory}")
    
    if exclude_pages:
        documents = filter_by_pages(doc_list=documents, exclude_info=exclude_pages)
        
    logger.info(f"{len(documents)} documents remained after page filtering.")

    logger.info(
        f"Total number of text chunks to create vector index store: {len(texts)}"
    )

    service_context = ServiceContext.from_defaults(
        embed_model=embeddings,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    if emb_store_type == "simple":
        vector_store = SimpleVectorStore()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        vector_index = VectorStoreIndex.from_documents(
            documents=documents,
            service_context=service_context,
            storage_context=storage_context
        )

        if index_name:
            vector_index.set_index_id(index_name)
        
    elif emb_store_type == "faiss":
        import faiss
        from llama_index.vector_stores import FaissVectorStore
        faiss_index = faiss.IndexFlatL2(emb_size)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        vector_index = VectorStoreIndex.from_documents(
            documents, 
            service_context=service_context,
            storage_context=storage_context
        )
        
    elif emb_store_type == "pinecone":
        from llama_index.vector_stores import PineconeVectorStore
        assert pinecone_api_key is not None and pinecone_env is not None
        
        os.environ["PINECONE_API_KEY"] = pinecone_api_key

        if not index_name:
            index_name = "index_{}".format(
                datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
            )

        vector_store = PineconeVectorStore(
            index_name=index_name,
            environment=pinecone_env)
        
    elif emb_store_type == "chroma":
        __import__('pysqlite3')
        import sys
        sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
        import chromadb
        from llama_index.vector_stores import ChromaVectorStore
        
        chroma_client = chromadb.PersistentClient(path=output_directory)
        chroma_collection = chroma_client.get_or_create_collection(index_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    elif emb_store_type == "weaviate":
        from llama_index.vector_stores import SimpleVectorStore
        

    else:
        raise ValueError("Invalid Vectorstore type.")
    
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    vector_index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, service_context=service_context
    )
    
    if emb_store_type in ["faiss", "simple", ]:
        vector_index.storage_context.persist(persist_dir=output_directory)
    
    logger.info(f"Successfully created {emb_store_type} vectorstore at {output_directory}")

    return vector_index

def get_argument_parser():
    """Argument Parser

    Returns:
        args: argument dictionary
    """
    parser = argparse.ArgumentParser("Embedding Store Creation")
    parser.add_argument(
        "--embed_store", "-e", type=str, default="chroma",
        help="simple|chroma|faiss|pinecone"
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
        exclude_pages_path = None
            
    vector_index = generate_vectorindex(
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
    
    sample_query_engine = vector_index.as_query_engine(similarity_top_k=2)
    sample_query = "Test Query"
    response = sample_query_engine.query(sample_query)
    assert len(response.source_nodes) == 2
    
    LOGGER.info("Test for vector index passed")

if __name__ == "__main__":
    main()

# python ./src/scripts/create_vectorstore.py 