import os
from shutil import rmtree
from typing import List, Optional, Literal, Dict, Union
from logging import Logger
from config import LOGGER
from datetime import datetime

from llama_index.embeddings.base import BaseEmbedding
from llama_index.schema import Document, TextNode, NodeWithScore
from llama_index.indices.vector_store import VectorStoreIndex
from llama_index import SimpleDirectoryReader
from llama_index import ServiceContext
from llama_index.storage import StorageContext
from llama_index import load_index_from_storage

def filter_by_pages(
    doc_list: List[Document],
    exclude_info: Dict[str, List]
) -> List[Document]:
    filtered_list = []
    for doc in doc_list:
        file_name = doc.metadata["file_name"]
        page = doc.metadata["page_label"]
        if file_name not in exclude_info.keys():
            filtered_list.append(doc)
            continue
        if int(page) not in exclude_info[file_name]:
            filtered_list.append(doc)

    return filtered_list

def convert_doc_to_dict(doc: Union[Document, NodeWithScore, Dict]) -> Dict:
    if isinstance(doc, NodeWithScore):
        json_doc = {
            "page_content": doc.text,
            "metadata": doc.metadata,
            "score": doc.score
            } 
    elif isinstance(doc, Document):
        json_doc = {
            "page_content": doc.text,
            "metadata": doc.metadata,
            "score": ""
            }
    elif isinstance(doc, Dict):
        json_doc = {
            "page_content": doc["text"],
            "metadata": doc["metadata"],
            "score": "None"
        }
    return json_doc

def generate_vectorindex(
    embeddings: BaseEmbedding,
    emb_size: int,
    source_directory: Optional[str] = None,
    documents: Optional[List[TextNode]] = None,
    output_directory: Optional[str] = None,
    emb_store_type: Literal["simple", "faiss", "pinecone", "chroma", "weaviate"] = "faiss",
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

    if not documents:
        logger.info(f"Loading documents from {source_directory}")
        documents = SimpleDirectoryReader(source_directory).load_data()
        logger.info(f"Loaded {len(documents)} documents from {source_directory}")
        
        if exclude_pages:
            documents = filter_by_pages(doc_list=documents, exclude_info=exclude_pages)
    
    else:
        logger.info("Processing documents from provided list.")
    
    logger.info(f"{len(documents)} documents remained after page filtering.")
    logger.info(f"Total number of documents to create vector index store: {len(documents)}")

    # Start here
    service_context = ServiceContext.from_defaults(
        embed_model=embeddings, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    if emb_store_type == "simple":
        logger.info(f"Creating {emb_store_type} Vectorstore")
        from llama_index.vector_stores import SimpleVectorStore
        vector_store = SimpleVectorStore()
        
    elif emb_store_type == "faiss":
        logger.info(f"Creating {emb_store_type} Vectorstore")
        import faiss
        from llama_index.vector_stores import FaissVectorStore
        faiss_index = faiss.IndexFlatL2(emb_size)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        
    elif emb_store_type == "pinecone":
        logger.info(f"Creating {emb_store_type} Vectorstore")
        from llama_index.vector_stores import PineconeVectorStore
        assert pinecone_api_key is not None and pinecone_env is not None, "Both Pinecone API Key and Env must be provided."
        os.environ["PINECONE_API_KEY"] = pinecone_api_key

        if not index_name:
            index_name = "index_{}".format(
                datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
            )

        vector_store = PineconeVectorStore(
            index_name=index_name,
            environment=pinecone_env)
        
    elif emb_store_type == "chroma":
        logger.info(f"Creating {emb_store_type} Vectorstore")
        __import__('pysqlite3')
        import sys
        sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
        import chromadb
        from llama_index.vector_stores import ChromaVectorStore
        
        chroma_client = chromadb.PersistentClient(path=output_directory)
        chroma_collection = chroma_client.get_or_create_collection(index_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    elif emb_store_type == "weaviate":
        logger.info(f"Creating {emb_store_type} Vectorstore")
        import weaviate
        from llama_index.vector_stores import WeaviateVectorStore
        from weaviate.embedded import EmbeddedOptions

        embedded_options = EmbeddedOptions(
            persistence_data_path=output_directory,
            binary_path=os.path.join(output_directory, "bin")
        )
        client = weaviate.Client(
            embedded_options=embedded_options
            )
        vector_store = WeaviateVectorStore(weaviate_client=client, index_name=index_name)

    else:
        raise ValueError("Invalid Vectorstore type.")
    
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    vector_index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, service_context=service_context
    )
    
    if emb_store_type in ["faiss", "simple"]:
        if emb_store_type == "simple" and index_name:
            vector_index.set_index_id(index_name)
            print(f"Succesfully set index name {index_name}")
        logger.info(f"Persist {emb_store_type} to local directory {output_directory}")
        vector_index.storage_context.persist(persist_dir=output_directory)
    
    logger.info(f"Successfully created {emb_store_type} vectorstore at {output_directory}")
    
def load_vectorindex(
    db_directory: str,
    emb_store_type: Literal["simple", "faiss", "pinecone", "chroma", "weaviate"] = "faiss",
    logger: Logger = LOGGER,
    index_name: Optional[str] = None,
    pinecone_api_key: Optional[str] = None,
    pinecone_env: Optional[str] = None
):
    assert os.path.exists(db_directory), f"Database does not exist at path {db_directory}"
    
    if emb_store_type == "simple":
        from llama_index.vector_stores import SimpleVectorStore
        vector_store = SimpleVectorStore.from_persist_dir(db_directory)
        storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=db_directory)
        vector_index = load_index_from_storage(storage_context=storage_context, index_id=index_name)

    elif emb_store_type == "faiss":
        from llama_index.vector_stores import FaissVectorStore
        vector_store = FaissVectorStore.from_persist_dir(db_directory)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir=db_directory)
        vector_index = load_index_from_storage(storage_context=storage_context)
        
    elif emb_store_type == "pinecone":
        from llama_index.vector_stores import PineconeVectorStore
        assert pinecone_api_key is not None and pinecone_env is not None, "Both Pinecone API Key and Env must be provided."
        os.environ["PINECONE_API_KEY"] = pinecone_api_key
        vector_store = PineconeVectorStore(index_name=index_name, environment=pinecone_env)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        vector_index = load_index_from_storage(storage_context=storage_context)
    
    elif emb_store_type == "chroma":
        __import__('pysqlite3')
        import sys
        sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
        import chromadb
        from llama_index.vector_stores import ChromaVectorStore
        chroma_client = chromadb.PersistentClient(path=db_directory)
        chroma_collection = chroma_client.get_or_create_collection(index_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        
    elif emb_store_type == "weaviate":
        import weaviate
        from llama_index.vector_stores import WeaviateVectorStore
        from weaviate.embedded import EmbeddedOptions

        embedded_options = EmbeddedOptions(
            persistence_data_path=db_directory,binary_path=os.path.join(db_directory, "bin")
            )
        client = weaviate.Client(embedded_options=embedded_options)
        vector_store = WeaviateVectorStore(weaviate_client=client, index_name=index_name)
        vector_index = VectorStoreIndex.from_vector_store(vector_store)
        
    else:
        raise ValueError("Vectorstore type not supported.")
    
    logger.info(f"{emb_store_type} VectorStore successfully loaded from {db_directory}.")
    
    return vector_index

