from llama_index.schema import Document
from typing import Union, Dict

import logging
import os
import sys

def convert_prompt_to_string(prompt) -> str:
    return prompt.format(**{v: v for v in prompt.template_vars})

def generate_query(profile: str, scan: str):
    return "Patient Profile: {}\nScan ordered: {}".format(profile, scan)

def convert_doc_to_dict(doc: Union[Document, Dict]) -> Dict:
    if isinstance(doc, Document):
        json_doc = {
            "page_content": doc.text,
            "metadata": {
                "source": doc.metadata["file_name"],
                "page": doc.metadata["page_label"]
            }
            }
    elif isinstance(doc, Dict):
        json_doc = {
            "page_content": doc["text"],
            "metadata": {
                "source": doc["metadata"]["file_name"],
                "page": doc["metadata"]["page_label"]
            }
        }
    return json_doc

def get_experiment_logs(description: str, log_folder: str):
    logger = logging.getLogger(description)

    stream_handler = logging.StreamHandler(sys.stdout)

    if not os.path.exists(log_folder):
        os.makedirs(log_folder, exist_ok=True)

    file_handler = logging.FileHandler(filename=os.path.join(log_folder, "logfile.log"))

    formatter = logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.setLevel(logging.INFO)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    
    return logger

def generate_vectorstore(
    embeddings: Callable,
    source_directory: Optional[str] = None,
    output_directory: str = "./vectorstore",
    emb_store_type: str = "faiss",
    chunk_size: int = 1000,
    chunk_overlap: int = 250,
    exclude_pages: Optional[Dict] = None,
    additional_docs: Optional[str] = None,
) -> VectorStore:
    """Generate New Vector Index Database

    Args:
        source_directory (str): Directory contains source documents
        embeddings (Callable): Function to convert text to vector embeddings
        output_directory (str, optional): Output directory of vector index database. Defaults to "./vectorstore".
        emb_store_type (str, optional): Type of vector index database. Defaults to "faiss".
        chunk_size (int, optional): Maximum size of text chunks (characters) after split. Defaults to 1000.
        chunk_overlap (int, optional): Maximum overlapping window between text chunks. Defaults to 250.
        exclude_pages (Optional[Dict], optional): Dictionary of pages to be excluded from documents. Defaults to None.
        pinecone_idx_name (Optional[str], optional): Name of pinecone index to be created or loaded. Defaults to None.
        additional_docs (Optional[str], optional): Additional Tables, Images or Json to be added to doc list. Defaults to None.
        key_path (Optional[str], optional): Path to file containing API info.
            Defaults to os.path.join(MAIN_DIR, "auth", "api_keys.json").

    Returns:
        Vectorstore: Vector Database
    """

    if os.path.exists(output_directory):
        rmtree(output_directory)
    os.makedirs(output_directory, exist_ok=True)

    if source_directory:
        LOGGER.info(f"Loading documents from {source_directory}")

        documents = load_documents(source_directory, exclude_pages=exclude_pages)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        texts = text_splitter.split_documents(documents)

        LOGGER.info(f"Loaded {len(documents)} documents from {source_directory}")
        LOGGER.info(
            f"Split into {len(texts)} chunks of text (max. {chunk_size} characters each)"
        )
    else:
        texts = []

    if additional_docs:
        with open(additional_docs, "r") as f:
            add_doc_infos = json.load(f)
        for add_doc_info in add_doc_infos:
            if add_doc_info["mode"] == "table":
                texts.extend(
                    convert_csv_to_documents(
                        add_doc_info, concatenate_rows=concatenate_rows
                    )
                )
            elif add_doc_info["mode"] == "json":
                texts.extend(convert_json_to_documents(add_doc_info))
            else:
                LOGGER.warning(
                    "Invalid document type. No texts added to documents list"
                )

    LOGGER.info(
        f"Total number of text chunks to create vector index store: {len(texts)}"
    )

    if emb_store_type == "chroma":
        chroma_settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=output_directory,
            anonymized_telemetry=False,
        )
        db = Chroma.from_documents(
            texts,
            embeddings,
            persist_directory=output_directory,
            client_settings=chroma_settings,
        )
        db.persist()

    elif emb_store_type == "faiss":
        db = FAISS.from_documents(texts, embedding=embeddings)
        db.save_local(output_directory)
        assert "index.faiss" in os.listdir(
            output_directory
        ) and "index.pkl" in os.listdir(output_directory)

    elif emb_store_type == "pinecone":
        with open(key_path, "r") as f:
            keys = json.loads(f)
        PINECONE_API_KEY = keys["PINECONE_API"]["KEY"]
        PINECONE_ENV = keys["PINECONE_API"]["ENV"]

        pinecone.init(
            api_key=PINECONE_API_KEY,
            environment=PINECONE_ENV,
        )

        if not pinecone_idx_name:
            pinecone_idx_name = "index_{}".format(
                datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
            )

        if pinecone_idx_name not in pinecone.list_indexes():
            db = Pinecone.from_documents(
                texts, embedding=embeddings, index_name=pinecone_idx_name
            )

        else:
            db = Pinecone.from_existing_index(pinecone_idx_name, embeddings)
            db.add_documents(texts)

    LOGGER.info(
        f"Successfully created {emb_store_type} vectorstore at {output_directory}"
    )

    return db