"""Custom Hybrid Retriever for Text & Table
"""

from typing import List, Optional, Sequence

from llama_index.indices.query.schema import QueryBundle, QueryType
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.indices.base_retriever import BaseRetriever
from llama_index.schema import NodeWithScore

from utils import count_tokens

class CustomCombinedRetriever(BaseRetriever):
    """Custom Hybrid Retriever for Text & Table
    """
    def __init__(
        self,
        table_retriever: BaseRetriever,
        text_retriever: BaseRetriever,
        token_limit: int = 6000,
    ) -> None:
        """Custom Hybrid Retriever for Text & Table

        Args:
            table_retriever (BaseRetriever): Table retriever instance.
            text_retriever (BaseRetriever): Text retriever instance
            token_limit (int, optional): Maximum token limit. Defaults to 6000.
        """
        self.table_retriever = table_retriever
        self.text_retriever = text_retriever
        self._token_limit = token_limit

    def retrieve(
        self,
        str_or_query_bundle: QueryType,
        table_filter: Optional[Sequence[str]] = None,
        text_filter: Optional[Sequence[str]] = None
        ) -> List[NodeWithScore]:
        """Retrieve document from databases with metadata filtering

        Args:
            str_or_query_bundle (QueryType): Input query
            table_filter (Optional[Sequence[str]], optional): Metadata Filtering on table retriever. Defaults to None.
            text_filter (Optional[Sequence[str]], optional): Metadata Filtering on text retriever. Defaults to None.

        Returns:
            List[NodeWithScore]: List of retrieved nodes
        """
        if isinstance(str_or_query_bundle, str):
            str_or_query_bundle = QueryBundle(str_or_query_bundle)
        return self._retrieve(str_or_query_bundle, table_filter, text_filter)

    async def aretrieve(self, str_or_query_bundle: QueryType) -> List[NodeWithScore]:
        """Asynchronously retrieve document from databases with metadata filtering

        Args:
            str_or_query_bundle (QueryType): Input query
            table_filter (Optional[Sequence[str]], optional): Metadata Filtering on table retriever. Defaults to None.
            text_filter (Optional[Sequence[str]], optional): Metadata Filtering on text retriever. Defaults to None.

        Returns:
            List[NodeWithScore]: List of retrieved nodes
        """
        if isinstance(str_or_query_bundle, str):
            str_or_query_bundle = QueryBundle(str_or_query_bundle)
        return await self._aretrieve(str_or_query_bundle)

    def _retrieve(
        self,
        query_bundle: QueryBundle,
        table_filter: Optional[Sequence[str]] = None,
        text_filter: Optional[Sequence[str]] = None
        ) -> List[NodeWithScore]:
        """Retrieve document from table and text databases. First, the hybrid retriever individually retrieve from text and table database with metadata filters.
        Then the text and table docs are combined to a single list, with relevance order reversed for text documents to prevent Lost In The Middle problem.
        
        NOTE: For this use case, tables are prioritized, thus texts documents are discarded first if total tokens exceeding limit.

        Args:
            str_or_query_bundle (QueryType): Input query
            table_filter (Optional[Sequence[str]], optional): Metadata Filtering on table retriever. Defaults to None.
            text_filter (Optional[Sequence[str]], optional): Metadata Filtering on text retriever. Defaults to None.

        Returns:
            List[NodeWithScore]: List of retrieved nodes
        """
        token_counter = 0
        
        if isinstance(self.table_retriever._vector_store, ChromaVectorStore):
            if table_filter is not None:
                if table_filter == []:
                    table_nodes = []
                else:
                    self.table_retriever._kwargs["where"] = {"condition": {"$in": table_filter}}
                    table_nodes = self.table_retriever.retrieve(query_bundle)
            else:
                self.table_retriever._kwargs = {}
                table_nodes = self.table_retriever.retrieve(query_bundle)
        
        if isinstance(self.text_retriever._vector_store, ChromaVectorStore):
            if text_filter is not None:
                if text_filter == []:
                    text_nodes = []
                else:
                    self.text_retriever._kwargs["where"] = {"condition": {"$in": text_filter}}
                    text_nodes = self.text_retriever.retrieve(query_bundle)
            else:
                self.text_retriever._kwargs = {}
                text_nodes = self.text_retriever.retrieve(query_bundle)
        
        text_retrieved_nodes = []
        table_retrieved_nodes = []
        
        for node in table_nodes:
            node_tokens = count_tokens(node)
            if token_counter + node_tokens <= self._token_limit:
                table_retrieved_nodes.append(node)
                token_counter += node_tokens
            else:
                Warning("Maximum Tokens Exceeded from Table Nodes.")
                break
            
        for node in text_nodes:
            node_tokens = count_tokens(node)
            if token_counter + node_tokens <= self._token_limit:
                text_retrieved_nodes.append(node)
                token_counter += node_tokens
            else:
                Warning("Maximum Tokens Exceeded from Text Nodes.")
                break
 
        text_retrieved_nodes = list(reversed(text_retrieved_nodes))
 
        return table_retrieved_nodes + text_retrieved_nodes

    
