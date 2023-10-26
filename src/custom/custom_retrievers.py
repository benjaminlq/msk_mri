from typing import List
from llama_index import QueryBundle
from llama_index.indices.base_retriever import BaseRetriever
from llama_index.schema import NodeWithScore

from utils import count_tokens

class CustomCombinedRetriever(BaseRetriever):
    def __init__(
        self,
        table_retriever: BaseRetriever,
        text_retriever: BaseRetriever,
        token_limit: int = 6000,
    ) -> None:
        
        self.table_retriever = table_retriever
        self.text_retriever = text_retriever
        self._token_limit = token_limit
        
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""
        token_counter = 0
        
        table_nodes = self.table_retriever.retrieve(query_bundle)
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
    
