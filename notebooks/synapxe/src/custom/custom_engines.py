"""Custom Query Engine for Text & Table hybrid retrieval
"""
from typing import Optional, List, Sequence

from llama_index import QueryBundle
from llama_index.callbacks.schema import CBEventType, EventPayload
from llama_index.indices.query.schema import QueryBundle, QueryType
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.response.schema import RESPONSE_TYPE
from llama_index.schema import  NodeWithScore

class CustomRetrieverQueryEngine(RetrieverQueryEngine):
    """Custom Query Engine for Text & Table hybrid retrieval
    """
    def query(
        self,
        str_or_query_bundle: QueryType,
        table_filter: Optional[Sequence[str]] = None,
        text_filter: Optional[Sequence[str]] = None
        ) -> RESPONSE_TYPE:
        """RAG main query function. Retrieve documents from database(s) and generate prompt grounded
        on retrieved documents

        Args:
            str_or_query_bundle (QueryType): Input query
            table_filter (Optional[Sequence[str]], optional): Metadata Filtering on table retriever. Defaults to None.
            text_filter (Optional[Sequence[str]], optional): Metadata Filtering on text retriever. Defaults to None.

        Returns:
            RESPONSE_TYPE: RAG response
        """
        with self.callback_manager.as_trace("query"):
            if isinstance(str_or_query_bundle, str):
                str_or_query_bundle = QueryBundle(str_or_query_bundle)
            return self._query(str_or_query_bundle, table_filter, text_filter)

    async def aquery(
        self,
        str_or_query_bundle: QueryType,
        table_filter: Optional[Sequence[str]] = None,
        text_filter: Optional[Sequence[str]] = None
        ) -> RESPONSE_TYPE:
        """For asynchrorous query calls

        Args:
            str_or_query_bundle (QueryType): Input query
            table_filter (Optional[Sequence[str]], optional): Metadata Filtering on table retriever. Defaults to None.
            text_filter (Optional[Sequence[str]], optional): Metadata Filtering on text retriever. Defaults to None.

        Returns:
            RESPONSE_TYPE: RAG response
        """
        with self.callback_manager.as_trace("query"):
            if isinstance(str_or_query_bundle, str):
                str_or_query_bundle = QueryBundle(str_or_query_bundle)
            return await self._aquery(str_or_query_bundle, table_filter, text_filter)
        
    def _query(
        self,
        query_bundle: QueryBundle,
        table_filter: Optional[Sequence[str]] = None,
        text_filter: Optional[Sequence[str]] = None
        ) -> RESPONSE_TYPE:
        """RAG main query function. Retrieve documents from database(s) and generate prompt grounded
        on retrieved documents

        Args:
            str_or_query_bundle (QueryType): Input query
            table_filter (Optional[Sequence[str]], optional): Metadata Filtering on table retriever. Defaults to None.
            text_filter (Optional[Sequence[str]], optional): Metadata Filtering on text retriever. Defaults to None.

        Returns:
            RESPONSE_TYPE: RAG response
        """
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            with self.callback_manager.event(
                CBEventType.RETRIEVE,
                payload={EventPayload.QUERY_STR: query_bundle.query_str},
            ) as retrieve_event:
                nodes = self.retrieve(query_bundle, table_filter, text_filter)

                retrieve_event.on_end(
                    payload={EventPayload.NODES: nodes},
                )

            response = self._response_synthesizer.synthesize(
                query=query_bundle,
                nodes=nodes,
            )

            query_event.on_end(payload={EventPayload.RESPONSE: response})

        return response

    async def _aquery(
        self,
        query_bundle: QueryBundle,
        table_filter: Optional[Sequence[str]] = None,
        text_filter: Optional[Sequence[str]] = None
        ) -> RESPONSE_TYPE:
        """For asynchrorous query calls

        Args:
            str_or_query_bundle (QueryType): Input query
            table_filter (Optional[Sequence[str]], optional): Metadata Filtering on table retriever. Defaults to None.
            text_filter (Optional[Sequence[str]], optional): Metadata Filtering on text retriever. Defaults to None.

        Returns:
            RESPONSE_TYPE: RAG response
        """
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            with self.callback_manager.event(
                CBEventType.RETRIEVE,
                payload={EventPayload.QUERY_STR: query_bundle.query_str},
            ) as retrieve_event:
                nodes = await self.aretrieve(query_bundle, query_bundle, table_filter, text_filter)

                retrieve_event.on_end(
                    payload={EventPayload.NODES: nodes},
                )

            response = await self._response_synthesizer.asynthesize(
                query=query_bundle,
                nodes=nodes,
            )

            query_event.on_end(payload={EventPayload.RESPONSE: response})

        return response
    
    def retrieve(
        self, 
        query_bundle: QueryBundle,
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
        nodes = self._retriever.retrieve(query_bundle, table_filter, text_filter)
        return self._apply_node_postprocessors(nodes, query_bundle=query_bundle)

    async def aretrieve(
        self, 
        query_bundle: QueryBundle,
        table_filter: Optional[Sequence[str]] = None,
        text_filter: Optional[Sequence[str]] = None
        ) -> List[NodeWithScore]:
        """Asynchronously retrieve document from databases with metadata filtering

        Args:
            str_or_query_bundle (QueryType): Input query
            table_filter (Optional[Sequence[str]], optional): Metadata Filtering on table retriever. Defaults to None.
            text_filter (Optional[Sequence[str]], optional): Metadata Filtering on text retriever. Defaults to None.

        Returns:
            List[NodeWithScore]: List of retrieved nodes
        """
        nodes = await self._retriever.aretrieve(query_bundle, table_filter, text_filter)
        return self._apply_node_postprocessors(nodes, query_bundle=query_bundle)