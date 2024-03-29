"""Custom Generic Langchain Module for context docs reorder
"""
from langchain.chains import RetrievalQA
from typing import List, Optional
from langchain.schema import Document, BaseDocumentTransformer
from langchain.callbacks.manager import CallbackManagerForChainRun

class ReOrderQARetrieval(RetrievalQA):
    """Custom Generic Langchain Module for context docs reorder
    """
    reorder_fn: Optional[BaseDocumentTransformer] = None
    
    def _get_docs(
        self,
        question: str,
        *,
        run_manager: CallbackManagerForChainRun,
    ) -> List[Document]:
        """Get docs."""
        docs = self.retriever.get_relevant_documents(
            question, callbacks=run_manager.get_child()
        )
        
        docs = self.reorder_fn.transform_documents(docs) if self.reorder_fn else docs
     
        return docs