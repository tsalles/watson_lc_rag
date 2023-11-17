import os
from typing import List

from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import DiscoveryV2
from ibm_watson.discovery_v2 import QueryLargePassages

from langchain.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)

from langchain.schema import BaseRetriever, Document


class DiscoveryRetriever(BaseRetriever):
    """`IBM Watson Discovery` retriever."""

    wds_client: DiscoveryV2 = None
    wds_project_id: str = None

    def __init__(self):
        super(DiscoveryRetriever, self).__init__()
        authenticator = IAMAuthenticator(os.environ["DISCOVERY_APIKEY"])
        self.wds_client = DiscoveryV2(version="2023-03-31", authenticator=authenticator)
        self.wds_project_id = os.environ["DISCOVERY_PROJECT_ID"]

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        result: List[Document] = []
        passage_query = QueryLargePassages(enabled=True, per_document=True, count=1)
        for d in self.wds_client.query(
            project_id=self.wds_project_id, natural_language_query=query, passages=passage_query
        ).get_result()["results"]:
            page_content = d["document_passages"][0]["passage_text"]
            metadata = {
                "publicationdate": d["extracted_metadata"]["publicationdate"],
                "filename": d["extracted_metadata"]["filename"],
            }
            result.append(Document(page_content=page_content, metadata=metadata))
        return result

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        raise NotImplementedError

