# Import necessary libraries and modules
import os
from abc import ABC
from dataclasses import dataclass
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    List,
    Optional,
    TypedDict,
    Union,
    cast,
)
from urllib.parse import urljoin

import aiohttp
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import (
    QueryCaptionResult,
    QueryType,
    VectorizedQuery,
    VectorQuery,
)
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from core.authentication import AuthenticationHelper
from text import nonewlines

# Define a dataclass for Document
@dataclass
class Document:
    # Define the attributes of the Document class
    id: Optional[str]
    content: Optional[str]
    embedding: Optional[List[float]]
    image_embedding: Optional[List[float]]
    category: Optional[str]
    sourcepage: Optional[str]
    sourcefile: Optional[str]
    oids: Optional[List[str]]
    groups: Optional[List[str]]
    captions: List[QueryCaptionResult]
    score: Optional[float] = None
    reranker_score: Optional[float] = None

    # Define a method to serialize the Document object for results
    def serialize_for_results(self) -> dict[str, Any]:
        return {
            "id": self.id,  # Document ID
            "content": self.content,  # Document content
            "embedding": Document.trim_embedding(self.embedding),  # Document embedding
            "imageEmbedding": Document.trim_embedding(self.image_embedding),  # Image embedding
            "category": self.category,  # Document category
            "sourcepage": self.sourcepage,  # Source page of the document
            "sourcefile": self.sourcefile,  # Source file of the document
            "oids": self.oids,  # List of oids
            "groups": self.groups,  # List of groups
            "captions": (  # List of captions
                [
                    {
                        "additional_properties": caption.additional_properties,  # Additional properties of the caption
                        "text": caption.text,  # Caption text
                        "highlights": caption.highlights,  # Caption highlights
                    }
                    for caption in self.captions
                ]
                if self.captions
                else []
            ),
            "score": self.score,  # Document score
            "reranker_score": self.reranker_score,  # Reranker score
        }

    # Define a class method to trim the embedding
    @classmethod
    def trim_embedding(cls, embedding: Optional[List[float]]) -> Optional[str]:
        """Returns a trimmed list of floats from the vector embedding."""
        if embedding:
            if len(embedding) > 2:
                # Format the embedding list to show the first 2 items followed by the count of the remaining items.
                return f"[{embedding[0]}, {embedding[1]} ...+{len(embedding) - 2} more]"
            else:
                return str(embedding)

        return None
# Define a dataclass for ThoughtStep
@dataclass
class ThoughtStep:
    title: str  # Title of the thought step
    description: Optional[Any]  # Description of the thought step
    props: Optional[dict[str, Any]] = None  # Properties of the thought step

# Define an abstract base class for Approach
class Approach(ABC):
    # Initialize the Approach class with necessary parameters
    def __init__(
        self,
        search_client: SearchClient,  # Client for search operations
        openai_client: AsyncOpenAI,  # Client for OpenAI operations
        auth_helper: AuthenticationHelper,  # Helper for authentication operations
        query_language: Optional[str],  # Language for the query
        query_speller: Optional[str],  # Speller for the query
        embedding_deployment: Optional[str],  # Deployment for the embedding
        embedding_model: str,  # Model for the embedding
        embedding_dimensions: int,  # Dimensions for the embedding
        openai_host: str,  # Host for OpenAI
        vision_endpoint: str,  # Endpoint for vision operations
        vision_token_provider: Callable[[], Awaitable[str]],  # Provider for vision token
    ):
        # Assign the parameters to class variables
        self.search_client = search_client
        self.openai_client = openai_client
        self.auth_helper = auth_helper
        self.query_language = query_language
        self.query_speller = query_speller
        self.embedding_deployment = embedding_deployment
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.openai_host = openai_host
        self.vision_endpoint = vision_endpoint
        self.vision_token_provider = vision_token_provider

    # Define a method to build a filter
    def build_filter(self, overrides: dict[str, Any], auth_claims: dict[str, Any]) -> Optional[str]:
        # Get the category to exclude from the overrides
        exclude_category = overrides.get("exclude_category")
        # Build security filters using the auth helper
        security_filter = self.auth_helper.build_security_filters(overrides, auth_claims)
        filters = []
        # If there is a category to exclude, add it to the filters
        if exclude_category:
            filters.append("category ne '{}'".format(exclude_category.replace("'", "''")))
        # If there is a security filter, add it to the filters
        if security_filter:
            filters.append(security_filter)
        # If there are no filters, return None, else join the filters with 'and'
        return None if len(filters) == 0 else " and ".join(filters)

    # Define an asynchronous method to perform a search
    async def search(
        self,
        top: int,  # Number of top results to return
        query_text: Optional[str],  # Text of the query
        filter: Optional[str],  # Filter for the search
        vectors: List[VectorQuery],  # List of vector queries
        use_semantic_ranker: bool,  # Flag to use semantic ranker
        use_semantic_captions: bool,  # Flag to use semantic captions
        minimum_search_score: Optional[float],  # Minimum search score
        minimum_reranker_score: Optional[float],  # Minimum reranker score
    ) -> List[Document]:  # Return a list of Document objects
        # If semantic ranker is to be used and there is a query text, perform a semantic search
        if use_semantic_ranker and query_text:
            results = await self.search_client.search(
                search_text=query_text,
                filter=filter,
                query_type=QueryType.SEMANTIC,
                query_language=self.query_language,
                query_speller=self.query_speller,
                semantic_configuration_name="default",
                top=top,
                query_caption="extractive|highlight-false" if use_semantic_captions else None,
                vector_queries=vectors,
            )
        else:
            # If semantic ranker is not to be used or there is no query text, perform a regular search
            results = await self.search_client.search(
                search_text=query_text or "", filter=filter, top=top, vector_queries=vectors
            )

        documents = []
        # Iterate over the pages in the results
        async for page in results.by_page():
            # Iterate over the documents in the page
            async for document in page:
                # Append a Document object created from the document to the documents list
                documents.append(
                    Document(
                        id=document.get("id"),
                        content=document.get("content"),
                        embedding=document.get("embedding"),
                        image_embedding=document.get("imageEmbedding"),
                        category=document.get("category"),
                        sourcepage=document.get("sourcepage"),
                        sourcefile=document.get("sourcefile"),
                        oids=document.get("oids"),
                        groups=document.get("groups"),
                        captions=cast(List[QueryCaptionResult], document.get("@search.captions")),
                        score=document.get("@search.score"),
                        reranker_score=document.get("@search.reranker_score"),
                    )
                )

        # Filter the documents to only include those with a score and reranker score greater than or equal to the minimums
        qualified_documents = [
            doc
            for doc in documents
            if (
                (doc.score or 0) >= (minimum_search_score or 0)
                and (doc.reranker_score or 0) >= (minimum_reranker_score or 0)
            )
        ]

        # Return the list of qualified documents
        return qualified_documents

    def get_sources_content(
        self, results: List[Document], use_semantic_captions: bool, use_image_citation: bool
    ) -> list[str]:
        # Check if semantic captions are to be used
        if use_semantic_captions:
            # Generate the sources content with semantic captions
            return [
                (self.get_citation((doc.sourcepage or ""), use_image_citation))
                + ": "
                + nonewlines(" . ".join([cast(str, c.text) for c in (doc.captions or [])]))
                for doc in results
            ]
        else:
            # Generate the sources content without semantic captions
            return [
                (self.get_citation((doc.sourcepage or ""), use_image_citation)) + ": " + nonewlines(doc.content or "")
                for doc in results
            ]

    def get_citation(self, sourcepage: str, use_image_citation: bool) -> str:
        # Check if image citation is to be used
        if use_image_citation:
            return sourcepage
        else:
            # Extract the path and extension from the sourcepage
            path, ext = os.path.splitext(sourcepage)
            # If the extension is ".png", extract the page number from the path and return the PDF citation
            if ext.lower() == ".png":
                page_idx = path.rfind("-")
                page_number = int(path[page_idx + 1:])
                return f"{path[:page_idx]}.pdf#page={page_number}"
            # If the extension is not ".png", return the sourcepage as it is
            return sourcepage

    async def compute_text_embedding(self, q: str):
        SUPPORTED_DIMENSIONS_MODEL = {
            "text-embedding-ada-002": False,
            "text-embedding-3-small": True,
            "text-embedding-3-large": True,
        }

        class ExtraArgs(TypedDict, total=False):
            dimensions: int

        dimensions_args: ExtraArgs = (
            {"dimensions": self.embedding_dimensions} if SUPPORTED_DIMENSIONS_MODEL[self.embedding_model] else {}
        )
        # Create an embedding using the OpenAI client
        embedding = await self.openai_client.embeddings.create(
            # Azure OpenAI takes the deployment name as the model name
            model=self.embedding_deployment if self.embedding_deployment else self.embedding_model,
            input=q,
            **dimensions_args,
        )
        # Get the query vector from the embedding
        query_vector = embedding.data[0].embedding
        # Return a VectorizedQuery object with the query vector and other parameters
        return VectorizedQuery(vector=query_vector, k_nearest_neighbors=50, fields="embedding")

    async def compute_image_embedding(self, q: str):
        # Define the endpoint for the image embedding computation
        endpoint = urljoin(self.vision_endpoint, "computervision/retrieval:vectorizeText")
        # Set the headers for the request
        headers = {"Content-Type": "application/json"}
        # Set the parameters for the request
        params = {"api-version": "2023-02-01-preview", "modelVersion": "latest"}
        # Set the data for the request
        data = {"text": q}

        # Set the authorization header using the vision token provider
        headers["Authorization"] = "Bearer " + await self.vision_token_provider()

        # Create a session for making the request
        async with aiohttp.ClientSession() as session:
            # Send a POST request to the endpoint with the specified parameters, headers, and data
            async with session.post(
                url=endpoint, params=params, headers=headers, json=data, raise_for_status=True
            ) as response:
                # Parse the response as JSON
                json = await response.json()
                # Get the image query vector from the response
                image_query_vector = json["vector"]
        # Return a VectorizedQuery object with the image query vector and other parameters
        return VectorizedQuery(vector=image_query_vector, k_nearest_neighbors=50, fields="imageEmbedding")

    async def run(
        self,
        messages: list[ChatCompletionMessageParam],
        stream: bool = False,
        session_state: Any = None,
        context: dict[str, Any] = {},
    ) -> Union[dict[str, Any], AsyncGenerator[dict[str, Any], None]]:
        # This method is not implemented yet
        raise NotImplementedError
