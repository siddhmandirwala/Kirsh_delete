import numpy as np
from numpy.typing import NDArray

from enum import Enum

class DataCategory(str, Enum):
    POSTS = "posts"
    ARTICLES = "articles"
    REPOSITORIES = "repositories"
    QUERIES = "queries"

from uuid import uuid4
from typing import Any, Dict, List
from pydantic import BaseModel, Field

class BaseDocument(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))

    def save_to_mongo(self, collection):
        """Save document to MongoDB."""
        collection.insert_one(self.dict())

    @classmethod
    def from_mongo(cls, data: dict):
        """Load document from MongoDB."""
        data["id"] = data.pop("_id")
        return cls(**data)

class VectorBaseDocument(BaseDocument):
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class CleaningHandler:
    def clean_repository(self, data: Dict) -> Dict:
        # Filter out None or non-string values before joining
        content_values = [v for v in data.get("content", {}).values() if v and isinstance(v, str)]
        return {
            "_id": data["_id"],
            "content": self._clean_text(" ".join(content_values)),
            "name": data.get("name"),
            "link": data.get("link"),
            "platform": data.get("platform"),
        }

    def clean_post(self, data: Dict) -> Dict:
        content_values = [v for v in data.get("content", {}).values() if v and isinstance(v, str)]
        return {
            "id": data["id"],
            "content": self._clean_text(" ".join(content_values)),
            "platform": data.get("platform"),
            "image": data.get("image"),
        }

    def clean_article(self, data: Dict) -> Dict:
        content_values = [v for v in data.get("content", {}).values() if v and isinstance(v, str)]
        return {
            "id": data["id"],
            "content": self._clean_text(" ".join(content_values)),
            "link": data.get("link"),
            "platform": data.get("platform"),
        }

    @staticmethod
    def _clean_text(text: str) -> str:
        import re
        text = re.sub(r"[^\w\s.,!?]", " ", text)
        return re.sub(r"\s+", " ", text).strip()
    
def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
        """
        Splits the input text into chunks of specified size with overlap.
    
        Args:
            text (str): The input text to split.
            chunk_size (int): Maximum size of each chunk.
            chunk_overlap (int): Overlap between consecutive chunks.
    
        Returns:
            list[str]: List of chunks.
        """
        import re
        sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s", text)
    
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence + " "
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
    
        if current_chunk:
            chunks.append(current_chunk.strip())
    
        return chunks

class ChunkingHandler:
    def chunk(self, cleaned_content: str) -> list[dict]:
        """
        Splits the cleaned content into chunks and returns them as dictionaries.
        """
        chunks = chunk_text(cleaned_content)  # Returns list of strings
        chunk_dicts = [{"content": chunk} for chunk in chunks]  # Wrap each string in a dictionary
        return chunk_dicts
    
class EmbeddingModel:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_tensor=False)

embedding_model = EmbeddingModel()

class EmbeddingHandler:
    def embed_chunks(self, chunks):
        """
        Embeds the given chunks and returns them with their embeddings.
        """
        embedded_chunks = []
        for chunk in chunks:
            embedding = embedding_model.embed([chunk["content"]])  # Call the `embed` method
            embedded_chunk = {
                "content": chunk["content"],
                "embedding": embedding.tolist(),  # Convert to list
                "metadata": chunk.get("metadata", {}),
            }
            embedded_chunks.append(embedded_chunk)
        return embedded_chunks
    
def convert_numpy_to_list(data):
    """Recursively converts all NumPy arrays to Python lists."""
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, dict):
        return {k: convert_numpy_to_list(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_to_list(v) for v in data]
    return data

from pymongo import MongoClient
from typing import List, Dict

class DataPipeline:
    def __init__(self, collection):
        self.collection = collection
        self.cleaning_handler = CleaningHandler()
        self.chunking_handler = ChunkingHandler()
        self.embedding_handler = EmbeddingHandler()

    def process_repository_by_id(self, repo_id: str):
        """Fetch raw data from MongoDB, process it, and update it."""
        # Fetch raw data from MongoDB
        raw_data = self.collection.find_one({"_id": repo_id})
        if not raw_data:
            print(f"No repository found with id: {repo_id}")
            return

        # Step 1: Clean the data
        cleaned_data = self.cleaning_handler.clean_repository(raw_data)

        # Step 2: Chunk the content
        chunks = self.chunking_handler.chunk(cleaned_data["content"])

        # Step 3: Embed the chunks
        embedded_chunks = self.embedding_handler.embed_chunks(chunks)

        # Step 4: Update MongoDB with processed data
        update_data = {
            "cleaned_content": cleaned_data["content"],
            "chunks": convert_numpy_to_list(embedded_chunks),
        }
        self.collection.update_one({"_id": repo_id}, {"$set": update_data})

        print(f"Repository {repo_id} processed successfully.")

    def process_multiple_repositories_by_ids(self, repo_ids: List[str]):
        """Process multiple repositories by their IDs."""
        for repo_id in repo_ids:
            self.process_repository_by_id(repo_id)

from pymongo import MongoClient

# Define a mapping of database names to a tuple containing the collection name and the list of repository IDs.
databases_to_process = {
    "medium_scraper": {
        "collection_name": "repositories",
        "repo_ids": [
            "23f21c5e-c1e0-437b-98d9-031af1ca61fa"
            
        ]
    },
    "github_scraper": {
        "collection_name": "repositories",
        "repo_ids": [
            "dfc752ea-96d5-49e9-98d3-0672b4f208cf"
        ]
    }
}

# Initialize the MongoDB client
client = MongoClient("mongodb://localhost:27017/")

for db_name, config in databases_to_process.items():
    db = client[db_name]
    collection = db[config["collection_name"]]

    # Initialize pipeline for the given database and collection
    pipeline = DataPipeline(collection)

    # Process repositories by their IDs for this database
    repo_ids = config["repo_ids"]
    pipeline.process_multiple_repositories_by_ids(repo_ids)