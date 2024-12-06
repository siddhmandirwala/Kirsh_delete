# inference_pipeline.py

import uuid
import time
from datetime import datetime
from typing import List, Dict, Optional
from loguru import logger
from pydantic import BaseModel, Field
from uuid import uuid4
from tenacity import retry, stop_after_attempt, wait_exponential
import yaml
import json

# Core dependencies
from sentence_transformers import SentenceTransformer
import numpy as np
import openai
from qdrant_client import QdrantClient
from fastapi import FastAPI, HTTPException

# Import existing classes from your modules
from main import (
    Settings,
    MongoDatabaseConnector,
    NoSQLBaseDocument,
    VectorBaseDocument,
    EmbeddedArticleChunk,
    EmbeddedPostChunk,
    EmbeddedRepositoryChunk,
    EmbeddedQuery,
    DataCategory,
)
from cleaning import convert_numpy_to_list

class QueryInteraction(BaseModel):
    """Model for storing query interactions"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    query: str
    retrieved_chunks: List[Dict]
    response: str
    processing_time: float
    num_chunks: int

class PerformanceMetrics(BaseModel):
    """Model for storing performance metrics"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    processing_time: float
    num_chunks: int
    embedding_time: Optional[float]
    vector_search_time: Optional[float]
    llm_generation_time: Optional[float]

class InferencePipeline:
    def __init__(self, config_path: str = "config.yaml"):
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize settings
        self.settings = Settings.load_settings()
        
        # Initialize MongoDB connection
        self.mongo_connector = MongoDatabaseConnector()
        self.mongo_db = self.mongo_connector.get_database(self.settings.DATABASE_NAME)
        
        # Initialize Vector DB connection
        self.vector_client = QdrantClient(
            host=self.config["vector_db"]["host"],
            port=self.config["vector_db"]["port"],
            api_key=self.settings.QDRANT_APIKEY,
            prefer_grpc=True
        )
        
        # Initialize the embedding model
        self.embedding_model = SentenceTransformer(self.config["embedding"]["model_id"])
        
        # Initialize OpenAI
        openai.api_key = self.settings.OPENAI_API_KEY
        
        logger.info("InferencePipeline initialized successfully.")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def embed_query(self, query: str) -> List[float]:
        """Embeds the user query using the embedding model with retry logic"""
        start_time = time.time()
        try:
            embedding = self.embedding_model.encode(query, convert_to_tensor=False)
            embedding_time = time.time() - start_time
            logger.info(f"Query embedded in {embedding_time:.2f} seconds")
            return embedding.tolist(), embedding_time
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def search_vector_db(self, embedding: List[float], top_k: int = 5) -> tuple[List[Dict], float]:
        """Searches the Vector DB with retry logic"""
        start_time = time.time()
        try:
            search_result = self.vector_client.search(
                collection_name=self.config["vector_db"]["collection_name"],
                query_vector=embedding,
                limit=top_k,
                with_payload=True,
                with_vectors=False
            )
            
            retrieved_chunks = [
                {
                    "content": hit.payload.get("content"),
                    "metadata": hit.payload.get("metadata"),
                    "score": hit.score
                }
                for hit in search_result
            ]
            
            search_time = time.time() - start_time
            logger.info(f"Retrieved {len(retrieved_chunks)} chunks in {search_time:.2f} seconds")
            return retrieved_chunks, search_time
            
        except Exception as e:
            logger.error(f"Error searching Vector DB: {e}")
            raise

    def construct_prompt(self, retrieved_chunks: List[Dict], query: str) -> str:
        """Constructs a prompt with retrieved context and query"""
        # Format chunks with metadata and relevance scores
        formatted_chunks = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            chunk_text = (
                f"Chunk {i} (Relevance: {chunk['score']:.2f}):\n"
                f"Source: {chunk['metadata'].get('source', 'Unknown')}\n"
                f"Content: {chunk['content']}\n"
            )
            formatted_chunks.append(chunk_text)

        context = "\n\n".join(formatted_chunks)
        
        # Construct the complete prompt
        prompt = (
            f"Use the following chunks of information to answer the question. "
            f"If the information is not sufficient, say so.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )
        
        logger.info("Prompt constructed successfully")
        return prompt

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_response(self, prompt: str) -> tuple[str, float]:
        """Generates a response from the LLM with retry logic"""
        start_time = time.time()
        try:
            response = openai.Completion.create(
                engine=self.config["llm"]["model_id"],
                prompt=prompt,
                max_tokens=self.config["llm"]["max_tokens"],
                temperature=self.config["llm"]["temperature"],
                top_p=0.9,
                n=1,
                stop=None,
            )
            
            answer = response.choices[0].text.strip()
            generation_time = time.time() - start_time
            
            logger.info(f"Response generated in {generation_time:.2f} seconds")
            return answer, generation_time
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def _store_interaction(self, interaction: QueryInteraction):
        """Store query interaction in MongoDB"""
        try:
            self.mongo_db["query_interactions"].insert_one(
                interaction.dict(by_alias=True)
            )
            logger.info(f"Stored query interaction: {interaction.id}")
        except Exception as e:
            logger.error(f"Failed to store query interaction: {e}")

    def _store_metrics(self, metrics: PerformanceMetrics):
        """Store performance metrics in MongoDB"""
        try:
            self.mongo_db["performance_metrics"].insert_one(
                metrics.dict(by_alias=True)
            )
            logger.info(f"Stored performance metrics: {metrics.id}")
        except Exception as e:
            logger.error(f"Failed to store performance metrics: {e}")

    def handle_query(self, query: str) -> str:
        """Main query handling method"""
        start_time = time.time()
        
        try:
            # Step 1: Embed query
            query_embedding, embedding_time = self.embed_query(query)
            
            # Step 2: Search Vector DB
            retrieved_chunks, search_time = self.search_vector_db(query_embedding)
            
            if not retrieved_chunks:
                logger.warning("No relevant chunks retrieved")
                return "I'm sorry, I couldn't find relevant information to answer your question."
            
            # Step 3: Construct prompt
            prompt = self.construct_prompt(retrieved_chunks, query)
            
            # Step 4: Generate response
            response, generation_time = self.generate_response(prompt)
            
            # Calculate total processing time
            total_time = time.time() - start_time
            
            # Store interaction and metrics
            interaction = QueryInteraction(
                query=query,
                retrieved_chunks=retrieved_chunks,
                response=response,
                processing_time=total_time,
                num_chunks=len(retrieved_chunks)
            )
            
            metrics = PerformanceMetrics(
                processing_time=total_time,
                num_chunks=len(retrieved_chunks),
                embedding_time=embedding_time,
                vector_search_time=search_time,
                llm_generation_time=generation_time
            )
            
            self._store_interaction(interaction)
            self._store_metrics(metrics)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return "I apologize, but I encountered an error while processing your request."

# Initialize FastAPI app
app = FastAPI()
inference_pipeline = InferencePipeline()

class QueryRequest(BaseModel):
    text: str

@app.post("/query")
async def process_query(query: QueryRequest):
    try:
        response = inference_pipeline.handle_query(query.text)
        return {"response": response}
    except Exception as e:
        logger.error(f"API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)