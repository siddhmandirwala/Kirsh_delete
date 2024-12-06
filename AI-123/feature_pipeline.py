# feature_pipeline.py

import uuid
from typing import List, Dict
from loguru import logger
from pymongo import MongoClient, errors
from pydantic import BaseModel, Field
from uuid import uuid4

# Import existing classes from your modules
from cleaning import (
    CleaningHandler,
    ChunkingHandler,
    EmbeddingHandler,
    convert_numpy_to_list,
    DataPipeline,
)
from scraper_github import GithubCrawler
from scraper_medium import MediumCrawler
from main import Settings, MongoDatabaseConnector, NoSQLBaseDocument, VectorBaseDocument, EmbeddedArticleChunk, EmbeddedPostChunk, EmbeddedRepositoryChunk, EmbeddedQuery, DataCategory

# Assuming you have VectorBaseDocument and its subclasses defined appropriately
# and a connection to Vector DB is established within those classes.

class FeaturePipeline:
    def __init__(self):
        # Initialize MongoDB connection
        self.mongo_connector = MongoDatabaseConnector()
        self.mongo_db = self.mongo_connector.get_database("twin")
        
        # Initialize cleaning, chunking, and embedding handlers
        self.cleaning_handler = CleaningHandler()
        self.chunking_handler = ChunkingHandler()
        self.embedding_handler = EmbeddingHandler()
        
        # Initialize Vector DB connection (assuming Qdrant as an example)
        from qdrant_client import QdrantClient
        self.vector_client = QdrantClient(host=settings.QDRANT_DATABASE_HOST, port=settings.QDRANT_DATABASE_PORT, api_key=settings.QDRANT_APIKEY)
        
    def process_repository_by_id(self, repo_id: str):
        """Fetch raw data from MongoDB, process it, and push to Vector DB."""
        collection = self.mongo_db["repositories"]
        
        try:
            # Step 1: Fetch raw data from MongoDB
            raw_data = collection.find_one({"_id": repo_id})
            if not raw_data:
                logger.error(f"No repository found with id: {repo_id}")
                return
            
            # Step 2: Clean the data
            cleaned_data = self.cleaning_handler.clean_repository(raw_data)
            
            # Step 3: Chunk the content
            chunks = self.chunking_handler.chunk(cleaned_data["content"])
            
            # Step 4: Embed the chunks
            embedded_chunks = self.embedding_handler.embed_chunks(chunks)
            
            # Step 5: Insert embedded chunks into Vector DB
            if embedded_chunks:
                # Determine the VectorBaseDocument subclass based on DataCategory
                data_category = cleaned_data.get("category", "repositories")  # Default to 'repositories' if not specified
                if data_category == DataCategory.ARTICLES:
                    VectorBaseDocumentClass = EmbeddedArticleChunk
                elif data_category == DataCategory.POSTS:
                    VectorBaseDocumentClass = EmbeddedPostChunk
                elif data_category == DataCategory.REPOSITORIES:
                    VectorBaseDocumentClass = EmbeddedRepositoryChunk
                elif data_category == DataCategory.QUERIES:
                    VectorBaseDocumentClass = EmbeddedQuery
                else:
                    logger.error(f"Unsupported data category: {data_category}")
                    return
                
                success = VectorBaseDocumentClass.bulk_insert(embedded_chunks)
                if success:
                    logger.info(f"Successfully inserted {len(embedded_chunks)} chunks into Vector DB.")
                else:
                    logger.error("Failed to insert some chunks into Vector DB.")
            else:
                logger.warning("No embedded chunks to insert into Vector DB.")
            
            # Step 6: Update MongoDB with processed data
            update_data = {
                "cleaned_content": cleaned_data["content"],
                "embedded_chunks": convert_numpy_to_list(embedded_chunks),
            }
            collection.update_one({"_id": repo_id}, {"$set": update_data})
            
            logger.info(f"Repository {repo_id} processed and updated successfully.")
        
        except Exception as e:
            logger.exception(f"An error occurred while processing repository {repo_id}: {e}")
    
    def process_all_repositories(self):
        """Process all repositories in the MongoDB collection."""
        collection = self.mongo_db["repositories"]
        try:
            all_repos = collection.find()
            for repo in all_repos:
                repo_id = repo["_id"]
                self.process_repository_by_id(repo_id)
        except errors.PyMongoError as e:
            logger.error(f"Failed to fetch repositories from MongoDB: {e}")
    
    def run(self):
        """Run the feature pipeline."""
        logger.info("Starting the Feature Pipeline...")
        self.process_all_repositories()
        logger.info("Feature Pipeline completed.")

if __name__ == "__main__":
    # Load settings
    settings = Settings.load_settings()
    
    # Initialize and run the feature pipeline
    pipeline = FeaturePipeline()
    pipeline.run()