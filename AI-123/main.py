# main.py

import os
import shutil
import subprocess
import tempfile
import uuid
from abc import ABC, abstractmethod
from typing import Generic, Type, TypeVar, Dict, Any, List
from loguru import logger
from pydantic import UUID4, BaseModel, Field
from pymongo import MongoClient, errors
from pymongo.errors import ConnectionFailure
from pydantic_settings import BaseSettings, SettingsConfigDict

import chromedriver_autoinstaller
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from zenml.client import Client
from zenml.exceptions import EntityExistsError

# -------------------------
# Custom Exceptions
# -------------------------

class LLMTwinException(Exception):
    """Base exception for LLMTwin."""
    pass

class ImproperlyConfigured(LLMTwinException):
    """Exception raised for improperly configured settings."""
    pass

# -------------------------
# Settings Class
# -------------------------

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # --- Required settings even when working locally. ---
    
    # OpenAI API
    OPENAI_MODEL_ID: str = "gpt-4o-mini"
    OPENAI_API_KEY: str | None = None

    # Huggingface API
    HUGGINGFACE_ACCESS_TOKEN: str | None = None

    # Comet ML (during training)
    COMET_API_KEY: str | None = None
    COMET_PROJECT: str = "twin"

    # --- Required settings when deploying the code. ---
    # --- Otherwise, default values work fine. ---

    # MongoDB database
    DATABASE_HOST: str = "mongodb://llm_engineering:llm_engineering@127.0.0.1:27017"
    DATABASE_NAME: str = "twin"

    # Qdrant vector database
    USE_QDRANT_CLOUD: bool = False
    QDRANT_DATABASE_HOST: str = "localhost"
    QDRANT_DATABASE_PORT: int = 6333
    QDRANT_CLOUD_URL: str = "str"
    QDRANT_APIKEY: str | None = None

    # AWS Authentication
    AWS_REGION: str = "eu-central-1"
    AWS_ACCESS_KEY: str | None = None
    AWS_SECRET_KEY: str | None = None
    AWS_ARN_ROLE: str | None = None

    # --- Optional settings used to tweak the code. ---

    # AWS SageMaker
    HF_MODEL_ID: str = "mlabonne/TwinLlama-3.1-8B-DPO"
    GPU_INSTANCE_TYPE: str = "ml.g5.2xlarge"
    SM_NUM_GPUS: int = 1
    MAX_INPUT_LENGTH: int = 2048
    MAX_TOTAL_TOKENS: int = 4096
    MAX_BATCH_TOTAL_TOKENS: int = 4096
    COPIES: int = 1  # Number of replicas
    GPUS: int = 1  # Number of GPUs
    CPUS: int = 2  # Number of CPU cores

    SAGEMAKER_ENDPOINT_CONFIG_INFERENCE: str = "twin"
    SAGEMAKER_ENDPOINT_INFERENCE: str = "twin"
    TEMPERATURE_INFERENCE: float = 0.01
    TOP_P_INFERENCE: float = 0.9
    MAX_NEW_TOKENS_INFERENCE: int = 150

    # RAG
    TEXT_EMBEDDING_MODEL_ID: str = "sentence-transformers/all-MiniLM-L6-v2"
    RERANKING_CROSS_ENCODER_MODEL_ID: str = "cross-encoder/ms-marco-MiniLM-L-4-v2"
    RAG_MODEL_DEVICE: str = "cpu"

    # LinkedIn Credentials
    LINKEDIN_USERNAME: str | None = None
    LINKEDIN_PASSWORD: str | None = None

    @property
    def OPENAI_MAX_TOKEN_WINDOW(self) -> int:
        official_max_token_window = {
            "gpt-3.5-turbo": 16385,
            "gpt-4-turbo": 128000,
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
        }.get(self.OPENAI_MODEL_ID, 128000)

        max_token_window = int(official_max_token_window * 0.90)
        return max_token_window

    @classmethod
    def load_settings(cls) -> "Settings":
        """
        Tries to load the settings from the ZenML secret store.
        If the secret does not exist, it initializes the settings from the .env file and default values.

        Returns:
            Settings: The initialized settings object.
        """
        try:
            logger.info("Loading settings from the ZenML secret store.")
            settings_secrets = Client().get_secret("settings")
            settings = cls(**settings_secrets.secret_values)
            logger.info("Settings loaded successfully from ZenML.")
        except (RuntimeError, KeyError):
            logger.warning(
                "Failed to load settings from the ZenML secret store. "
                "Defaulting to loading the settings from the '.env' file."
            )
            settings = cls()
            logger.info("Settings loaded successfully from .env.")
        return settings

    def export(self) -> None:
        """
        Exports the settings to the ZenML secret store.
        """
        env_vars = self.model_dump()
        for key, value in env_vars.items():
            env_vars[key] = str(value)

        client = Client()

        try:
            client.create_secret(name="settings", values=env_vars)
            logger.info("Settings exported successfully to ZenML secret store.")
        except EntityExistsError:
            logger.warning(
                "Secret 'settings' already exists. "
                "Delete it manually by running 'zenml secret delete settings', before trying to recreate it."
            )

# Initialize Settings
settings = Settings.load_settings()

# -------------------------
# MongoDatabaseConnector Class
# -------------------------

class MongoDatabaseConnector:
    _instance: MongoClient | None = None

    def __new__(cls, *args, **kwargs) -> MongoClient:
        if cls._instance is None:
            try:
                cls._instance = MongoClient(settings.DATABASE_HOST)
                logger.info(f"Connected to MongoDB at {settings.DATABASE_HOST}")
            except ConnectionFailure as e:
                logger.error(f"Couldn't connect to the database: {e!s}")
                raise
        return cls._instance

# Create a single MongoDB connection instance
connection = MongoDatabaseConnector()
_database = connection.get_database(settings.DATABASE_NAME)

# -------------------------
# Install Chromedriver
# -------------------------

chromedriver_autoinstaller.install()

# -------------------------
# Define NoSQLBaseDocument
# -------------------------

T = TypeVar("T", bound="NoSQLBaseDocument")

class NoSQLBaseDocument(BaseModel, Generic[T], ABC):
    id: UUID4 = Field(default_factory=uuid.uuid4)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, self.__class__):
            return False
        return self.id == value.id

    def __hash__(self) -> int:
        return hash(self.id)

    @classmethod
    def from_mongo(cls: Type[T], data: dict) -> T:
        if not data:
            raise ValueError("Data is empty.")
        id = data.pop("_id")
        return cls(**dict(data, id=id))

    def to_mongo(self: T, **kwargs) -> dict:
        exclude_unset = kwargs.pop("exclude_unset", False)
        by_alias = kwargs.pop("by_alias", True)
        parsed = self.model_dump(exclude_unset=exclude_unset, by_alias=by_alias, **kwargs)
        if "_id" not in parsed and "id" in parsed:
            parsed["_id"] = str(parsed.pop("id"))
        for key, value in parsed.items():
            if isinstance(value, uuid.UUID):
                parsed[key] = str(value)
        return parsed

    def model_dump(self: T, **kwargs) -> dict:
        dict_ = super().model_dump(**kwargs)
        for key, value in dict_.items():
            if isinstance(value, uuid.UUID):
                dict_[key] = str(value)
        return dict_

    def save(self: T, **kwargs) -> T | None:
        collection = _database[self.get_collection_name()]
        try:
            collection.insert_one(self.to_mongo(**kwargs))
            logger.info(f"Document saved to {self.get_collection_name()} collection.")
            return self
        except errors.WriteError:
            logger.exception("Failed to insert document.")
            return None

    @classmethod
    def get_or_create(cls: Type[T], **filter_options) -> T:
        collection = _database[cls.get_collection_name()]
        try:
            instance = collection.find_one(filter_options)
            if instance:
                logger.info("Document found in the database.")
                return cls.from_mongo(instance)
            new_instance = cls(**filter_options)
            new_instance = new_instance.save()
            return new_instance
        except errors.OperationFailure:
            logger.exception(f"Failed to retrieve document with filter options: {filter_options}")
            raise

    @classmethod
    def bulk_insert(cls: Type[T], documents: List[T], **kwargs) -> bool:
        collection = _database[cls.get_collection_name()]
        try:
            collection.insert_many(doc.to_mongo(**kwargs) for doc in documents)
            logger.info(f"Bulk insert successful for {cls.__name__}.")
            return True
        except (errors.WriteError, errors.BulkWriteError):
            logger.error(f"Failed to insert documents of type {cls.__name__}")
            return False

    @classmethod
    def find(cls: Type[T], **filter_options) -> T | None:
        collection = _database[cls.get_collection_name()]
        try:
            instance = collection.find_one(filter_options)
            if instance:
                return cls.from_mongo(instance)
            return None
        except errors.OperationFailure:
            logger.error("Failed to retrieve document")
            return None

    @classmethod
    def bulk_find(cls: Type[T], **filter_options) -> List[T]:
        collection = _database[cls.get_collection_name()]
        try:
            instances = collection.find(filter_options)
            documents = [cls.from_mongo(instance) for instance in instances if instance]
            logger.info(f"Bulk find retrieved {len(documents)} documents.")
            return documents
        except errors.OperationFailure:
            logger.error("Failed to retrieve documents")
            return []

    @classmethod
    def get_collection_name(cls: Type[T]) -> str:
        if not hasattr(cls, "Settings") or not hasattr(cls.Settings, "name"):
            raise ImproperlyConfigured(
                "Document should define a Settings configuration class with the name of the collection."
            )
        return cls.Settings.name

# -------------------------
# Define VectorBaseDocument
# -------------------------

class VectorBaseDocument(NoSQLBaseDocument, ABC):
    """
    Abstract base class for documents stored in the Vector DB.
    """
    embedding: List[float] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def bulk_insert(cls: Type[T], documents: List[T], **kwargs) -> bool:
        """
        Inserts multiple documents into the Vector DB.

        Args:
            documents (List[T]): A list of document instances to insert.
            **kwargs: Additional keyword arguments.

        Returns:
            bool: True if insertion is successful, False otherwise.
        """
        try:
            logger.info(f"Inserting {len(documents)} documents into Vector DB.")
            from qdrant_client import QdrantClient
            client = QdrantClient(
                host=settings.QDRANT_DATABASE_HOST,
                port=settings.QDRANT_DATABASE_PORT,
                api_key=settings.QDRANT_APIKEY,
                prefer_grpc=True
            )
            vectors = [doc.embedding for doc in documents]
            payloads = [doc.metadata for doc in documents]
            client.upsert(
                collection_name=cls.get_collection_name(),
                points=[
                    {"id": str(doc.id), "vector": vector, "payload": payload}
                    for doc, vector, payload in zip(documents, vectors, payloads)
                ]
            )
            logger.info("Bulk insert into Vector DB successful.")
            return True
        except Exception as e:
            logger.error(f"Failed to insert documents into Vector DB: {e}")
            return False

# -------------------------
# Define EmbeddedDocument Classes
# -------------------------

class EmbeddedArticleChunk(NoSQLBaseDocument):
    text: str
    source: str
    # Add other fields as needed

class EmbeddedPostChunk(NoSQLBaseDocument):
    text: str
    source: str
    # Add other fields as needed

class EmbeddedRepositoryChunk(NoSQLBaseDocument):
    text: str
    source: str
    # Add other fields as needed

class EmbeddedQuery(NoSQLBaseDocument):
    query_text: str
    embedding: List[float]
    # Add other fields as needed

class DataCategory(NoSQLBaseDocument):
    category_name: str
    description: str
    # Add other fields as needed

# -------------------------
# Define Document and RepositoryDocument
# -------------------------

class Document(NoSQLBaseDocument, ABC):
    content: Dict[str, Any]
    platform: str
    author_id: UUID4 = Field(alias="author_id")
    author_full_name: str = Field(alias="author_full_name")

class RepositoryDocument(Document):
    name: str
    link: str

    class Settings:
        name = "repositories"

# -------------------------
# Define BaseCrawler and GithubCrawler
# -------------------------

class BaseCrawler(ABC):
    model: Type[NoSQLBaseDocument]

    @abstractmethod
    def extract(self, link: str, **kwargs) -> None:
        pass

class GithubCrawler(BaseCrawler):
    model = RepositoryDocument

    def __init__(self, ignore=(".git", ".toml", ".lock", ".png")) -> None:
        super().__init__()
        self._ignore = ignore
        self.driver = self.initialize_driver()

    def initialize_driver(self) -> webdriver.Chrome:
        options = Options()
        options.add_argument("--no-sandbox")
        options.add_argument("--headless=new")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--log-level=3")
        options.add_argument("--disable-popup-blocking")
        options.add_argument("--disable-notifications")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-background-networking")
        options.add_argument("--ignore-certificate-errors")
        options.add_argument(f"--user-data-dir={tempfile.mkdtemp()}")
        options.add_argument(f"--data-path={tempfile.mkdtemp()}")
        options.add_argument(f"--disk-cache-dir={tempfile.mkdtemp()}")
        options.add_argument("--remote-debugging-port=9226")

        self.set_extra_driver_options(options)

        driver = webdriver.Chrome(options=options)
        return driver

    def set_extra_driver_options(self, options: Options) -> None:
        """Override this method to set additional driver options."""
        pass

    def login(self) -> None:
        """Override this method to implement login functionality if required."""
        pass

    def extract(self, link: str, user: Dict) -> None:
        old_model = self.model.find(link=link)
        if old_model is not None:
            logger.info(f"Repository already exists in the database: {link}")
            return

        logger.info(f"Starting scraping GitHub repository: {link}")
        repo_name = link.rstrip("/").split("/")[-1]
        local_temp = tempfile.mkdtemp()

        try:
            os.chdir(local_temp)
            subprocess.run(["git", "clone", link], check=True)
            repo_path = os.path.join(local_temp, os.listdir(local_temp)[0])

            tree = {}
            for root, _, files in os.walk(repo_path):
                dir = root.replace(repo_path, "").lstrip("/")
                if any(dir.startswith(ignore) for ignore in self._ignore):
                    continue

                for file in files:
                    if any(file.endswith(ext) for ext in self._ignore):
                        continue
                    file_path = os.path.join(dir, file)
                    with open(os.path.join(root, file), "r", errors="ignore") as f:
                        tree[file_path] = f.read().replace(" ", "")

            instance = self.model(
                content=tree,
                name=repo_name,
                link=link,
                platform="github",
                author_id=user.get("id", uuid4()),
                author_full_name=user.get("full_name", "Unknown"),
            )
            instance.save()

        except subprocess.CalledProcessError as e:
            logger.exception(f"Failed to clone repository {link}: {e}")
            raise
        except Exception as e:
            logger.exception(f"Failed to scrape GitHub repository {link}: {e}")
            raise
        finally:
            shutil.rmtree(local_temp)

        logger.info(f"Finished scraping GitHub repository: {link}")