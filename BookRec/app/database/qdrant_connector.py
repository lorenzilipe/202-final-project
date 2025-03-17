from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
import socket

class QdrantConnector:
    def __init__(self):
        load_dotenv()
        
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.api_key = os.getenv("QDRANT_API_KEY")
        self.client = None
        self.model = None
        self.model_name = "all-MiniLM-L6-v2"  # Default model
        self.collection_name = "books"  # Fixed collection name
        
        # We don't store debug_area anymore, we'll use the log function that's overridden from main.py

    def log(self, message, level="info"):
        """
        Log message - this will be overridden by main.py
        Default implementation just prints to console
        """
        print(f"[{level.upper()}] {message}")

    def connect(self):
        """Connect to Qdrant and return success status and message"""
        
        # Validate configuration
        if not self.qdrant_url:
            self.log("Qdrant URL is not set. Check your .env file.", level="error")
            return False, "Qdrant URL is not set. Check your .env file."
            
        if not self.api_key:
            self.log("Qdrant API key is not set. Check your .env file.", level="error")
            return False, "Qdrant API key is not set. Check your .env file."
        
        self.log(f"Connecting to Qdrant at: {self.qdrant_url}")
        
        try:
            # Create client with appropriate timeout
            self.client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.api_key,
                timeout=30,
                prefer_grpc=False  # Use HTTP API to avoid some validation errors
            )
        
            # Test connection
            try:
                # First try a simple health check
                health_status = self.client.http.health()
                self.log("Qdrant health check successful", level="success")
            except Exception as health_err:
                self.log(f"Health check failed, but will continue: {str(health_err)}", level="warning")
            
            # Get collections list
            collections = self.client.get_collections()
            available_collections = [c.name for c in collections.collections]
            self.log(f"Connected to Qdrant. Available collections: {available_collections}", level="success")
            
            # Check if books collection exists
            if self.collection_name not in available_collections:
                self.log(f"'{self.collection_name}' collection not found in Qdrant", level="warning")
                return False, f"'{self.collection_name}' collection not found in Qdrant"
                
            return True, "Successfully connected to Qdrant"
            
        except Exception as e:
            error_message = f"Failed to connect to Qdrant: {str(e)}"
            self.log(error_message, level="error")
            return False, error_message
    
    def load_model(self):
        """Load the sentence transformer model for encoding queries"""
        try:
            self.log("Loading embedding model...", level="info")
            self.model = SentenceTransformer(self.model_name)
            self.log(f"Model loaded: {self.model_name}", level="success")
            return True
        except Exception as e:
            self.log(f"Error loading model: {e}", level="error")
            return False
    
    def search_similar_books(self, query_text, limit=5):
        """
        Search for books similar to the query text
        
        Parameters:
        - query_text: Text to search for similar books
        - limit: Maximum number of results to return
        
        Returns:
        - List of search results with title, summary and similarity score
        """
        if not self.client:
            self.log("Not connected to Qdrant. Call connect() first.", level="error")
            return []
            
        if not self.model:
            model_loaded = self.load_model()
            if not model_loaded:
                self.log("Failed to load the embedding model.", level="error")
                return []
        
        try:
            self.log(f"Searching for books similar to: '{query_text}'", level="info")
            
            # Encode the query text
            query_vector = self.model.encode(query_text).tolist()
            
            # Search for similar books
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=0.0  # No threshold to get all results
            )
            
            # Check if we got results
            if not search_results:
                self.log(f"No results found for query: '{query_text}'", level="warning")
            else:
                self.log(f"Found {len(search_results)} similar books", level="success")
                
            return search_results
            
        except Exception as e:
            self.log(f"Error searching for books: {e}", level="error")
            return []
    
    def format_results(self, results):
        """
        Format search results into a readable structure
        
        Parameters:
        - results: Search results from Qdrant
        
        Returns:
        - List of dictionaries with formatted book information
        """
        if not results:
            return []
        
        formatted_results = []
        for i, result in enumerate(results):
            print("DEBUG: Qdrant payload:", result.payload)

            # Extract book information from payload
            title = result.payload.get('title', 'Unknown')
            summary = result.payload.get('summary', 'No summary available')
            # Extract work_id from payload (if available)
            work_id = result.payload.get('work_id')
            
            # Truncate long summaries
            if len(summary) > 200:
                summary = summary[:200] + "..."
                
            # Create a structured result
            formatted_result = {
                "rank": i+1,
                "score": round(result.score, 4),
                "title": title,
                "summary": summary,
                "work_id": work_id
            }
            formatted_results.append(formatted_result)
            
        return formatted_results