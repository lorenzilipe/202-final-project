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

    def connect(self):
        """Connect to Qdrant and return success status and message"""
        
        # Validate configuration
        if not self.qdrant_url:
            return False, "Qdrant URL is not set. Check your .env file."
            
        if not self.api_key:
            return False, "Qdrant API key is not set. Check your .env file."
        
        print(f"Connecting to Qdrant at: {self.qdrant_url}")
        
        try:
            # Create client with appropriate timeout
            self.client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.api_key,
                timeout=30
            )
        
            # Test connection
            collections = self.client.get_collections()
            available_collections = [c.name for c in collections.collections]
            print(f"Connected to Qdrant. Available collections: {available_collections}")
            
            # Check if books collection exists
            if self.collection_name not in available_collections:
                return False, f"'{self.collection_name}' collection not found in Qdrant"
                
            return True, "Successfully connected to Qdrant"
            
        except Exception as e:
            error_message = f"Failed to connect to Qdrant: {str(e)}"
            print(error_message)
            return False, error_message
    
    def load_model(self):
        """Load the sentence transformer model for encoding queries"""
        try:
            self.model = SentenceTransformer(self.model_name)
            print(f"Model loaded: {self.model_name}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
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
            print("Not connected to Qdrant. Call connect() first.")
            return []
            
        if not self.model:
            model_loaded = self.load_model()
            if not model_loaded:
                print("Failed to load the embedding model.")
                return []
        
        try:
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
                print(f"No results found for query: '{query_text}'")
            else:
                print(f"Found {len(search_results)} similar books")
                
            return search_results
            
        except Exception as e:
            print(f"Error searching for books: {e}")
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
            # Extract book information from payload
            title = result.payload.get('title', 'Unknown')
            summary = result.payload.get('summary', 'No summary available')
            
            # Truncate long summaries
            if len(summary) > 200:
                summary = summary[:200] + "..."
                
            # Create a structured result
            formatted_result = {
                "rank": i+1,
                "score": round(result.score, 4),
                "title": title,
                "summary": summary
            }
            formatted_results.append(formatted_result)
            
        return formatted_results
    
    def debug_collection(self):
        """
        Debug the collection to see if it contains data
        
        Returns:
        - Dictionary with debug information
        """
        debug_info = {}
        try:
            # Check collection info
            collection_info = self.client.get_collection(self.collection_name)
            debug_info["vectors_count"] = collection_info.vectors_count
            debug_info["status"] = collection_info.status
            
            # Get vector dimensions
            debug_info["vector_size"] = collection_info.config.params.vectors.size
            debug_info["distance"] = str(collection_info.config.params.vectors.distance)
            
            # Get some sample points
            sample_points = self.client.scroll(
                collection_name=self.collection_name,
                limit=2
            )
            
            if sample_points and sample_points[0]:
                debug_info["has_points"] = True
                debug_info["sample_count"] = len(sample_points[0])
                
                # Check payload structure of first point
                if sample_points[0]:
                    first_point = sample_points[0][0]
                    debug_info["sample_id"] = first_point.id
                    debug_info["sample_payload_keys"] = list(first_point.payload.keys()) if first_point.payload else []
                    debug_info["has_title"] = "title" in first_point.payload
                    debug_info["has_summary"] = "summary" in first_point.payload
            else:
                debug_info["has_points"] = False
                
            return debug_info
        except Exception as e:
            return {"error": str(e)}