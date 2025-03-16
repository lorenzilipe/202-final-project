from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
import socket

class QdrantConnector:
    def __init__(self, debug_area=None):
        load_dotenv()
        
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.api_key = os.getenv("QDRANT_API_KEY")
        self.client = None
        self.model = None
        self.model_name = "all-MiniLM-L6-v2"  # Default model
        self.collection_name = "books"  # Fixed collection name
        self.debug_area = debug_area  # Store reference to debug area

    def log(self, message, level="info"):
        """Log message to Streamlit if debug area exists, otherwise print"""
        print(message)  # Always print to console
        
        if self.debug_area:
            # small delay using st.empty() to avoid state conflicts
            import time
            time.sleep(0.1)  # Small delay to prevent message collisions
            
            # Use the appropriate Streamlit method based on level
            if level == "error":
                self.debug_area.error(message)
            elif level == "warning":
                self.debug_area.warning(message)
            elif level == "success":
                self.debug_area.success(message)
            else:
                self.debug_area.info(message)

    def connect(self):
        """Connect to Qdrant and return success status and message"""
        
        # Validate configuration
        if not self.qdrant_url:
            self.log("Qdrant URL is not set. Check your .env file.", level="error")
            return False, "Qdrant URL is not set. Check your .env file."
            
        if not self.api_key:
            self.log("Qdrant API key is not set. Check your .env file.", level="error")
            return False, "Qdrant API key is not set. Check your .env file."
        
        print(f"Connecting to Qdrant at: {self.qdrant_url}")
        
        try:
            # Create client with appropriate timeout
            self.client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.api_key,
                timeout=30,
                prefer_grpc=False  # Use HTTP API to avoid some validation errors
            )
        
            # Test connection using a simpler method that's less likely to cause validation errors
            try:
                # First try a simple health check which is less likely to have schema issues
                health_status = self.client.http.health()
                self.log("Qdrant health check successful", level="success")
            except Exception as health_err:
                self.log(f"Health check failed, but will try to continue: {str(health_err)}", level="warning")
            
            # Get collections list
            try:
                collections_list = self.client.http.collections_api.get_collections()
                available_collections = [c.name for c in collections_list.collections]
                self.log(f"Connected to Qdrant. Available collections: {available_collections}", level="success")
                
                # Check if books collection exists
                if self.collection_name not in available_collections:
                    self.log(f"'{self.collection_name}' collection not found in Qdrant", level="warning")
                    return False, f"'{self.collection_name}' collection not found in Qdrant"
                
                # If we got here, we have a working connection
                return True, "Successfully connected to Qdrant"
                
            except Exception as list_err:
                # If we can't list collections, try a direct check for our collection
                self.log(f"Failed to list collections: {str(list_err)}", level="warning")
                
                try:
                    # Try to directly check if our collection exists
                    collection_exists = self.client.http.collections_api.collection_exists(self.collection_name)
                    if collection_exists:
                        self.log(f"Collection '{self.collection_name}' exists", level="success")
                        return True, f"Connected to Qdrant. Collection '{self.collection_name}' exists."
                    else:
                        self.log(f"Collection '{self.collection_name}' does not exist", level="warning")
                        return False, f"Collection '{self.collection_name}' does not exist"
                except Exception as exists_err:
                    self.log(f"Failed to check collection existence: {str(exists_err)}", level="error")
                    # Continue to search test as a last resort
            
            # If we couldn't verify the collection, try a simple search as a last resort
            try:
                self.log("Attempting direct search as a connection test...", level="info")
                if not self.model:
                    self.load_model()
                test_vector = self.model.encode("test query").tolist()
                
                # Perform a minimal search with just one result
                search_result = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=test_vector,
                    limit=1
                )
                self.log("Search test successful, connection is working", level="success")
                return True, "Connected to Qdrant (verified through search)"
            
            except Exception as search_err:
                error_msg = str(search_err)
                
                # This is a special case - if we get a 404 on search, the collection doesn't exist
                if "404" in error_msg:
                    self.log(f"Collection '{self.collection_name}' not found", level="error")
                    return False, f"Collection '{self.collection_name}' not found"
                
                self.log(f"Search test failed: {error_msg}", level="error")
                return False, f"Failed to verify connection: {error_msg}"
            
        except Exception as e:
            error_message = f"Failed to connect to Qdrant: {str(e)}"
            self.log(error_message, level="error")
            
            if "validation errors" in str(e).lower():
                self.log("Validation error detected - this might be due to a version mismatch between client and server", level="warning")
                
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