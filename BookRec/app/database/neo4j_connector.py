from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

class Neo4jConnector:
    def __init__(self):
        load_dotenv()
        
        self.uri = os.getenv("NEO4J_URI")
        self.user = os.getenv("NEO4J_USER")
        self.password = os.getenv("NEO4J_PASSWORD")
        self.driver = None

    def connect(self):
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.user, self.password)
            )
            print("Connected to Neo4j database")
            return True
        except Exception as e:
            print(f"Failed to connect to Neo4j database: {e}")
            return False
    
    def close(self):
        if self.driver is not None:
            self.driver.close()
            print("Neo4j connection closed")
    
    def execute_query(self, query, parameters=None):
        assert self.driver is not None, "Driver not initialized. Call connect() first."
        
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]
            
    def get_book_info(self, work_id):
        """Get information about a specific book"""
        query = """
        MATCH (b:Book {work_id: $work_id})
        RETURN b.work_id as work_id, b.title as title, b.average_rating as average_rating, b.ratings_count as ratings_count
        """
        results = self.execute_query(query, {"work_id": work_id})
        return results[0] if results else None


    #This is the funtion that is called for the collaberative filtering recommendations, currenly it always returns empty :(
    # need to be fixed!!!!! 
    def get_similar_books_cf(self, book_ids, limit=5):
        """
        Get book recommendations based on collaborative filtering
        
        Parameters:
        - book_ids: List of book IDs the user likes
        - limit: Maximum number of recommendations to return
        """
        # Check if we only have one book selected
        if len(book_ids) == 1:
            # More relaxed query when only one book is selected
            query = """
            // Match users who read the selected book
            MATCH (user:User)-[r:INTERACTED]->(seed:Book)
            WHERE seed.work_id IN $book_ids
            
            // Find other books these users read
            MATCH (user)-[r2:INTERACTED]->(recommendation:Book)
            WHERE NOT recommendation.work_id IN $book_ids
            
            // Group by book and count how many users who read the seed book also read this book
            WITH recommendation, COUNT(DISTINCT user) as userOverlap, AVG(r2.rating) as avgRating
            
            // Return recommended books ordered by overlap count
            RETURN 
                recommendation.work_id as work_id,
                recommendation.title as title,
                recommendation.average_rating as average_rating,
                recommendation.ratings_count as ratings_count,
                avgRating as user_rating,
                userOverlap as score
            ORDER BY score DESC, avgRating DESC
            LIMIT $limit
            """
        else:
            # Original query for multiple books, but with rating threshold lowered
            query = """
            // Match users who interacted with the input books
            MATCH (user:User)-[r:INTERACTED]->(seed:Book)
            WHERE seed.work_id IN $book_ids AND r.rating >= 3
            
            // Find other books these users interacted with
            MATCH (user)-[r2:INTERACTED]->(recommendation:Book)
            WHERE r2.rating >= 3
            AND NOT recommendation.work_id IN $book_ids
            
            // Group by book and count how many users who liked the seed books also liked this book
            WITH recommendation, COUNT(DISTINCT user) as userOverlap, AVG(r2.rating) as avgRating
            
            // Return recommended books ordered by overlap count
            RETURN 
                recommendation.work_id as work_id,
                recommendation.title as title,
                recommendation.average_rating as average_rating,
                recommendation.ratings_count as ratings_count,
                avgRating as user_rating,
                userOverlap as score
            ORDER BY score DESC, avgRating DESC
            LIMIT $limit
            """
        
        return self.execute_query(query, {"book_ids": book_ids, "limit": limit})
    
    def format_cf_results(self, results):
        """Format collaborative filtering results for display"""
        formatted_results = []
        for i, result in enumerate(results):
            # Create a structured result
            formatted_result = {
                "rank": i+1,
                "book_id": result["work_id"],
                "score": result["score"],
                "title": result["title"],
                "average_rating": result.get("average_rating", 0),
                "ratings_count": result.get("ratings_count", 0),
                "user_rating": result.get("user_rating", 0),
                "summary": f"Average Rating: {result.get('average_rating', 'N/A')} from {result.get('ratings_count', 0)} ratings",
                "source": "collaborative"
            }
            formatted_results.append(formatted_result)
            
        return formatted_results
            
    def get_user_reading_history(self, user_id, limit=10):
        """Get books that a user has interacted with"""
        query = """
        // Find all books rated by this user
        MATCH (user:User {user_id: $user_id})-[r:INTERACTED]->(book:Book)
        
        // Return books with rating information
        RETURN 
            book.work_id as work_id,
            book.title as title,
            r.rating as rating,
            book.average_rating as average_rating,
            book.ratings_count as ratings_count
        ORDER BY r.rating DESC
        LIMIT $limit
        """
        
        return self.execute_query(query, {"user_id": user_id, "limit": limit})
        
    def get_popular_books(self, limit=10):
        """Get the most popular books based on ratings count"""
        query = """
        // Match all books
        MATCH (book:Book)
        
        // Only include books with sufficient ratings
        WHERE book.ratings_count >= 1000
        
        // Return popular books
        RETURN 
            book.work_id as work_id,
            book.title as title,
            book.average_rating as average_rating,
            book.ratings_count as ratings_count,
            book.ratings_count * book.average_rating as score
        ORDER BY score DESC
        LIMIT $limit
        """
        
        return self.execute_query(query, {"limit": limit})