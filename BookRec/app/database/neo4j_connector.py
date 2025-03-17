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
        if not self.driver:
            print("Driver not initialized. Call connect() first.")
            return []
        
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            print(f"Query execution error: {str(e)}")
            return []

    def get_collaborative_recommendations(self, user_id, min_rating=4.0, min_common_books=2, limit=10):
        """
        Returns collaborative filtering recommendations based on the user's ratings.
        The query assumes that each user has INTERACTED relationships with Book nodes, 
        with a 'rating' property.
        """
        cf_query = """
        // Step 1: Get books the target user rated highly.
        MATCH (target:User {user_id: $user_id})-[r:INTERACTED]->(b:Book)
        WHERE r.rating >= $min_rating
        WITH target, collect(b.work_id) AS targetBooks

        // Step 2: Find similar users who also rated these books highly.
        MATCH (target)-[r:INTERACTED]->(b:Book)<-[:INTERACTED]-(other:User)
        WHERE r.rating >= $min_rating AND other.user_id <> target.user_id
        WITH target, other, count(b) AS commonBooks, targetBooks
        WHERE commonBooks >= $min_common_books

        // Step 3: Retrieve additional books that these similar users rated highly.
        MATCH (other)-[r2:INTERACTED]->(rec:Book)
        WHERE r2.rating >= $min_rating AND NOT rec.work_id IN targetBooks
        WITH rec, count(DISTINCT other) AS similarUserCount, avg(r2.rating) AS avgRating
        RETURN rec.work_id AS work_id, rec.title AS title, similarUserCount, avgRating,
               (similarUserCount * avgRating) AS cf_score
        ORDER BY cf_score DESC
        LIMIT $limit
        """
        params = {
            "user_id": user_id,
            "min_rating": min_rating,
            "min_common_books": min_common_books,
            "limit": limit
        }
        return self.execute_query(cf_query, params)

    def get_all_book_titles(self, limit=1000):
        """
        Returns a list of all book titles and their work_ids from the database.
        """
        query = "MATCH (b:Book) RETURN b.title AS title, b.work_id AS work_id LIMIT $limit"
        return self.execute_query(query, {"limit": limit})
    
    def insert_user_ratings(self, user_id, rated_books_data):
        """
        Inserts ratings for a given user into the database.
        rated_books_data should be a dictionary where each key is a work_id and each value is a dictionary
        with at least a 'rating' key.
        """
        query = """
        UNWIND $ratings AS ratingData
        MERGE (u:User {user_id: $user_id})
        WITH u, ratingData
        MATCH (b:Book {work_id: ratingData.work_id})
        MERGE (u)-[r:INTERACTED]->(b)
        SET r.rating = ratingData.rating
        """
        params = {
            "user_id": user_id,
            "ratings": [{"work_id": work_id, "rating": data["rating"]} for work_id, data in rated_books_data.items()]
        }
        return self.execute_query(query, params)

    def clear_temp_user(self, user_id):
        """
        Clears the temporary user's data by deleting the user node and all its relationships.
        """
        query = """
        MATCH (u:User {user_id: $user_id})
        DETACH DELETE u
        """
        return self.execute_query(query, {"user_id": user_id})
