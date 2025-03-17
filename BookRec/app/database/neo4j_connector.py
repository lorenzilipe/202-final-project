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
        // Step 1: Get target user's ratings for common books
        MATCH (target:User {user_id: "temp_user"})-[r:INTERACTED]->(b:Book)
        WITH target, collect({work_id: b.work_id, rating: r.rating}) AS targetRatings
        // Step 2: Find common books between target and other users and compute similarity per common book
        MATCH (target)-[r1:INTERACTED]->(b:Book)<-[:INTERACTED]-(other:User)
        WHERE other.user_id <> target.user_id
        WITH target, targetRatings, other, b.work_id AS commonBook, r1.rating AS otherRating,
                head([t IN targetRatings WHERE t.work_id = b.work_id]) AS targetRating
        WITH target, targetRatings, other, (1 - abs(targetRating.rating - otherRating)/4.0) AS simScore
        // Step 3: For each similar user, average the similarity scores over common books
        WITH target, targetRatings, other, collect(simScore) AS simScores, count(*) AS commonCount
        WHERE commonCount >= 2
        WITH target, targetRatings, other, reduce(s = 0.0, x IN simScores | s + x) / size(simScores) AS user_similarity
        // Step 4: Get candidate recommendations from similar users (books not rated by target)
        MATCH (other)-[r2:INTERACTED]->(rec:Book)
        WHERE NOT rec.work_id IN [t IN targetRatings | t.work_id]
        WITH rec, user_similarity, r2.rating AS candidateRating
        // Step 5: For each candidate book, sum weighted ratings from all similar users and compute average
        WITH rec, sum(user_similarity * candidateRating) AS weightedSum, count(candidateRating) AS ratingCount
        RETURN rec.work_id AS work_id, rec.title AS title, weightedSum / ratingCount AS cf_score
        ORDER BY cf_score DESC
        LIMIT 10
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
