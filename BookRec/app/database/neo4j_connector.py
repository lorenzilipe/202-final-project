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