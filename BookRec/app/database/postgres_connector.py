import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv

load_dotenv()

class PostgresConnector:
    def __init__(self):
        self.host = os.getenv("POSTGRES_HOST", "localhost")
        self.port = os.getenv("POSTGRES_PORT", "5432")
        self.user = os.getenv("POSTGRES_USER", "postgres")
        self.password = os.getenv("POSTGRES_PASSWORD", "postgres_password")
        self.database = os.getenv("POSTGRES_DB", "booksdb")
        self.conn = None

    def connect(self):
        try:
            self.conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                dbname=self.database
            )
            print("Connected to PostgreSQL")
            return True
        except Exception as e:
            print(f"Failed to connect to PostgreSQL: {e}")
            return False

    def close(self):
        if self.conn:
            self.conn.close()
            print("PostgreSQL connection closed")

    def get_books_metadata(self, work_ids, filters=None):
        """
        Given a list of work_ids, return enriched metadata from the books and authors tables.
        Optionally filter by additional criteria provided in the 'filters' dictionary.
        Expected keys in filters:
          - max_pages: (int) Maximum number of pages.
          - min_pub_date: (date) Oldest acceptable publication date (books with publication_date >= min_pub_date).
          - is_ebook: (bool) If True, only include books available as ebook.
          - format: (str) Book format (e.g., "Paperback", "Hardcover", etc.).
          - min_average_rating: (float) Minimum average rating.
          - min_rating_count: (int) Minimum rating count.
        Returns a list of dictionaries with keys:
          title, isbn, num_pages, average_rating, publisher, description, link, publication_date, author_names.
        """
        if not self.conn:
            if not self.connect():
                return []

        query = """
        SELECT 
            b.title,
            b.isbn,
            b.num_pages,
            b.average_rating,
            b.publisher,
            b.description,
            b.link,
            b.publication_date,
            (
              SELECT string_agg(a.name, ', ')
              FROM authors a
              WHERE a.author_id = ANY(b.author_ids)
            ) AS author_names
        FROM books b
        WHERE b.work_id = ANY(%s)
        """
        params = [work_ids]

        if filters:
            if filters.get("max_pages") is not None:
                query += " AND b.num_pages <= %s"
                params.append(filters["max_pages"])
            if filters.get("min_pub_date") is not None:
                query += " AND b.publication_date >= %s"
                params.append(filters["min_pub_date"])
            if filters.get("is_ebook") is not None:
                query += " AND b.is_ebook = %s"
                params.append(filters["is_ebook"])
            if filters.get("format") is not None:
                query += " AND b.format = %s"
                params.append(filters["format"])
            if filters.get("min_average_rating") is not None:
                query += " AND b.average_rating >= %s"
                params.append(filters["min_average_rating"])
            if filters.get("min_rating_count") is not None:
                query += " AND b.ratings_count >= %s"
                params.append(filters["min_rating_count"])

        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                results = cur.fetchall()
                return results
        except Exception as e:
            print(f"Error querying PostgreSQL: {e}")
            return []
    
    def get_description_mapping(self):
        """
        Returns a dictionary mapping work_id (as an integer) to book description.
        """
        if not self.conn:
            if not self.connect():
                return {}
        query = "SELECT work_id, description FROM books"
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query)
                results = cur.fetchall()
                mapping = {int(row['work_id']): row['description'] for row in results}
                return mapping
        except Exception as e:
            print(f"Error querying description mapping: {e}")
            return {}
