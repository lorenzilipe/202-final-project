# DSC 202 Final Project

**Link to presentation video:** [https://www.youtube.com/watch?v=pKXsPabvcv0](https://www.youtube.com/watch?v=pKXsPabvcv0)

**Link to presentation slideshow:** [https://docs.google.com/presentation/d/1o671vR0r_dYCRIOZDwQTxu5LLe__tNdmbrrKeJ7KW0s/edit?usp=sharing](https://docs.google.com/presentation/d/1o671vR0r_dYCRIOZDwQTxu5LLe__tNdmbrrKeJ7KW0s/edit?usp=sharing)

---

# Installation

## Downloads
1. Clone the repository into a folder.
2. Download [Docker Desktop](https://www.docker.com/products/docker-desktop/) and open it.
3. (Optional) Download [UCSD Goodreads Dataset (Comics & Graphic)](https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html#:~:text=goodreads_reviews_children.json.gz-,Comics%20%26%20Graphic,-\(89%2C411%20books%2C%207%2C347%2C630)

## Cleaning data
Since the clean data is too large to store on GitHub, you must do it yourself.

1. Run all cells in ```data-processing/goodreads_data_pipeline.ipynb```, ```data-processing/interactions_work_id.ipynb``` and ```data-processing/goodreads_data_pipeline_postgres```

This will create some .csv files within ```./data-processing/data/```.

## Moving clean data to app container
After running the data cleaning notebooks in the step above, do the following:

1. Move ```data-processing/data/goodreads_books_comics_graphic_cleaned_neo4j.csv``` and ```data-processing/data/goodreads_interactions_comics_graphic_cleaned.csv``` into ```BookRec/neo4j_import/```
2. Move ```data-processing/data/goodreads_books_cleaned.csv``` and ```data-processing/data/goodreads_authors_cleaned.csv``` into ```BookRec/postgres_init/```

## Loading data into Neo4j
Now that the data can be detected by the container, we must upload it to the Neo4j db

First, run (from the ```./BookRec/``` directory):

```shell
> docker-compose up -d
```

Then, navigate to [http://localhost:7474/browser/](http://localhost:7474/browser/) and enter the username: ```neo4j``` and password: ```dsc202friends```.

Finally, run the following code block in the neo4j terminal:

```cypher
CREATE CONSTRAINT book_unique IF NOT EXISTS 
FOR (b:Book) REQUIRE b.work_id IS UNIQUE;

CREATE CONSTRAINT user_unique IF NOT EXISTS 
FOR (u:User) REQUIRE u.user_id IS UNIQUE;

LOAD CSV WITH HEADERS FROM 'file:///goodreads_books_comics_graphic_cleaned_neo4j.csv' AS row
MERGE (b:Book {work_id: row.work_id})
SET b.title = row.title,
    b.authors = row.authors,
    b.ratings_count = toInteger(row.ratings_count),
    b.average_rating = toFloat(row.average_rating);

:auto CALL {
  LOAD CSV WITH HEADERS FROM 'file:///goodreads_interactions_comics_graphic_cleaned.csv' AS row
  MERGE (u:User {user_id: row.user_id})
  WITH row, u
  MATCH (b:Book {work_id: row.work_id})
  MERGE (u)-[r:INTERACTED]->(b)
  SET r.rating = toFloat(row.rating),
      r.timestamp = row.timestamp
  RETURN count(*) AS batchCount
} IN TRANSACTIONS OF 500 ROWS
RETURN "Completed" AS status;
```

---

# Usage

First, ensure that the Docker container is running using ```docker-compose up -d```. If not, see the Installation section above.

Navigate to the application site at [http://localhost:8501/](http://localhost:8501/).

Follow the on-screen instructions and get your book recommendations!

---

# Qdrant Book Recomendation files

While it is not necessary to run these files to setup and use our application yourself, here is an explanation of how the Qdrant system works.

## Folder Structure

The `qdrant/bookrec_qdrant` folder contains three main files for the book recommendation system:

#### `create_qdrant.py`
- **Purpose**: Initializes a Qdrant vector database from a pandas DataFrame
- **Usage**: Only needed if the Qdrant database does not already exist
- **Note**: Since our database is already set up, you typically won't need to run this file

#### `process_data.py`
- **Purpose**: Prepares and processes data for the Qdrant database
- **Usage**: Generates the data that `create_qdrant.py` uses to populate the database
- **Note**: This file will likely be modified to use the final version of our dataset

#### `qdrant_search.py`
- **Purpose**: Core search functionality for book recommendations
- **Usage**: Performs semantic search queries against the Qdrant database
- **Features**:
  - Connects to the Qdrant database using API key and URL
  - Enables semantic search with natural language queries
  - Returns relevant book recommendations based on query similarity
- **Integration**: This will be incorporated into the final book recommendation function

# Postgres

Use the ```data_pipeline_postgres``` file in the data processing folder to obtain the clean data. The resulting data will be ```data/goodreads_books_cleaned_postgres.csv``` and ```data/goodreads_authors_cleaned.csv```.

Then use the schema in ```postgres_schema.txt``` as your schema when importing data into postgres.
