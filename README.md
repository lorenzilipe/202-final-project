# DSC 202 Final Project

In the book crossing data-cleaning notebook, the clean_and_standardize_series() function is the one that I used to clean the book titles, authors, and user cities and countries... Make sure data has been processed with this function before trying to do any merges so that all da tings match!

# Qdrant Book Recomendation files

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

## How It Works

The system uses vector embeddings to represent book descriptions semantically. When you search with a text query, it:
1. Converts your query to a vector embedding
2. Finds the most similar book embeddings in the database
3. Returns the matching books as recommendations

This semantic search approach allows users to find relevant books using natural language descriptions rather than just keywords.
