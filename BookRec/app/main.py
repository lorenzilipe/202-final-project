import streamlit as st
from database.neo4j_connector import Neo4jConnector
from database.qdrant_connector import QdrantConnector

# Set page configuration
st.set_page_config(
    page_title="Graph & Vector Database Demo",
    page_icon="🔍",
    layout="wide"
)

# Initialize database connections
@st.cache_resource
def init_connections():
    neo4j_conn = Neo4jConnector()
    neo4j_status = neo4j_conn.connect()
    
    qdrant_conn = QdrantConnector()
    qdrant_status = qdrant_conn.connect()
    
    return neo4j_conn, qdrant_conn, neo4j_status, qdrant_status

neo4j_conn, qdrant_conn, neo4j_status, qdrant_status = init_connections()

# App header
st.title("Graph & Vector Database Integration")
st.write("A demo application connecting Neo4j and Qdrant")

# Display connection status
col1, col2 = st.columns(2)
with col1:
    st.subheader("Neo4j Status")
    if neo4j_status:
        st.success("Connected to Neo4j")
    else:
        st.error("Not connected to Neo4j")

with col2:
    st.subheader("Qdrant Status")
    if qdrant_status:
        st.success("Connected to Qdrant")
    else:
        st.error("Not connected to Qdrant")

# Neo4j section
st.header("Neo4j Graph Database")
with st.expander("Run Cypher Query"):
    cypher_query = st.text_area("Enter Cypher Query:", value="MATCH (n) RETURN n LIMIT 10")
    if st.button("Execute Neo4j Query"):
        if neo4j_status:
            results = neo4j_conn.execute_query(cypher_query)
            st.json(results)
        else:
            st.error("Neo4j is not connected")

# Qdrant section
st.header("Qdrant Vector Database")

# Book search section
with st.expander("Book Search"):
    st.subheader("Search for Similar Books")
    
    query = st.text_input("What kind of book are you looking for?", 
                     placeholder="I'm looking for books about artificial intelligence")
    num_results = st.slider("Number of results:", min_value=1, max_value=20, value=5)

    if st.button("Search Books"):
        if qdrant_status:
            # Make sure model is loaded
            if not qdrant_conn.model:
                with st.spinner("Loading model..."):
                    qdrant_conn.load_model()
            
            # Perform search
            if query:
                with st.spinner("Searching..."):
                    results = qdrant_conn.search_similar_books(
                        query_text=query,
                        limit=num_results
                    )
                    
                    formatted_results = qdrant_conn.format_results(results)
                    
                    if formatted_results:
                        for result in formatted_results:
                            with st.container():
                                st.markdown(f"### {result['title']}")
                                st.markdown(f"**Similarity Score:** {result['score']}")
                                st.markdown(f"{result['summary']}")
                                st.divider()
                    else:
                        st.warning("No matching books found. Try a different search.")
            else:
                st.warning("Please enter a search query")
        else:
            st.error("Database connection failed. Check Qdrant settings.")

# Add a debug section
with st.expander("Debug Collection"):
    if qdrant_status:
        if st.button("Debug Collection"):
            debug_info = qdrant_conn.debug_collection()
            st.json(debug_info)
            
            if debug_info.get("vectors_count", 0) == 0:
                st.error("Your collection exists but contains no vectors!")
            elif not debug_info.get("has_points", False):
                st.error("Cannot retrieve points from collection!")
            elif not debug_info.get("has_title", False) or not debug_info.get("has_summary", False):
                st.error("Points are missing expected payload fields!")
            else:
                st.success("Collection structure looks valid!")
                
            # Test a zero-threshold search
            if st.button("Try Zero-Threshold Search"):
                test_query = "book"
                st.write(f"Testing generic query: '{test_query}'")
                results = qdrant_conn.search_similar_books(test_query, limit=10)
                if results:
                    st.success(f"Found {len(results)} results!")
                    st.json(qdrant_conn.format_results(results))
                else:
                    st.error("Still no results with zero threshold.")
    else:
        st.error("Connect to Qdrant first")