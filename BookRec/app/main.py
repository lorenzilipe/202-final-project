import streamlit as st
from database.neo4j_connector import Neo4jConnector
from database.qdrant_connector import QdrantConnector

# Set page configuration
st.set_page_config(
    page_title="Book Recommender",
    page_icon="📚",
    layout="wide"
)

# Create a more stable way to handle debug messages
if "debug_messages" not in st.session_state:
    st.session_state.debug_messages = []

# Function to log messages in a stable way
def log_to_debug(message, level="info"):
    st.session_state.debug_messages.append({"message": message, "level": level})
    print(message)  # Also print to console

# Initialize database connections
@st.cache_resource
def init_connections():
    neo4j_conn = Neo4jConnector()
    # Don't pass the debug_area directly, we'll use our custom logging instead
    qdrant_conn = QdrantConnector()
    return neo4j_conn, qdrant_conn

neo4j_conn, qdrant_conn = init_connections()

# Override Qdrant connector's log method
qdrant_conn.log = log_to_debug

# Set up the UI
st.title("Book Recommender System")

# Create two columns for database status
col1, col2 = st.columns(2)

# Neo4j Column
with col1:
    st.header("Neo4j Database")
    if st.button("Test Neo4j Connection", key="neo4j_test"):
        if neo4j_conn.connect():
            st.success("Neo4j Connection Successful")
            log_to_debug("Connected to Neo4j successfully", "success")
        else:
            st.error("Neo4j Connection Failed")
            log_to_debug("Failed to connect to Neo4j", "error")

# Qdrant Column
with col2:
    st.header("Qdrant Cloud")
    if st.button("Test Qdrant Connection", key="qdrant_test"):
        status, message = qdrant_conn.connect()
        if status:
            st.success("Qdrant Connection Successful")
            log_to_debug("Connected to Qdrant successfully", "success")
        else:
            st.error(f"Qdrant Connection Failed: {message}")
            log_to_debug(f"Failed to connect to Qdrant: {message}", "error")

# Book search section
st.header("Book Search")
st.write("Search for books by describing what you're looking for")

# Simple form for search
with st.form("book_search_form"):
    query = st.text_input("What kind of book are you looking for?", 
                          placeholder="I'm looking for books about artificial intelligence")
    
    num_results = st.slider("Number of results:", min_value=1, max_value=20, value=5)
    
    submit_button = st.form_submit_button("Search Books")

# Process search when form is submitted
if submit_button:
    if not query:
        st.warning("Please enter a search query")
    else:
        # Connect to Qdrant if not already connected
        if not hasattr(qdrant_conn, 'client') or qdrant_conn.client is None:
            status, message = qdrant_conn.connect()
            if not status:
                st.error(f"Failed to connect to Qdrant: {message}")
                st.stop()
        
        # Load the model if not already loaded
        if not hasattr(qdrant_conn, 'model') or qdrant_conn.model is None:
            with st.spinner("Loading embedding model..."):
                if not qdrant_conn.load_model():
                    st.error("Failed to load embedding model")
                    st.stop()
        
        # Perform the search
        with st.spinner("Searching for books..."):
            results = qdrant_conn.search_similar_books(
                query_text=query,
                limit=num_results
            )
            
            formatted_results = qdrant_conn.format_results(results)
            
            if formatted_results:
                st.success(f"Found {len(formatted_results)} books that match your query")
                
                # Display results in cards
                for result in formatted_results:
                    with st.container():
                        col1, col2 = st.columns([1, 4])
                        with col1:
                            st.metric("Score", f"{result['score']:.2f}")
                        with col2:
                            st.markdown(f"### {result['title']}")
                            st.markdown(f"{result['summary']}")
                        st.divider()
            else:
                st.warning("No matching books found. Try a different search.")

# Debug messages expander
with st.expander("Debug Messages", expanded=False):
    if not st.session_state.debug_messages:
        st.write("No debug messages yet. Click the test buttons above to see connection debug messages.")
    else:
        # Display debug messages in reverse order (newest first)
        for msg in reversed(st.session_state.debug_messages):
            if msg["level"] == "error":
                st.error(msg["message"])
            elif msg["level"] == "warning":
                st.warning(msg["message"])
            elif msg["level"] == "success":
                st.success(msg["message"])
            else:
                st.info(msg["message"])