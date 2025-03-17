import streamlit as st
from database.neo4j_connector import Neo4jConnector
from database.qdrant_connector import QdrantConnector

# Set page configuration
st.set_page_config(
    page_title="Book Recommender",
    page_icon="📚",
    layout="wide"
)

# --------------------------
# Debug Logging Functionality
# --------------------------
if "debug_messages" not in st.session_state:
    st.session_state.debug_messages = []

def log_to_debug(message, level="info"):
    st.session_state.debug_messages.append({"message": message, "level": level})
    print(message)

# --------------------------
# Initialize Database Connections
# --------------------------
@st.cache_resource
def init_connections():
    neo4j_conn = Neo4jConnector()
    qdrant_conn = QdrantConnector()
    return neo4j_conn, qdrant_conn

neo4j_conn, qdrant_conn = init_connections()
qdrant_conn.log = log_to_debug  # Override Qdrant connector logging

# --------------------------
# Header and Database Test Buttons
# --------------------------
st.title("Book Recommender System")
st.write("This demo integrates Collaborative Filtering (via Neo4j) and Semantic Search (via Qdrant).")

col1, col2 = st.columns(2)
with col1:
    st.header("Neo4j Database")
    if st.button("Test Neo4j Connection", key="neo4j_test"):
        if neo4j_conn.connect():
            st.success("Neo4j Connection Successful")
            log_to_debug("Connected to Neo4j successfully", "success")
        else:
            st.error("Neo4j Connection Failed")
            log_to_debug("Failed to connect to Neo4j", "error")
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

# --------------------------
# User ID Input (Displayed on Main Page)
# --------------------------
st.header("User Identification")
user_id = st.text_input("Enter your User ID (if new, leave blank):")
if not user_id:
    user_id = "temp_user"

# --------------------------
# Mode Selection: "Rate Books" or "Describe Your Query"
# --------------------------
st.header("Recommendation Mode")
mode = st.radio("Choose how you want to get recommendations:",
                ("Rate Books", "Describe Your Query"))

# --------------------------
# Mode: Rate Books
# --------------------------
# In Rate Books mode, the user selects books via auto-complete (from Neo4j) and rates them.
rated_books_data = {}
if mode == "Rate Books":
    st.subheader("Rate Books")
    st.write("Select up to 5 books and rate each (ratings from 1 to 5).")
    with st.spinner("Loading available book titles from Neo4j..."):
        available_books = neo4j_conn.get_all_book_titles(limit=1000)
    if available_books:
        # Create a dictionary mapping title to work_id
        book_options = {record["title"]: record["work_id"] for record in available_books}
    else:
        book_options = {}
        st.error("No book titles found from Neo4j.")
    selected_titles = st.multiselect("Select books to rate:", list(book_options.keys()))
    if selected_titles:
        for title in selected_titles:
            rating = st.slider(f"Rate '{title}':", min_value=1.0, max_value=5.0, value=3.0, step=0.5)
            rated_books_data[book_options[title]] = {"title": title, "rating": rating}

# --------------------------
# Mode: Describe Your Query
# --------------------------
query_text = ""
if mode == "Describe Your Query":
    st.subheader("Describe Your Query")
    query_text = st.text_input("What kind of book are you looking for?",
                               placeholder="Describe your book preferences here...")

# --------------------------
# Recommendation Trigger Button
# --------------------------
if st.button("Get Recommendations"):
    # --------------------------
    # Qdrant Semantic Search Part
    # --------------------------
    semantic_results = []
    if mode == "Rate Books" and rated_books_data:
        st.info("Generating semantic recommendations based on your rated books...")
        aggregated_semantic = {}
        for work_id, info in rated_books_data.items():
            current_query = info["title"]
            user_rating = info["rating"]
            if not hasattr(qdrant_conn, 'client') or qdrant_conn.client is None:
                status, message = qdrant_conn.connect()
                if not status:
                    st.error(f"Failed to connect to Qdrant: {message}")
                    st.stop()
            if not hasattr(qdrant_conn, 'model') or qdrant_conn.model is None:
                with st.spinner("Loading embedding model..."):
                    if not qdrant_conn.load_model():
                        st.error("Failed to load embedding model.")
                        st.stop()
            with st.spinner(f"Searching similar books for '{current_query}'..."):
                q_results = qdrant_conn.search_similar_books(query_text=current_query, limit=5)
                formatted = qdrant_conn.format_results(q_results)
            # Aggregate semantic results: boost score = user_rating * qdrant_score.
            for result in formatted:
                rid = result.get("work_id")
                if not rid:
                    continue
                boost = user_rating * result.get("score", 0)
                if rid in aggregated_semantic:
                    aggregated_semantic[rid]["qdrant_score"] += boost
                else:
                    aggregated_semantic[rid] = {
                        "title": result.get("title", "Unknown"),
                        "qdrant_score": boost
                    }
        semantic_results = sorted(aggregated_semantic.values(), key=lambda x: x["qdrant_score"], reverse=True)
    elif mode == "Describe Your Query" and query_text:
        st.info("Generating semantic recommendations based on your query...")
        if not hasattr(qdrant_conn, 'client') or qdrant_conn.client is None:
            status, message = qdrant_conn.connect()
            if not status:
                st.error(f"Failed to connect to Qdrant: {message}")
                st.stop()
        if not hasattr(qdrant_conn, 'model') or qdrant_conn.model is None:
            with st.spinner("Loading embedding model..."):
                if not qdrant_conn.load_model():
                    st.error("Failed to load embedding model.")
                    st.stop()
        with st.spinner("Searching for similar books via Qdrant..."):
            qdrant_results = qdrant_conn.search_similar_books(query_text=query_text, limit=5)
            semantic_results = qdrant_conn.format_results(qdrant_results)

    # --------------------------
    # Insert Ratings into Neo4j (Only for Rate Books Mode)
    # --------------------------
    if mode == "Rate Books" and rated_books_data:
        with st.spinner("Saving your ratings to Neo4j..."):
            neo4j_conn.insert_user_ratings(user_id=user_id, rated_books_data=rated_books_data)
            log_to_debug(f"Inserted ratings for user {user_id}", "info")

    # --------------------------
    # Neo4j Collaborative Filtering Part (Only for Rate Books Mode)
    # --------------------------
    cf_results = []
    if mode == "Rate Books":
        # Check if at least 2 books are rated 4.0 or above.
        count_high_ratings = sum(1 for data in rated_books_data.values() if data["rating"] >= 4.0)
        if count_high_ratings < 2:
            st.error("Please rate at least 2 books with a rating of 4.0 or higher for collaborative filtering recommendations.")
            log_to_debug("Not enough high ratings to perform CF.", "warning")
        else:
            with st.spinner("Computing collaborative filtering recommendations via Neo4j..."):
                cf_results = neo4j_conn.get_collaborative_recommendations(user_id=user_id, limit=10)
                log_to_debug(f"CF Results: {cf_results}", "info")

    # --------------------------
    # Aggregation of Recommendations
    # --------------------------
    def aggregate_recommendations(cf_results, semantic_results, boost_factor=0.5):
        """
        Aggregates recommendations from CF (Neo4j) and Semantic (Qdrant) search.
        For Rate Books mode, CF is the base; if a book appears in both, boost its CF score.
        """
        aggregated = {}
        # Use CF results as base.
        for item in cf_results:
            work_id = item["work_id"]
            aggregated[work_id] = {
                "title": item.get("title"),
                "cf_score": item.get("cf_score", 0),
                "qdrant_score": 0
            }
        # Boost CF results if they also appear in semantic results.
        for item in semantic_results:
            work_id = item.get("work_id")
            if work_id and work_id in aggregated:
                aggregated[work_id]["qdrant_score"] = item.get("qdrant_score", item.get("score", 0))
        for work_id, rec in aggregated.items():
            rec["combined_score"] = rec["cf_score"] + boost_factor * rec["qdrant_score"]
        aggregated_list = list(aggregated.values())
        aggregated_list.sort(key=lambda x: x["combined_score"], reverse=True)
        return aggregated_list

    if mode == "Rate Books":
        final_recommendations = aggregate_recommendations(cf_results, semantic_results, boost_factor=0.5)
    else:
        final_recommendations = sorted(semantic_results, key=lambda x: x.get("score", 0), reverse=True)

    # --------------------------
    # Display Final Aggregated Recommendations
    # --------------------------
    if final_recommendations:
        st.success(f"Found {len(final_recommendations)} recommended books.")
        for rec in final_recommendations:
            with st.container():
                col1, col2 = st.columns([1, 4])
                with col1:
                    score = rec.get("combined_score", rec.get("score", 0))
                    st.metric("Score", f"{score:.2f}")
                with col2:
                    st.markdown(f"### {rec['title']}")
                st.divider()
    else:
        st.warning("No recommendations found.")

    # --------------------------
    # Clear Temporary User Data (for temp_user only)
    # --------------------------
    if user_id == "temp_user":
        with st.spinner("Clearing temporary user ratings from Neo4j..."):
            neo4j_conn.clear_temp_user(user_id)
            log_to_debug("Cleared temp_user data", "info")

# --------------------------
# Debug Messages Section
# --------------------------
with st.expander("Debug Messages", expanded=False):
    if not st.session_state.debug_messages:
        st.write("No debug messages yet. Use the test buttons above to see connection debug messages.")
    else:
        for msg in reversed(st.session_state.debug_messages):
            if msg["level"] == "error":
                st.error(msg["message"])
            elif msg["level"] == "warning":
                st.warning(msg["message"])
            elif msg["level"] == "success":
                st.success(msg["message"])
            else:
                st.info(msg["message"])
