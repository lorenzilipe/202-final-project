import streamlit as st
from database.neo4j_connector import Neo4jConnector
from database.qdrant_connector import QdrantConnector

# Set page configuration
st.set_page_config(
    page_title="Hybrid Book Recommender",
    page_icon="📚",
    layout="wide"
)

# Create session state for storing intermediate results
if "selected_books" not in st.session_state:
    st.session_state.selected_books = []  # IDs of books selected by user

if "initial_book" not in st.session_state:
    st.session_state.initial_book = None  # The initial book the user selected
    
if "similar_books" not in st.session_state:
    st.session_state.similar_books = []  # Semantically similar books from Qdrant
    
if "final_recommendations" not in st.session_state:
    st.session_state.final_recommendations = []  # Final combined recommendations
    
if "search_results" not in st.session_state:
    st.session_state.search_results = []  # Store search results
    
if "current_step" not in st.session_state:
    st.session_state.current_step = 1  # Track which step we're on

if "debug_messages" not in st.session_state:
    st.session_state.debug_messages = []  # For storing debug messages

# Function to log debug messages
def log_debug(message, level="info"):
    print(f"[{level.upper()}] {message}")
    st.session_state.debug_messages.append({"message": message, "level": level})

# Initialize database connections
@st.cache_resource
def init_connections():
    neo4j_conn = Neo4jConnector()
    qdrant_conn = QdrantConnector()
    return neo4j_conn, qdrant_conn

neo4j_conn, qdrant_conn = init_connections()

# Function to reset session state and start over
def reset_recommendations():
    st.session_state.selected_books = []
    st.session_state.initial_book = None
    st.session_state.similar_books = []
    st.session_state.final_recommendations = []
    st.session_state.search_results = []
    st.session_state.current_step = 1
    
# Function to select a book
def select_book(book_id, title, summary):
    st.session_state.initial_book = {
        "book_id": book_id,
        "title": title,
        "summary": summary
    }
    st.session_state.selected_books = [book_id]
    st.session_state.current_step = 2
    log_debug(f"Selected book: {title} (ID: {book_id})", "success")

# Function to continue to final recommendations
def continue_to_recommendations(additional_selections):
    # Always include the initial book
    selected_books = [st.session_state.initial_book["book_id"]]
    
    # Add any additional selections
    if additional_selections:
        selected_books.extend(additional_selections)
    
    # Update session state
    st.session_state.selected_books = selected_books
    st.session_state.current_step = 3
    
    log_debug(f"Selected {len(selected_books)} books in total", "success")

# App title
st.title("Hybrid Book Recommendation System")
st.write("This system combines content-based (semantic) and collaborative filtering to recommend books.")

# Add a reset button in the sidebar
st.sidebar.title("Controls")
if st.sidebar.button("Start Over"):
    reset_recommendations()
    st.experimental_rerun()

# Step 1: User searches for a book
if st.session_state.current_step == 1:
    st.header("Step 1: Find a book you like")
    st.write("Describe the kind of book you're looking for, and select one you like from the results.")

    # Search by description
    search_query = st.text_input("Description", key="query_input", 
                                placeholder="Describe the kind of book you're looking for (e.g., science fiction about time travel)")
    
    if st.button("Search"):
        if not search_query:
            st.warning("Please enter a search query")
        else:
            # Ensure connections
            status, message = qdrant_conn.connect()
            if not status:
                st.error(f"Failed to connect to Qdrant: {message}")
                st.stop()
                
            # Load model if needed
            if not hasattr(qdrant_conn, 'model') or qdrant_conn.model is None:
                with st.spinner("Loading embedding model..."):
                    if not qdrant_conn.load_model():
                        st.error("Failed to load embedding model")
                        st.stop()
                
            # Search for books
            with st.spinner("Searching for books..."):
                results = qdrant_conn.search_similar_books(query_text=search_query, limit=5)
                if results:
                    # Store results in session state
                    st.session_state.search_results = results
                    st.experimental_rerun()  # Force rerun to show results
                else:
                    st.warning("No books found matching your query")
    
    # Display search results if we have any
    if st.session_state.search_results:
        st.success(f"Found {len(st.session_state.search_results)} books matching your query")
        
        # Display each search result with a selection button
        for i, result in enumerate(st.session_state.search_results):
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                book_id = result.id
                title = result.payload.get('title', 'Unknown')
                summary = result.payload.get('summary', 'No description available')
                
                with col1:
                    st.subheader(title)
                    st.write(summary[:200] + "..." if len(summary) > 200 else summary)
                    st.caption(f"Match score: {result.score:.2f}")
                
                with col2:
                    # Create a unique key for each button using the book ID
                    if st.button(f"Select", key=f"sel_{book_id}"):
                        select_book(book_id, title, summary)
                        st.experimental_rerun()
                        
            st.divider()

# Step 2: Show similar books once initial book is selected
elif st.session_state.current_step == 2:
    st.header("Step 2: Select additional books you like")
    
    # Display the initial book
    st.subheader("Your selected book:")
    initial_book = st.session_state.initial_book
    st.markdown(f"### {initial_book['title']}")
    st.markdown(initial_book['summary'][:300] + "..." if len(initial_book.get('summary', '')) > 300 else initial_book.get('summary', ''))
    
    # If we haven't fetched similar books yet, get them now
    if not st.session_state.similar_books:
        with st.spinner("Finding similar books..."):
            # Ensure connections
            status, message = qdrant_conn.connect()
            if not status:
                st.error(f"Failed to connect to Qdrant: {message}")
                st.stop()
                
            # Get similar books
            results = qdrant_conn.search_similar_books(
                query_text=initial_book['title'] + " " + initial_book.get('summary', ''),
                limit=5
            )
            
            # Format and filter out the original book
            formatted_books = []
            for result in results:
                if result.id != initial_book['book_id']:
                    formatted_books.append({
                        "book_id": result.id,
                        "score": round(result.score, 4),
                        "title": result.payload.get('title', 'Unknown'),
                        "summary": result.payload.get('summary', 'No description available'),
                        "selected": False
                    })
            
            st.session_state.similar_books = formatted_books
            log_debug(f"Found {len(formatted_books)} similar books", "info")
    
    # Display similar books with checkboxes for selection
    st.subheader("Similar books you might also like:")
    st.write("Select additional books you like to improve recommendations:")
    
    # Create a container for selected books
    selected_additional = []
    
    # Display each similar book with a checkbox
    for i, book in enumerate(st.session_state.similar_books):
        with st.container():
            col1, col2 = st.columns([4, 1])
            
            with col1:
                st.markdown(f"### {book['title']}")
                st.markdown(book['summary'][:200] + "..." if len(book['summary']) > 200 else book['summary'])
                st.caption(f"Match score: {book['score']:.2f}")
            
            with col2:
                # Each checkbox needs a unique key
                selected = st.checkbox("Select", key=f"check_{book['book_id']}")
                if selected:
                    selected_additional.append(book['book_id'])
            
            st.divider()
    
    # Button to continue to recommendations
    if st.button("Generate Recommendations"):
        continue_to_recommendations(selected_additional)
        st.experimental_rerun()

# Step 3: Generate and show hybrid recommendations
elif st.session_state.current_step == 3:
    st.header("Step 3: Your Personalized Recommendations")
    
    # If we haven't generated final recommendations yet, do it now
    if not st.session_state.final_recommendations:
        with st.spinner("Generating personalized recommendations..."):
            recommendations = []
            
            # Connect to both systems
            qdrant_status, message = qdrant_conn.connect()
            neo4j_status = neo4j_conn.connect()
            
            if not qdrant_status:
                st.error(f"Failed to connect to Qdrant: {message}")
            if not neo4j_status:
                st.error("Failed to connect to Neo4j")
                
            if not (qdrant_status and neo4j_status):
                st.stop()
            
            # 1. First, try to get collaborative filtering recommendations (from Neo4j)
            # This calls teh get_similar_books_cf method from the Neo4jConnector class to get collaborative filtering recommendations
            log_debug(f"Attempting to get collaborative filtering recommendations...", "info")
            cf_results = []
            
            try:
                # Get collaborative filtering recommendations based on all selected books
                cf_results = neo4j_conn.get_similar_books_cf(
                    book_ids=st.session_state.selected_books,
                    limit=4
                )
                
                log_debug(f"Found {len(cf_results)} collaborative filtering recommendations", 
                          "success" if cf_results else "warning")
                
            except Exception as e:
                log_debug(f"Error getting collaborative filtering recommendations: {str(e)}", "error")
            
            # Format collaborative filtering results
            for i, result in enumerate(cf_results):
                formatted_result = {
                    "rank": i+1,
                    "book_id": result["work_id"],
                    "score": float(result["score"]),
                    "title": result["title"],
                    "summary": f"Average Rating: {result.get('average_rating', 'N/A')} from {result.get('ratings_count', 0)} ratings",
                    "source": "collaborative"
                }
                recommendations.append(formatted_result)
            
            # 2. Get semantic recommendations from Qdrant
            log_debug("Getting semantic recommendations...", "info")
            semantic_results = []
            
            try:
                # Get all selected book titles for better semantic search
                book_titles = []
                if st.session_state.initial_book:
                    book_titles.append(st.session_state.initial_book['title'])
                
                # Build search query from all selected books
                search_query = " ".join(book_titles)
                if not search_query:
                    search_query = "good books"
                    
                # Get semantic recommendations
                semantic_search = qdrant_conn.search_similar_books(
                    query_text=search_query,
                    limit=10  # Get more results to filter
                )
                
                # Filter out books that are already selected
                semantic_results = [
                    result for result in semantic_search 
                    if result.id not in st.session_state.selected_books
                ][:3]  # Limit to 3 results
                
                log_debug(f"Found {len(semantic_results)} semantic recommendations", 
                          "success" if semantic_results else "warning")
                
            except Exception as e:
                log_debug(f"Error getting semantic recommendations: {str(e)}", "error")
            
            # Format semantic results
            for i, result in enumerate(semantic_results):
                formatted_result = {
                    "rank": i+1,
                    "book_id": result.id,
                    "score": round(result.score, 4),
                    "title": result.payload.get('title', 'Unknown'),
                    "summary": result.payload.get('summary', 'No description available')[:200] + "..." 
                           if len(result.payload.get('summary', '')) > 200 
                           else result.payload.get('summary', 'No description available'),
                    "source": "semantic"
                }
                recommendations.append(formatted_result)
            
            # 3. If we still don't have recommendations, get popular books as fallback
            if not recommendations:
                log_debug("No personalized recommendations found. Getting popular books as fallback...", "warning")
                
                try:
                    popular_books_query = """
                    // Match all books with ratings
                    MATCH (b:Book)
                    WHERE b.ratings_count > 1000
                    RETURN 
                        b.work_id as work_id,
                        b.title as title,
                        b.average_rating as average_rating,
                        b.ratings_count as ratings_count,
                        b.average_rating * ln(b.ratings_count) as score
                    ORDER BY score DESC
                    LIMIT 3
                    """
                    
                    popular_books = neo4j_conn.execute_query(popular_books_query, {})
                    
                    # Format popular books
                    for i, book in enumerate(popular_books):
                        formatted_result = {
                            "rank": i+1,
                            "book_id": book["work_id"],
                            "score": float(book["score"]),
                            "title": book["title"],
                            "summary": f"Popular book with average rating: {book.get('average_rating', 'N/A')} from {book.get('ratings_count', 0)} ratings",
                            "source": "popular"
                        }
                        recommendations.append(formatted_result)
                        
                    log_debug(f"Added {len(popular_books)} popular books as recommendations", "info")
                    
                except Exception as e:
                    log_debug(f"Error getting popular books: {str(e)}", "error")
            
            # Store in session state
            st.session_state.final_recommendations = recommendations
            
            log_debug(f"Generated {len(recommendations)} total recommendations", "success")
    
    # Display the books that were used for recommendations
    with st.expander("Books you selected", expanded=False):
        st.write("These books were used to generate your recommendations:")
        for book_id in st.session_state.selected_books:
            # Try to get book info from Neo4j
            book_info = neo4j_conn.execute_query("""
                MATCH (b:Book {work_id: $work_id})
                RETURN b.title as title
                LIMIT 1
            """, {"work_id": book_id})
            
            if book_info and book_info[0].get('title'):
                st.write(f"- **{book_info[0]['title']}**")
            else:
                # If not in Neo4j, try to use the initial book info
                if st.session_state.initial_book and st.session_state.initial_book['book_id'] == book_id:
                    st.write(f"- **{st.session_state.initial_book['title']}**")
                else:
                    # Check in similar books
                    title_found = False
                    for similar in st.session_state.similar_books:
                        if similar['book_id'] == book_id:
                            st.write(f"- **{similar['title']}**")
                            title_found = True
                            break
                    
                    if not title_found:
                        st.write(f"- Book ID: {book_id}")
    
    # Display final recommendations
    recommendations = st.session_state.final_recommendations
    
    if recommendations:
        st.write(f"Based on your selections, here are {len(recommendations)} personalized book recommendations:")
        
        for rec in recommendations:
            with st.container():
                col1, col2, col3 = st.columns([1, 3, 1])
                
                with col1:
                    st.metric("Score", f"{rec['score']:.2f}")
                    
                with col2:
                    st.markdown(f"### {rec['title']}")
                    st.markdown(rec['summary'])
                
                with col3:
                    if rec['source'] == "semantic":
                        source_tag = "Content-Based"
                    elif rec['source'] == "collaborative":
                        source_tag = "Collaborative"
                    else:
                        source_tag = "Popular"
                    st.markdown(f"**Source:**  \n{source_tag}")
                
                st.divider()
                
        # Add button to start over
        if st.button("Get new recommendations"):
            reset_recommendations()
            st.experimental_rerun()
    else:
        st.warning("No recommendations could be generated. Please try with a different book.")
        
        if st.button("Start over"):
            reset_recommendations()
            st.experimental_rerun()

# Debug area at the bottom
with st.expander("Debug Information", expanded=False):
    st.write(f"Current step: {st.session_state.current_step}")
    st.write(f"Selected books: {st.session_state.selected_books}")
    
    # Show debug messages
    if st.session_state.debug_messages:
        for msg in reversed(st.session_state.debug_messages):  # Show newest messages first
            level = msg["level"]
            if level == "error":
                st.error(msg["message"])
            elif level == "warning":
                st.warning(msg["message"])
            elif level == "success":
                st.success(msg["message"])
            else:
                st.info(msg["message"])
    else:
        st.write("No debug messages.")