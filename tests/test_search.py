"""
Test semantic search on USGS station data in Qdrant.
"""

import os
import sys
from typing import List
from dotenv import load_dotenv
import openai
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import json

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = "usgs_stations"
EMBEDDING_MODEL = "text-embedding-3-small"

# Initialize clients
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def create_query_embedding(query: str) -> List[float]:
    """Create embedding for search query."""
    try:
        response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=query
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error creating embedding: {e}")
        return None

def search_stations(query: str, limit: int = 5, filter_dict: dict = None):
    """
    Search for relevant stations using semantic search.
    
    Args:
        query: Natural language search query
        limit: Number of results to return
        filter_dict: Optional filters (e.g., {"station_type": "surface_water"})
    """
    # Create query embedding
    print(f"\nSearching for: '{query}'")
    embedding = create_query_embedding(query)
    if not embedding:
        print("Failed to create query embedding")
        return
    
    # Build filter if provided
    search_params = {
        "collection_name": COLLECTION_NAME,
        "query": embedding,
        "limit": limit
    }
    
    if filter_dict:
        # Add Qdrant filter
        conditions = []
        for key, value in filter_dict.items():
            conditions.append(
                FieldCondition(
                    key=key,
                    match=MatchValue(value=value)
                )
            )
        if conditions:
            search_params["query_filter"] = Filter(must=conditions)
    
    # Perform search
    try:
        resp = qdrant_client.query_points(**search_params)
        results = resp.points
        
        print(f"\nFound {len(results)} results:\n")
        print("-" * 80)
        
        for i, result in enumerate(results, 1):
            payload = result.payload
            print(f"\n{i}. Station: {payload['station_name']}")
            print(f"   ID: {payload['station_id']}")
            print(f"   Type: {payload['station_type']}")
            print(f"   State: {payload['state']}")
            print(f"   Location: ({payload['latitude']:.4f}, {payload['longitude']:.4f})")
            print(f"   Similarity Score: {result.score:.4f}")
            print(f"   Description: {payload['description'][:200]}...")
        
        print("-" * 80)
        
        return results
        
    except Exception as e:
        print(f"Search error: {e}")
        return []

def interactive_search():
    """Interactive search mode."""
    print("=== USGS Station Search Test ===")
    print("Type 'quit' to exit, 'filters' to see filter options\n")
    
    while True:
        query = input("\nEnter search query: ").strip()
        
        if query.lower() == 'quit':
            print("Exiting...")
            break
        
        if query.lower() == 'filters':
            print("\nAvailable filters:")
            print("  - station_type: surface_water, groundwater, spring")
            print("  - state: AZ, CA, CO, NM, UT, NV (or others)")
            print("\nExample with filter:")
            print("  Query: 'monitoring stations'")
            print("  Filter: {'station_type': 'groundwater', 'state': 'AZ'}")
            continue
        
        # Ask for optional filters
        use_filter = input("Apply filters? (y/n): ").strip().lower()
        filter_dict = None
        
        if use_filter == 'y':
            filter_dict = {}
            station_type = input("Station type (surface_water/groundwater/spring or Enter to skip): ").strip()
            if station_type:
                filter_dict['station_type'] = station_type
            
            state = input("State code (e.g., AZ, CA or Enter to skip): ").strip().upper()
            if state:
                filter_dict['state'] = state
            
            if not filter_dict:
                filter_dict = None
        
        # Perform search
        search_stations(query, limit=5, filter_dict=filter_dict)

def run_test_queries():
    """Run a set of predefined test queries."""
    test_queries = [
        ("water flow monitoring in Arizona", None),
        ("groundwater levels", {"station_type": "groundwater"}),
        ("Colorado River surface water", {"station_type": "surface_water"}),
        ("spring monitoring stations in California", {"station_type": "spring", "state": "CA"}),
        ("real-time discharge measurements", None),
        ("high elevation monitoring stations", None),
    ]
    
    print("=== Running Test Queries ===\n")
    
    for query, filters in test_queries:
        print("\n" + "="*80)
        if filters:
            print(f"Query: '{query}' with filters: {filters}")
        search_stations(query, limit=3, filter_dict=filters)
        print("\nPress Enter to continue...")
        input()

def check_collection_stats():
    """Display collection statistics."""
    try:
        info = qdrant_client.get_collection(COLLECTION_NAME)
        print("\n=== Collection Statistics ===")
        print(f"Collection: {COLLECTION_NAME}")
        print(f"Total points: {info.points_count}")
        print(f"Indexed vectors: {info.indexed_vectors_count}")
        print(f"Status: {info.status}")
        print("="*30)
    except Exception as e:
        print(f"Error getting collection info: {e}")
        print("Make sure you've run ingest_data.py first!")
        sys.exit(1)

def main():
    """Main test function."""
    # Check for API key
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not found in environment variables.")
        sys.exit(1)
    
    # Check collection exists
    check_collection_stats()
    
    print("\nSelect test mode:")
    print("1. Interactive search")
    print("2. Run predefined test queries")
    print("3. Both")
    
    choice = input("\nChoice (1/2/3): ").strip()
    
    if choice == "1":
        interactive_search()
    elif choice == "2":
        run_test_queries()
    elif choice == "3":
        run_test_queries()
        print("\n" + "="*80)
        print("Switching to interactive mode...")
        print("="*80)
        interactive_search()
    else:
        print("Invalid choice. Running interactive search...")
        interactive_search()

if __name__ == "__main__":
    main()