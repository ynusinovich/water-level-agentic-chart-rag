"""
Ingest USGS monitoring station metadata into Qdrant vector database.
"""

import os
import sys
import time
import json
from typing import List, Dict, Any
import requests
from datetime import datetime
import uuid
from dotenv import load_dotenv
from tqdm import tqdm
import argparse
import openai
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, 
    VectorParams, 
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    PayloadSchemaType
)

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = "usgs_stations"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536

# Initialize clients
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
qdrant_client = QdrantClient(
    host=QDRANT_HOST,
    port=QDRANT_PORT,
    grpc_port=6334,
    prefer_grpc=True,
    timeout=120.0
)

# USGS API configuration
USGS_SITE_API = "https://waterservices.usgs.gov/nwis/site/"
PARAMETER_CODES = {
    "00060": "Discharge (cubic feet per second)",
    "00065": "Gage height (feet)",
    "00010": "Temperature (Celsius)",
    "00095": "Specific conductance (microsiemens/cm)",
    "00400": "pH",
    "72019": "Depth to water level (feet below land surface)",
    "00045": "Precipitation (inches)"
}

def fetch_usgs_stations(state_code: str, site_type: str = "ST", require_iv: bool = False) -> List[Dict]:
    """
    Fetch USGS station metadata for a given state and site type.
    
    Args:
        state_code: Two-letter state code (e.g., "AZ", "CA")
        site_type: Site type code ("ST" for stream, "GW" for groundwater, "SP" for spring)
    
    Returns:
        List of station dictionaries
    """
    params = {
        "format": "rdb",
        "stateCd": state_code,
        "siteType": site_type,
        "siteStatus": "active",
    }
    if require_iv:
        params["hasDataTypeCd"] = "iv"  # Has instantaneous values (real-time data)
    
    try:
        response = requests.get(USGS_SITE_API, params=params, timeout=30)
        response.raise_for_status()
        return parse_rdb_response(response.text, site_type)
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {state_code}: {e}")
        return []

def parse_rdb_response(rdb_text: str, site_type: str) -> List[Dict]:
    """Parse USGS RDB (tab-delimited) format response."""
    lines = rdb_text.strip().split('\n')
    stations = []
    
    # Find the header line (starts with "agency_cd")
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("agency_cd"):
            header_idx = i
            break
    
    if header_idx is None:
        return stations
    
    headers = lines[header_idx].split('\t')
    
    # Skip the format line (header_idx + 1) and process data lines
    for line in lines[header_idx + 2:]:
        if line.startswith('#') or not line.strip():
            continue
            
        values = line.split('\t')
        if len(values) < len(headers):
            continue
            
        # Create station dict
        station = {}
        for h, v in zip(headers, values):
            station[h] = v
        
        # Transform to our format
        station_data = transform_station_data(station, site_type)
        if station_data:
            stations.append(station_data)
    
    return stations

def transform_station_data(raw_station: Dict, site_type: str) -> Dict:
    """Transform raw USGS data to our schema."""
    fips_to_postal = {"04":"AZ","06":"CA","08":"CO","35":"NM","49":"UT","32":"NV"}
    state_fips = raw_station.get("state_cd", "")
    state_postal = fips_to_postal.get(state_fips, state_fips)

    try:
        # Map site types
        type_map = {
            "ST": "surface_water",
            "GW": "groundwater", 
            "SP": "spring"
        }
        
        # Extract relevant fields
        station = {
            "station_id": raw_station.get("site_no", ""),
            "station_name": raw_station.get("station_nm", ""),
            "station_type": type_map.get(site_type, "unknown"),
            "latitude": float(raw_station.get("dec_lat_va", 0) or 0),
            "longitude": float(raw_station.get("dec_long_va", 0) or 0),
            "state": state_postal,
            "county": raw_station.get("county_cd", ""),
            "huc_code": raw_station.get("huc_cd", ""),
            "altitude": raw_station.get("alt_va", ""),
            "active": True,
            "agency": raw_station.get("agency_cd", "USGS"),
        }
        
        # Create description for embedding
        station["description"] = create_station_description(station)
        
        return station
        
    except (KeyError, ValueError) as e:
        print(f"Error transforming station data: {e}")
        return None

def create_station_description(station: Dict) -> str:
    """Create a text description for embedding."""
    type_readable = {
        "surface_water": "surface water",
        "groundwater": "groundwater",
        "spring": "spring water"
    }
    
    description = (
        f"USGS monitoring station {station['station_id']}: {station['station_name']}. "
        f"This {type_readable.get(station['station_type'], 'water')} monitoring station "
        f"is located at latitude {station['latitude']:.4f}, longitude {station['longitude']:.4f} "
        f"in {station['state']}. "
    )
    
    if station.get('altitude'):
        description += f"Elevation: {station['altitude']} feet. "
    
    description += (
        f"The station provides real-time hydrological data for water resource management, "
        f"flood forecasting, and environmental monitoring. "
        f"HUC: {station.get('huc_code', 'N/A')}"
    )
    
    return description

def create_embeddings(texts: List[str]) -> List[List[float]]:
    """Create embeddings for a batch of texts using OpenAI."""
    try:
        response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        print(f"Error creating embeddings: {e}")
        return []

def setup_qdrant_collection(drop: bool = False):
    collections = qdrant_client.get_collections().collections
    if drop and any(c.name == COLLECTION_NAME for c in collections):
        qdrant_client.delete_collection(COLLECTION_NAME)

    if not any(c.name == COLLECTION_NAME for c in collections):
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIMENSION, distance=Distance.COSINE)
        )

    # Create payload indexes (ignore 'already exists' errors)
    try:
        qdrant_client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="state",
            field_schema=PayloadSchemaType.KEYWORD
        )
    except Exception:
        pass

    try:
        qdrant_client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="station_type",
            field_schema=PayloadSchemaType.KEYWORD
        )
    except Exception:
        pass

    print("Collection and payload indexes are ready.")
    return

def wait_for_qdrant(url=f"http://{QDRANT_HOST}:{QDRANT_PORT}/readyz", timeout_s=30):
    for _ in range(timeout_s):
        try:
            if requests.get(url, timeout=1).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("Qdrant not ready")

def chunked(seq, size: int):
    """Yield successive chunks from a list."""
    for i in range(0, len(seq), size):
        yield seq[i : i + size]

def to_point_uuid(site_id: str) -> str:
    # Stable, deterministic UUID based on the site id
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"usgs:{site_id}"))

def ingest_stations_to_qdrant(stations: List[Dict]):
    """Ingest stations into Qdrant with embeddings."""
    if not stations:
        print("No stations to ingest.")
        return
    
    print(f"Creating embeddings for {len(stations)} stations...")
    
    # Batch process embeddings
    batch_size = 50
    all_points = []
    
    for i in tqdm(range(0, len(stations), batch_size)):
        batch = stations[i:i + batch_size]
        texts = [s["description"] for s in batch]
        
        # Create embeddings with retry logic
        embeddings = create_embeddings(texts)
        if not embeddings:
            print(f"Failed to create embeddings for batch {i//batch_size}")
            continue
        
        # Create Qdrant points
        for station, embedding in zip(batch, embeddings):
            point = PointStruct(
                id=to_point_uuid(station["station_id"]),
                vector=embedding,
                payload={
                    "station_id": station["station_id"],
                    "station_name": station["station_name"],
                    "station_type": station["station_type"],
                    "latitude": station["latitude"],
                    "longitude": station["longitude"],
                    "state": station["state"],
                    "county": station["county"],
                    "huc_code": station["huc_code"],
                    "description": station["description"],
                    "active": station["active"]
                }
            )
            all_points.append(point)
        
        # Rate limiting for OpenAI
        time.sleep(1)
    
    # Upload to Qdrant (chunked + retries)
    if all_points:
        print(f"Uploading {len(all_points)} points to Qdrant...")
        BATCH = int(os.getenv("QDRANT_UPSERT_BATCH", "500"))  # tune via .env if desired
        RETRIES = 3

        uploaded = 0
        for batch in chunked(all_points, BATCH):
            for attempt in range(1, RETRIES + 1):
                try:
                    qdrant_client.upsert(
                        collection_name=COLLECTION_NAME,
                        points=batch,
                        wait=False,  # don’t block on indexing; speeds things up
                    )
                    uploaded += len(batch)
                    break
                except Exception as e:
                    if attempt == RETRIES:
                        raise
                    print(f"Upsert failed for batch of {len(batch)} "
                        f"(attempt {attempt}/{RETRIES}): {e}. Retrying...")
                    time.sleep(2 ** attempt)  # 2s, 4s backoff
        print(f"Uploaded {uploaded} / {len(all_points)} points.")
    else:
        print("No points to upload.")

def main():
    """Main ingestion pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--iv-only", action="store_true",
                        help="Only include sites with instantaneous values (iv). Off by default.")
    args = parser.parse_args()

    print("=== USGS Station Data Ingestion ===\n")
    
    # Check for API key
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not found in environment variables.")
        print("Please set it in your .env file.")
        sys.exit(1)
    
    # Setup Qdrant collection
    wait_for_qdrant()
    setup_qdrant_collection()
    
    # Define states and site types to ingest
    # Start with a few states for testing, expand later
    states = ["AZ", "CA", "CO", "NM", "UT", "NV"]  # Southwest states
    site_types = [
        ("ST", "surface_water"),
        ("GW", "groundwater"),
        ("SP", "spring")
    ]
    
    all_stations = []
    
    print(f"Fetching station data for {len(states)} states...")
    for state in states:
        print(f"\nProcessing {state}...")
        for site_code, site_name in site_types:
            print(f"  Fetching {site_name} stations...")
            stations = fetch_usgs_stations(state, site_code, require_iv=args.iv_only)
            print(f"  Found {len(stations)} {site_name} stations")

            for s in stations:
                s["state"] = state  # e.g., "CA"
                s["description"] = create_station_description(s)

            all_stations.extend(stations)
            
            # Rate limiting for USGS API
            time.sleep(0.5)
    
    print(f"\nTotal stations collected: {len(all_stations)}")
    
    # Deduplicate by station_id
    unique_stations = {}
    for station in all_stations:
        if station and station.get("station_id"):
            unique_stations[station["station_id"]] = station
    
    stations_list = list(unique_stations.values())
    print(f"Unique stations: {len(stations_list)}")
    
    # Ingest to Qdrant
    ingest_stations_to_qdrant(stations_list)
    
    # Print summary
    collection_info = qdrant_client.get_collection(COLLECTION_NAME)
    print(f"\n=== Ingestion Complete ===")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Points count: {collection_info.points_count}")
    print(f"Indexed vectors: {collection_info.indexed_vectors_count}")

if __name__ == "__main__":
    main()