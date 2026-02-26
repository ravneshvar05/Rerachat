"""
search.py — Dual search engine combining SQLite (structured) + ChromaDB (semantic).

Flow:
  1. Build SQL query from ParsedQuery filters → get matching unit IDs
  2. Run ChromaDB semantic search (filtered to same unit IDs when possible)
  3. Merge + rank results
  4. Return enriched result dicts ready for the answer generator
"""

import sqlite3
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

from config import settings
from logger import logger
from query_parser import ParsedQuery


# ─── Connections (lazy singletons) ───────────────────────────────────────────

_conn: sqlite3.Connection | None = None
_collection: chromadb.Collection | None = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(settings.sqlite_path, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
    return _conn


import streamlit as st

@st.cache_resource(show_spinner="Initializing AI search model (takes ~15s on first run)...")
def _get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(settings.chroma_path))
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    return client.get_or_create_collection(
        name=settings.CHROMA_COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )


# ─── Step 1: SQLite Structured Filter ────────────────────────────────────────

def _build_sql_filter(q: ParsedQuery) -> tuple[str, list]:
    """Build a WHERE clause + params from the parsed query."""
    conditions = []
    params = []

    if q.city:
        conditions.append("LOWER(p.city) LIKE ?")
        params.append(f"%{q.city.lower()}%")

    if q.bhk:
        conditions.append("u.bhk = ?")
        params.append(q.bhk)

    if q.property_type:
        conditions.append("u.property_type = ?")
        params.append(q.property_type.upper())

    if q.min_area_sqft:
        conditions.append("u.super_builtup_sqft >= ?")
        params.append(q.min_area_sqft)

    if q.max_area_sqft:
        conditions.append("u.super_builtup_sqft <= ?")
        params.append(q.max_area_sqft)

    # Room flags
    flag_map = {
        "must_have_pooja_room":    "u.has_pooja_room",
        "must_have_study_room":    "u.has_study_room",
        "must_have_terrace":       "u.has_terrace",
        "must_have_servant_room":  "u.has_servant_room",
        "must_have_garden":        "u.has_garden",
        "must_have_home_theatre":  "u.has_home_theatre",
    }
    for attr, col in flag_map.items():
        if getattr(q, attr):
            conditions.append(f"{col} = 1")
            
    # Special handling for Gym (often a project amenity rather than just a unit feature)
    if q.must_have_gym:
        conditions.append(
            "(u.has_gym = 1 OR "
            "LOWER(p.amenities_text) LIKE '%gym%' OR "
            "LOWER(p.amenities_text) LIKE '%health club%')"
        )

    # Project flags
    project_flag_map = {
        "must_have_pool":      "p.has_pool",
        "must_have_clubhouse": "p.has_clubhouse",
        "must_have_park":      "p.has_park",
    }
    for attr, col in project_flag_map.items():
        if getattr(q, attr):
            conditions.append(f"{col} = 1")

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    return where, params


def sqlite_search(q: ParsedQuery, limit: int = 50) -> list[dict]:
    """Run structured SQL search. Returns list of unit+project dicts."""
    where, params = _build_sql_filter(q)

    sql = f"""
        SELECT
            u.unit_id, u.unit_type, u.property_type, u.bhk,
            u.carpet_area_sqft, u.super_builtup_sqft,
            u.balcony_area_sqft, u.wash_area_sqft,
            u.applicable_buildings, u.description,
            u.has_pooja_room, u.has_study_room, u.has_terrace,
            u.has_servant_room, u.has_garden, u.has_home_theatre, u.has_gym,
            u.attached_bathrooms,
            p.project_id, p.project_name, p.developer_name,
            p.city, p.neighbourhood, p.amenities_text,
            p.has_clubhouse, p.has_pool, p.has_park,
            p.has_sports_courts, p.has_parking,
            p.project_status, p.total_units
        FROM units u
        JOIN projects p ON u.project_id = p.project_id
        {where}
        LIMIT {limit}
    """

    conn = _get_conn()
    rows = conn.execute(sql, params).fetchall()
    results = [dict(row) for row in rows]
    logger.debug(f"SQLite returned {len(results)} result(s) for filters: city={q.city}, bhk={q.bhk}, type={q.property_type}")
    return results


# ─── Step 2: ChromaDB Semantic Search ────────────────────────────────────────

def semantic_search(q: ParsedQuery, candidate_ids: list[str] | None, top_k: int = 10) -> list[str]:
    """
    Run semantic search in ChromaDB.
    If candidate_ids is provided, only search within those IDs (post-SQL filter).
    Returns list of unit_ids in ranked order.
    """
    if not q.semantic_query:
        return candidate_ids[:top_k] if candidate_ids else []

    collection = _get_collection()

    # Build metadata filter for ChromaDB (narrows search space)
    filters = []
    if q.city:
        filters.append({"city": {"$eq": q.city}})
    if q.bhk:
        filters.append({"bhk": {"$eq": q.bhk}})
    if q.property_type:
        filters.append({"property_type": {"$eq": q.property_type.upper()}})

    chroma_where = {}
    if len(filters) == 1:
        chroma_where = filters[0]
    elif len(filters) > 1:
        chroma_where = {"$and": filters}

    kwargs: dict[str, Any] = {
        "query_texts": [q.semantic_query],
        "n_results": min(top_k, max(collection.count(), 1)),
        "include": ["distances", "metadatas"],
    }
    if chroma_where:
        kwargs["where"] = chroma_where

    try:
        results = collection.query(**kwargs)
        ranked_ids = results["ids"][0]   # list of unit_ids, best match first
        logger.debug(f"ChromaDB returned {len(ranked_ids)} semantic match(es)")
        return ranked_ids
    except Exception as e:
        logger.warning(f"ChromaDB search failed: {e}. Falling back to SQL results.")
        return candidate_ids[:top_k] if candidate_ids else []


# ─── Step 3: Merge + Rank + Group ────────────────────────────────────────────────

def merge_and_group_results(
    sql_results: list[dict],
    semantic_ranked_ids: list[str],
    top_k: int = 5,
) -> list[dict]:
    """
    Merge SQL and ChromaDB results, then group them by project.
    This ensures we return `top_k` UNIQUE projects, not just 5 units from
    the same project.
    """
    sql_map = {r["unit_id"]: r for r in sql_results}
    
    # 1. Order units based on semantic rank first, then SQL
    ordered_units = []
    seen_units = set()

    for uid in semantic_ranked_ids:
        if uid in sql_map and uid not in seen_units:
            ordered_units.append(sql_map[uid])
            seen_units.add(uid)

    for r in sql_results:
        if r["unit_id"] not in seen_units:
            ordered_units.append(r)
            seen_units.add(r["unit_id"])

    # 2. Group by project
    project_groups = {} # project_id -> {project_info, units: []}
    
    for u in ordered_units:
        pid = u["project_id"]
        if pid not in project_groups:
            # Extract project-level fields
            proj_info = {
                "project_id": u["project_id"],
                "project_name": u["project_name"],
                "developer_name": u["developer_name"],
                "city": u["city"],
                "neighbourhood": u["neighbourhood"],
                "amenities_text": u["amenities_text"],
                "has_clubhouse": u["has_clubhouse"],
                "has_pool": u["has_pool"],
                "has_park": u["has_park"],
                "has_sports_courts": u["has_sports_courts"],
                "has_parking": u["has_parking"],
                "project_status": u["project_status"],
                "total_units": u["total_units"],
                "matching_units": []
            }
            project_groups[pid] = proj_info
            
        # Extract unit-level fields
        unit_info = {k: v for k, v in u.items() if k not in project_groups[pid] or k == "project_id"}
        project_groups[pid]["matching_units"].append(unit_info)

    # 3. Return top_k unique projects
    final_projects = list(project_groups.values())[:top_k]
    logger.info(f"Grouped results: {len(project_groups)} unique projects. Returning top {top_k}")
    return final_projects


# ─── Public API ───────────────────────────────────────────────────────────────

def search(q: ParsedQuery, top_k: int = 5) -> list[dict]:
    """
    Main search function. Call this from the chatbot.

    Args:
        q: ParsedQuery from query_parser.parse_query()
        top_k: Number of UNIQUE top projects to return

    Returns:
        List of project dicts, each containing a 'matching_units' list
    """
    if q.needs_clarification:
        return []

    # Step 1: SQL filter - fetch more to ensure we get enough unique projects
    sql_results = sqlite_search(q, limit=200)

    if not sql_results:
        logger.info("No SQL results, trying semantic-only search.")

    # Step 2: Semantic search
    sql_ids = [r["unit_id"] for r in sql_results] if sql_results else None
    # Ask for more semantic results so grouping has enough variety
    semantic_ids = semantic_search(q, candidate_ids=sql_ids, top_k=50)

    # Step 3: Merge and Group by Project
    return merge_and_group_results(sql_results, semantic_ids, top_k=top_k)
