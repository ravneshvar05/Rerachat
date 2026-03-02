"""
search.py — 4-step search engine:
  1. SQL structured filter  (SQLite: city, bhk, property_type, feature flags)
  2. FTS5 text search       (landmarks, amenities, project name full-text)
  3. ChromaDB vector search (semantic similarity on unit descriptions)
  4. Merge + score + group  (group by project, return top-K unique projects)

For AGGREGATE queries: skips vectors, runs SQL COUNT/GROUP BY directly.
For DETAIL queries: fetches full project with all rooms.
"""

import sqlite3
from typing import Any, Optional
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions

from config import settings
from logger import logger
from query_planner import SearchPlan


# ─── Connections (lazy singletons) ───────────────────────────────────────────

_conn: Optional[sqlite3.Connection] = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(str(settings.sqlite_path), check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL;")
        _conn.execute("PRAGMA foreign_keys=ON;")
    return _conn


@st.cache_resource(show_spinner="Loading AI search model...")
def _get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(settings.chroma_path))
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=settings.CHROMA_EMBED_MODEL
    )
    return client.get_or_create_collection(
        name=settings.CHROMA_COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )


# ─── Step 1: SQL Structured Filter ───────────────────────────────────────────

UNIT_FLAG_COLUMNS = {
    "has_pooja_room", "has_study_room", "has_terrace", "has_servant_room",
    "has_garden", "has_home_theatre", "has_gym", "has_dressing_room",
    "has_store_room", "has_courtyard", "has_lobby", "has_balcony",
}

PROJECT_FLAG_COLUMNS = {
    "has_clubhouse", "has_pool", "has_park", "has_sports_courts", "has_parking",
}


def _build_sql_filter(plan: SearchPlan) -> tuple[str, list]:
    conditions, params = [], []

    # City filter — supports multiple cities
    if plan.cities:
        placeholders = ",".join("?" * len(plan.cities))
        conditions.append(f"LOWER(p.city) IN ({placeholders})")
        params.extend(c.lower() for c in plan.cities)

    if plan.bhk is not None:
        conditions.append("u.bhk = ?")
        params.append(plan.bhk)

    if plan.property_type:
        conditions.append("u.property_type = ?")
        params.append(plan.property_type.upper())

    if plan.min_area_sqft:
        conditions.append("(u.super_builtup_sqft >= ? OR u.carpet_area_sqft >= ?)")
        params.extend([plan.min_area_sqft, plan.min_area_sqft])

    if plan.max_area_sqft:
        conditions.append("(u.super_builtup_sqft <= ? OR u.carpet_area_sqft <= ?)")
        params.extend([plan.max_area_sqft, plan.max_area_sqft])

    if plan.entrance_facing:
        conditions.append("LOWER(u.entrance_facing) = ?")
        params.append(plan.entrance_facing.lower())

    # Unit feature flags
    for flag in (plan.must_have or []):
        if flag in UNIT_FLAG_COLUMNS:
            conditions.append(f"u.{flag} = 1")

    # Project feature flags
    for flag in (plan.project_must_have or []):
        if flag in PROJECT_FLAG_COLUMNS:
            conditions.append(f"p.{flag} = 1")

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    return where, params


def sql_search(plan: SearchPlan, limit: int = 300) -> list[dict]:
    """Run structured SQL search, returns unit+project dicts."""
    where, params = _build_sql_filter(plan)

    sql = f"""
        SELECT
            u.unit_id, u.unit_type, u.property_type, u.bhk, u.entrance_facing,
            u.description, u.carpet_area_sqft, u.super_builtup_sqft,
            u.balcony_area_sqft, u.wash_area_sqft, u.applicable_buildings,
            u.total_rooms, u.total_bedrooms, u.total_toilets, u.attached_bathrooms,
            u.master_bedroom_sqft, u.drawing_room_sqft, u.kitchen_sqft,
            u.has_pooja_room, u.has_study_room, u.has_terrace, u.has_servant_room,
            u.has_garden, u.has_home_theatre, u.has_gym, u.has_dressing_room,
            u.has_store_room, u.has_courtyard, u.has_lobby, u.has_balcony,
            p.project_id, p.project_name, p.developer_name, p.rera_number,
            p.project_status, p.possession_date,
            p.city, p.neighbourhood, p.address,
            p.has_clubhouse, p.has_pool, p.has_park, p.has_sports_courts,
            p.has_parking, p.has_commercial_shops,
            p.total_buildings, p.total_villas, p.society_description
        FROM units u
        JOIN projects p ON u.project_id = p.project_id
        {where}
        LIMIT {limit}
    """
    conn = _get_conn()
    rows = conn.execute(sql, params).fetchall()
    result = [dict(r) for r in rows]
    logger.debug(f"SQL returned {len(result)} row(s)")
    return result


# ─── Step 2: FTS5 Text Search ─────────────────────────────────────────────────

def fts5_search(plan: SearchPlan) -> set[str]:
    """
    Search the FTS5 index for landmark and/or amenity queries.
    Returns a set of matching project_ids.
    """
    if not plan.landmark_query and not plan.amenity_query and not plan.project_name_query:
        return set()

    conn = _get_conn()
    project_ids: set[str] = set()

    queries = []
    if plan.landmark_query:
        queries.append(plan.landmark_query)
    if plan.amenity_query:
        queries.append(plan.amenity_query)
    if plan.project_name_query:
        queries.append(plan.project_name_query)

    for q in queries:
        # Sanitize for FTS5 (escape special chars)
        safe_q = q.replace('"', '""')
        try:
            rows = conn.execute(
                'SELECT project_id FROM projects_fts WHERE projects_fts MATCH ? ORDER BY rank',
                (safe_q,),
            ).fetchall()
            for row in rows:
                project_ids.add(row["project_id"])
        except Exception as e:
            logger.warning(f"FTS5 query '{q}' failed: {e}")

    logger.debug(f"FTS5 matched {len(project_ids)} project(s)")
    return project_ids


# ─── Step 3: ChromaDB Vector Search ──────────────────────────────────────────

def vector_search(plan: SearchPlan, top_k: int = 50) -> list[tuple[str, float]]:
    """
    Run semantic search in ChromaDB.
    Returns list of (unit_id, similarity_score) sorted by best match.
    """
    if not plan.semantic_query:
        return []

    collection = _get_collection()
    count = collection.count()
    if count == 0:
        return []

    # Build metadata pre-filter (narrows search space)
    filters = []
    if len(plan.cities) == 1:
        filters.append({"city": {"$eq": plan.cities[0]}})
    if plan.bhk is not None:
        filters.append({"bhk": {"$eq": plan.bhk}})
    if plan.property_type:
        filters.append({"property_type": {"$eq": plan.property_type.upper()}})

    kwargs: dict[str, Any] = {
        "query_texts": [plan.semantic_query],
        "n_results": min(top_k, count),
        "include": ["distances", "metadatas"],
    }
    if len(filters) == 1:
        kwargs["where"] = filters[0]
    elif len(filters) > 1:
        kwargs["where"] = {"$and": filters}

    try:
        results = collection.query(**kwargs)
        ids = results["ids"][0]
        distances = results["distances"][0]   # cosine distance: lower = better
        scores = [(uid, 1.0 - dist) for uid, dist in zip(ids, distances)]
        logger.debug(f"ChromaDB returned {len(scores)} result(s)")
        return scores
    except Exception as e:
        logger.warning(f"ChromaDB search failed: {e}")
        return []


# ─── Step 4: Merge, Score, and Group by Project ───────────────────────────────

def merge_and_group(
    sql_results: list[dict],
    fts5_project_ids: set[str],
    vector_scores: list[tuple[str, float]],
    top_k: int = 5,
) -> list[dict]:
    """
    Merge SQL, FTS5, and vector results with a weighted score.
    Group by project, return top_k unique projects.

    Scoring:
      - SQL match:    base score 0.4  (unit passed hard filters)
      - FTS5 hit:     bonus 0.3       (project has landmark/amenity match)
      - Vector score: weighted 0.5 × similarity (best semantic match)
    """
    sql_map = {r["unit_id"]: r for r in sql_results}
    vector_map = {uid: score for uid, score in vector_scores}

    # Collect all candidate unit_ids
    all_unit_ids: set[str] = set(sql_map.keys()) | set(vector_map.keys())

    scored_units: list[tuple[float, dict]] = []

    for uid in all_unit_ids:
        unit = sql_map.get(uid)
        if unit is None:
            # Vector hit not in SQL — skip (doesn't pass hard filters)
            continue

        score = 0.4  # base for SQL match

        # FTS5 project match bonus
        if unit.get("project_id") in fts5_project_ids:
            score += 0.3

        # Vector similarity
        if uid in vector_map:
            score += 0.5 * vector_map[uid]

        scored_units.append((score, unit))

    # Within each project, select best-scoring unit
    project_best_score: dict[str, float] = {}
    project_units: dict[str, list[dict]] = {}

    for score, unit in scored_units:
        pid = unit["project_id"]
        if pid not in project_best_score or score > project_best_score[pid]:
            project_best_score[pid] = score
        project_units.setdefault(pid, []).append(unit)

    # Sort projects by best-unit score descending
    sorted_projects = sorted(project_best_score.keys(), key=lambda p: -project_best_score[p])

    results = []
    for pid in sorted_projects[:top_k]:
        units_in_project = project_units[pid]
        # Use any unit for project-level fields
        sample = units_in_project[0]

        # Get landmarks and amenities
        conn = _get_conn()
        landmarks = [r[0] for r in conn.execute(
            "SELECT landmark_name FROM project_landmarks WHERE project_id = ? ORDER BY id",
            (pid,),
        ).fetchall()]
        amenities = [r[0] for r in conn.execute(
            "SELECT amenity FROM project_amenities WHERE project_id = ? ORDER BY id",
            (pid,),
        ).fetchall()]

        proj = {
            "project_id":        pid,
            "project_name":      sample["project_name"],
            "developer_name":    sample["developer_name"],
            "city":              sample["city"],
            "neighbourhood":     sample["neighbourhood"],
            "address":           sample.get("address"),
            "project_status":    sample.get("project_status"),
            "possession_date":   sample.get("possession_date"),
            "has_clubhouse":     sample["has_clubhouse"],
            "has_pool":          sample["has_pool"],
            "has_park":          sample["has_park"],
            "has_sports_courts": sample["has_sports_courts"],
            "has_parking":       sample["has_parking"],
            "total_buildings":   sample.get("total_buildings"),
            "total_villas":      sample.get("total_villas"),
            "society_description": sample.get("society_description"),
            "nearby_landmarks":  landmarks,
            "amenities":         amenities,
            "relevance_score":   round(project_best_score[pid], 3),
            "matching_units":    units_in_project,
        }
        results.append(proj)

    logger.info(f"Merged: {len(results)} unique project(s) (from {len(sql_map)} SQL + {len(vector_map)} vector)")
    return results


# ─── Aggregate Query Handler ──────────────────────────────────────────────────

def aggregate_search(plan: SearchPlan) -> dict:
    """
    Handle AGGREGATE queries without vector search.
    Returns a summary dict with counts and breakdowns.
    """
    conn = _get_conn()
    where, params = _build_sql_filter(plan)

    base_sql = f"""
        FROM units u
        JOIN projects p ON u.project_id = p.project_id
        {where}
    """

    # Total unit count
    total = conn.execute(f"SELECT COUNT(*) {base_sql}", params).fetchone()[0]

    # Projects by city
    city_rows = conn.execute(
        f"SELECT p.city, COUNT(DISTINCT p.project_id) as cnt {base_sql} GROUP BY p.city ORDER BY cnt DESC",
        params,
    ).fetchall()

    # BHK breakdown
    bhk_rows = conn.execute(
        f"SELECT u.bhk, COUNT(*) as cnt {base_sql} GROUP BY u.bhk ORDER BY u.bhk",
        params,
    ).fetchall()

    # Property type breakdown
    type_rows = conn.execute(
        f"SELECT u.property_type, COUNT(*) as cnt {base_sql} GROUP BY u.property_type ORDER BY cnt DESC",
        params,
    ).fetchall()

    return {
        "total_units":      total,
        "by_city":          [dict(r) for r in city_rows],
        "by_bhk":           [dict(r) for r in bhk_rows],
        "by_property_type": [dict(r) for r in type_rows],
    }


# ─── Detail Query Handler ──────────────────────────────────────────────────────

def detail_search(plan: SearchPlan) -> list[dict]:
    """
    Fetch full project details including all rooms for DETAIL queries.
    """
    conn = _get_conn()

    # Find project by name using FTS first, then fallback to LIKE
    project_ids = list(fts5_search(plan))

    if not project_ids and plan.project_name_query:
        rows = conn.execute(
            "SELECT project_id FROM projects WHERE LOWER(project_name) LIKE ?",
            (f"%{plan.project_name_query.lower()}%",),
        ).fetchall()
        project_ids = [r["project_id"] for r in rows]

    if not project_ids:
        return []

    results = []
    for pid in project_ids[:2]:  # max 2 projects for detail
        proj_row = conn.execute("SELECT * FROM projects WHERE project_id = ?", (pid,)).fetchone()
        if not proj_row:
            continue
        proj = dict(proj_row)

        units = conn.execute("SELECT * FROM units WHERE project_id = ?", (pid,)).fetchall()
        proj["matching_units"] = []

        for u in units:
            unit = dict(u)
            rooms = conn.execute(
                "SELECT * FROM rooms WHERE unit_id = ? ORDER BY id", (u["unit_id"],)
            ).fetchall()
            unit["rooms"] = [dict(r) for r in rooms]
            proj["matching_units"].append(unit)

        proj["nearby_landmarks"] = [r[0] for r in conn.execute(
            "SELECT landmark_name FROM project_landmarks WHERE project_id = ?", (pid,)
        ).fetchall()]
        proj["amenities"] = [r[0] for r in conn.execute(
            "SELECT amenity FROM project_amenities WHERE project_id = ?", (pid,)
        ).fetchall()]

        proj["relevance_score"] = 1.0   # detail queries are always 100% relevant
        results.append(proj)

    return results


# ─── Public API ───────────────────────────────────────────────────────────────

def search(plan: SearchPlan, top_k: int = 5) -> dict:
    """
    Main search function. Routes based on query_type.

    Returns:
        {
          "query_type": "SEARCH" | "AGGREGATE" | "COMPARE" | "DETAIL",
          "results": [...]     # list of project dicts for SEARCH/COMPARE/DETAIL
          "aggregate": {...}   # summary dict for AGGREGATE
        }
    """
    if plan.needs_clarification:
        return {"query_type": "SEARCH", "results": [], "aggregate": None}

    qt = plan.query_type.upper()

    if qt == "AGGREGATE":
        agg = aggregate_search(plan)
        return {"query_type": "AGGREGATE", "results": [], "aggregate": agg}

    if qt == "DETAIL":
        results = detail_search(plan)
        return {"query_type": "DETAIL", "results": results, "aggregate": None}

    # SEARCH and COMPARE — full pipeline
    sql_results = sql_search(plan, limit=settings.MAX_SQL_CANDIDATES)

    if not sql_results:
        logger.info("No SQL results — falling back to pure semantic search.")

    fts5_hits = fts5_search(plan)
    vector_scores = vector_search(plan, top_k=settings.MAX_VECTOR_RESULTS)

    # For pure semantic fallback (no SQL results), include vector-only hits
    if not sql_results and vector_scores:
        conn_tmp = _get_conn()
        for uid, _ in vector_scores[:20]:
            row = conn_tmp.execute(
                """SELECT u.*, p.project_id as proj_id, p.project_name, p.developer_name,
                   p.city, p.neighbourhood, p.address, p.project_status, p.possession_date,
                   p.has_clubhouse, p.has_pool, p.has_park, p.has_sports_courts,
                   p.has_parking, p.has_commercial_shops, p.total_buildings,
                   p.total_villas, p.society_description
                   FROM units u JOIN projects p ON u.project_id = p.project_id
                   WHERE u.unit_id = ?""",
                (uid,),
            ).fetchone()
            if row:
                sql_results.append(dict(row))

    results = merge_and_group(sql_results, fts5_hits, vector_scores, top_k=top_k)
    return {"query_type": qt, "results": results, "aggregate": None}
