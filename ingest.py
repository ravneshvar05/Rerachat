"""
ingest.py — Reads all brochure JSONs and populates:
  1. SQLite  → structured, filterable data (projects + units tables)
  2. ChromaDB → semantic vector embeddings (one document per unit)

Run this every time you add new JSONs to the output folder.
Only NEW projects (not already in DB) will be inserted.

Usage:
    python ingest.py
"""

import json
import sqlite3
import uuid
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from config import settings
from logger import logger


# ─── SQLite Setup ─────────────────────────────────────────────────────────────

def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.sqlite_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # better concurrent reads
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def create_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS projects (
            project_id          TEXT PRIMARY KEY,
            project_name        TEXT,
            developer_name      TEXT,
            city                TEXT,
            neighbourhood       TEXT,
            rera_number         TEXT,
            project_status      TEXT,
            possession_date     TEXT,
            amenities_text      TEXT,   -- joined string for FTS
            has_clubhouse       INTEGER,
            has_pool            INTEGER,
            has_park            INTEGER,
            has_sports_courts   INTEGER,
            has_parking         INTEGER,
            total_units         INTEGER
        );

        CREATE TABLE IF NOT EXISTS units (
            unit_id             TEXT PRIMARY KEY,
            project_id          TEXT REFERENCES projects(project_id),
            unit_type           TEXT,
            property_type       TEXT,       -- APARTMENT / VILLA / ROW_HOUSE ...
            bhk                 INTEGER,
            carpet_area_sqft    REAL,
            super_builtup_sqft  REAL,
            balcony_area_sqft   REAL,
            wash_area_sqft      REAL,
            applicable_buildings TEXT,      -- stored as comma-separated string
            description         TEXT,
            -- derived room flags (for fast filtering)
            has_pooja_room      INTEGER DEFAULT 0,
            has_study_room      INTEGER DEFAULT 0,
            has_terrace         INTEGER DEFAULT 0,
            has_servant_room    INTEGER DEFAULT 0,
            has_garden          INTEGER DEFAULT 0,
            has_home_theatre    INTEGER DEFAULT 0,
            has_gym             INTEGER DEFAULT 0,
            attached_bathrooms  INTEGER DEFAULT 0,
            total_rooms         INTEGER DEFAULT 0
        );
    """)
    conn.commit()
    logger.info("SQLite tables ready.")


# ─── ChromaDB Setup ───────────────────────────────────────────────────────────

def get_chroma_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(settings.chroma_path))
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.get_or_create_collection(
        name=settings.CHROMA_COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info(f"ChromaDB collection '{settings.CHROMA_COLLECTION_NAME}' ready.")
    return collection


# ─── Text Builder (what gets embedded) ───────────────────────────────────────

def build_unit_text(project: dict, unit: dict) -> str:
    """Build a rich plain-text description of a unit for embedding."""
    rooms = unit.get("rooms", [])
    room_names = ", ".join(r["name"] for r in rooms if r.get("name"))
    amenities = ", ".join(project.get("amenities", []))
    desc = unit.get("description", "") or ""
    loc = project.get("location", {})
    society = project.get("society_layout", {})

    text = (
        f"{unit.get('bhk', '?')} BHK {unit.get('property_type', '')} "
        f"in {loc.get('city', '')} "
        f"by {project.get('developer_name', '')} "
        f"({project.get('project_name', '')}). "
        f"Unit type: {unit.get('unit_type', '')}. "
    )
    if unit.get("super_built_up_area_sqft"):
        text += f"Super built-up area: {unit['super_built_up_area_sqft']} sqft. "
    if unit.get("carpet_area_sqft"):
        text += f"Carpet area: {unit['carpet_area_sqft']} sqft. "
    if desc:
        text += f"{desc} "
    if room_names:
        text += f"Rooms: {room_names}. "
    if amenities:
        text += f"Project amenities: {amenities}. "
    if society.get("description"):
        text += f"{society['description']}"
    return text.strip()


# ─── Room Flag Extraction ─────────────────────────────────────────────────────

ROOM_FLAGS = {
    "has_pooja_room":    {"POOJA_ROOM"},
    "has_study_room":    {"STUDY_ROOM"},
    "has_terrace":       {"TERRACE"},
    "has_servant_room":  {"SERVANT_ROOM"},
    "has_garden":        {"OTHER"},         # detected by name keyword below
    "has_home_theatre":  {"OTHER"},
    "has_gym":           {"OTHER"},
}

GARDEN_KEYWORDS   = {"garden", "courtyard"}
THEATRE_KEYWORDS  = {"theatre", "theater", "home theatre"}
GYM_KEYWORDS      = {"gym", "gymnasium"}


def extract_room_flags(rooms: list) -> dict:
    flags = {k: 0 for k in ROOM_FLAGS}
    attached = 0
    for room in rooms:
        rt = room.get("room_type", "")
        name = (room.get("name") or "").lower()

        if rt == "POOJA_ROOM":
            flags["has_pooja_room"] = 1
        elif rt == "STUDY_ROOM":
            flags["has_study_room"] = 1
        elif rt == "TERRACE":
            flags["has_terrace"] = 1
        elif rt == "SERVANT_ROOM":
            flags["has_servant_room"] = 1

        # name-based detection for OTHER types
        if any(k in name for k in GARDEN_KEYWORDS):
            flags["has_garden"] = 1
        if any(k in name for k in THEATRE_KEYWORDS):
            flags["has_home_theatre"] = 1
        if any(k in name for k in GYM_KEYWORDS):
            flags["has_gym"] = 1

        # count attached bathrooms
        if rt == "BEDROOM" and room.get("attached_bathroom"):
            attached += 1

    flags["attached_bathrooms"] = attached
    flags["total_rooms"] = len(rooms)
    return flags


# ─── Ingestion Logic ──────────────────────────────────────────────────────────

def is_project_ingested(conn: sqlite3.Connection, project_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM projects WHERE project_id = ?", (project_id,)
    ).fetchone()
    return row is not None


def ingest_project(
    conn: sqlite3.Connection,
    collection: chromadb.Collection,
    data: dict,
    source_file: str,
) -> None:
    project_id = data.get("project_id") or str(uuid.uuid4())

    if is_project_ingested(conn, project_id):
        logger.info(f"  ↳ Already ingested: {project_id} — skipping.")
        return

    loc = data.get("location", {})
    society = data.get("society_layout", {})
    amenities_text = " | ".join(data.get("amenities", []))

    # ── Insert project row ──
    conn.execute(
        """INSERT INTO projects VALUES (
            :project_id, :project_name, :developer_name, :city, :neighbourhood,
            :rera_number, :project_status, :possession_date, :amenities_text,
            :has_clubhouse, :has_pool, :has_park, :has_sports_courts,
            :has_parking, :total_units
        )""",
        {
            "project_id":      project_id,
            "project_name":    data.get("project_name"),
            "developer_name":  data.get("developer_name"),
            "city":            loc.get("city"),
            "neighbourhood":   loc.get("neighbourhood"),
            "rera_number":     data.get("rera_registration_number"),
            "project_status":  data.get("project_status"),
            "possession_date": data.get("possession_date"),
            "amenities_text":  amenities_text,
            "has_clubhouse":   int(bool(society.get("has_clubhouse"))),
            "has_pool":        int(bool(society.get("has_swimming_pool"))),
            "has_park":        int(bool(society.get("has_park_or_garden"))),
            "has_sports_courts": int(bool(society.get("has_sports_courts"))),
            "has_parking":     int(bool(society.get("has_parking_area"))),
            "total_units":     society.get("total_units_in_project"),
        },
    )

    # ── Insert units + add to ChromaDB ──
    units = data.get("units", [])
    chroma_docs, chroma_meta, chroma_ids = [], [], []

    for idx, unit in enumerate(units):
        # ensure unique ID by appending loop index if needed, because sometimes unit_type is duplicated
        unit_id = f"{project_id}__{unit.get('unit_type', uuid.uuid4())}_{idx}"
        # make safe ID for ChromaDB (no spaces or special chars)
        unit_id_safe = unit_id.replace(" ", "_").replace("/", "-")

        flags = extract_room_flags(unit.get("rooms", []))

        conn.execute(
            """INSERT OR IGNORE INTO units VALUES (
                :unit_id, :project_id, :unit_type, :property_type, :bhk,
                :carpet_area_sqft, :super_builtup_sqft, :balcony_area_sqft,
                :wash_area_sqft, :applicable_buildings, :description,
                :has_pooja_room, :has_study_room, :has_terrace,
                :has_servant_room, :has_garden, :has_home_theatre, :has_gym,
                :attached_bathrooms, :total_rooms
            )""",
            {
                "unit_id":             unit_id_safe,
                "project_id":          project_id,
                "unit_type":           unit.get("unit_type"),
                "property_type":       unit.get("property_type"),
                "bhk":                 unit.get("bhk"),
                "carpet_area_sqft":    unit.get("carpet_area_sqft"),
                "super_builtup_sqft":  unit.get("super_built_up_area_sqft"),
                "balcony_area_sqft":   unit.get("balcony_area_sqft"),
                "wash_area_sqft":      unit.get("wash_area_sqft"),
                "applicable_buildings": ",".join(unit.get("applicable_buildings") or []),
                "description":         unit.get("description"),
                **flags,
            },
        )

        # Build the text that gets embedded
        text = build_unit_text(data, unit)
        chroma_docs.append(text)
        chroma_ids.append(unit_id_safe)
        chroma_meta.append({
            "project_id":    project_id,
            "unit_id":       unit_id_safe,
            "project_name":  data.get("project_name", ""),
            "city":          loc.get("city", ""),
            "bhk":           unit.get("bhk") or 0,
            "property_type": unit.get("property_type", ""),
            "area_sqft":     unit.get("super_built_up_area_sqft") or 0.0,
        })

    # Add to ChromaDB in one batch
    if chroma_docs:
        collection.upsert(
            documents=chroma_docs,
            metadatas=chroma_meta,
            ids=chroma_ids,
        )

    conn.commit()
    logger.success(f"  ✓ Ingested: {data.get('project_name')} — {len(units)} unit(s)")


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_ingestion() -> None:
    json_dir = settings.JSON_OUTPUT_DIR
    json_files = list(json_dir.glob("*.json"))

    if not json_files:
        logger.warning(f"No JSON files found in {json_dir}. Nothing to ingest.")
        return

    logger.info(f"Found {len(json_files)} JSON file(s) in '{json_dir}'")

    conn = get_db_connection()
    collection = get_chroma_collection()
    create_tables(conn)

    ingested, skipped = 0, 0
    for json_file in json_files:
        logger.info(f"Processing: {json_file.name}")
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            project_id = data.get("project_id", "")

            if is_project_ingested(conn, project_id):
                logger.info(f"  ↳ Already ingested: {json_file.name} — skipping.")
                skipped += 1
            else:
                ingest_project(conn, collection, data, json_file.name)
                ingested += 1

        except Exception as e:
            logger.error(f"  ✗ Failed to process {json_file.name}: {e}")

    conn.close()
    logger.info(
        f"\n{'─'*50}\n"
        f"Ingestion complete: {ingested} new, {skipped} skipped.\n"
        f"SQLite  → {settings.sqlite_path}\n"
        f"ChromaDB → {settings.chroma_path}\n"
        f"{'─'*50}"
    )


if __name__ == "__main__":
    run_ingestion()
