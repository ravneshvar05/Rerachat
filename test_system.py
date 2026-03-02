"""
test_system.py — Tests the chatbot pipeline end-to-end without the UI.

Run with:
    .\venv\Scripts\python test_system.py

Tests:
  1. SQLite — project/unit counts and sample data
  2. Query parser — extracts filters from natural language
  3. Search — returns correct results for known queries
  4. Answer generator — writes a natural language recommendation
"""
# knowldge graph
import json
from query_parser import parse_query
from search import search
from answer_generator import generate_answer

SEPARATOR = "─" * 60

def section(title):
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


# ─── Test 1: SQLite DB Check ─────────────────────────────────────────────────

section("TEST 1: SQLite Database")

import sqlite3
from config import settings

conn = sqlite3.connect(settings.sqlite_path)
conn.row_factory = sqlite3.Row

project_count = conn.execute("SELECT count(*) FROM projects").fetchone()[0]
unit_count    = conn.execute("SELECT count(*) FROM units").fetchone()[0]

print(f"✓ Projects in DB  : {project_count}")
print(f"✓ Units in DB     : {unit_count}")

print("\nSample units:")
rows = conn.execute("""
    SELECT p.project_name, u.bhk, u.property_type, p.city, u.unit_type
    FROM units u JOIN projects p ON u.project_id = p.project_id
    LIMIT 5
""").fetchall()
for r in rows:
    print(f"  • {r['bhk']} BHK {r['property_type']} — {r['project_name']} ({r['city']})")
conn.close()


# ─── Test 2: ChromaDB Check ───────────────────────────────────────────────────

section("TEST 2: ChromaDB Vector Store")

import chromadb
client = chromadb.PersistentClient(path=str(settings.chroma_path))
col = client.get_collection(settings.CHROMA_COLLECTION_NAME)
print(f"✓ Vectors in ChromaDB: {col.count()}")


# ─── Test 3: Query Parser ─────────────────────────────────────────────────────

section("TEST 3: Query Parser (Groq LLM)")

test_queries = [
    "I want a 3 BHK villa in Ahmedabad with a pooja room",
    "Show me 2 BHK apartments with gym",
    "I need something cheap",                    # should trigger clarification
]

for q in test_queries:
    print(f"\n  Query: \"{q}\"")
    result = parse_query(q)
    if result.needs_clarification:
        print(f"  → Needs clarification: {result.clarification_question[:80]}...")
    else:
        print(f"  → city={result.city}, bhk={result.bhk}, type={result.property_type}")
        print(f"     pooja={result.must_have_pooja_room}, gym={result.must_have_gym}")
        print(f"     semantic: {result.semantic_query[:60]}...")


# ─── Test 4: Search ───────────────────────────────────────────────────────────

section("TEST 4: Dual Search Engine")

test_cases = [
    ("3 BHK Villa Ahmedabad",         dict(city="Ahmedabad", bhk=3, property_type="VILLA")),
    ("2 BHK Apartment Ahmedabad",     dict(city="Ahmedabad", bhk=2, property_type="APARTMENT")),
    ("1 BHK Apartment",               dict(bhk=1, property_type="APARTMENT")),
]

from query_parser import ParsedQuery

for label, kwargs in test_cases:
    pq = ParsedQuery(semantic_query=label, **kwargs)
    results = search(pq, top_k=3)
    print(f"\n  Query: \"{label}\" → {len(results)} result(s)")
    for r in results:
        area = f"{r['super_builtup_sqft']} sqft" if r.get("super_builtup_sqft") else "area N/A"
        print(f"    • {r['project_name']} — {r['unit_type']} — {area}")


# ─── Test 5: End-to-End (Search + Answer) ────────────────────────────────────

section("TEST 5: Full End-to-End — Search + Groq Answer")

user_query = "I want a 3 BHK villa in Ahmedabad with pooja room and garden"
print(f"  User: \"{user_query}\"")

parsed = parse_query(user_query)
print(f"  Parsed: city={parsed.city}, bhk={parsed.bhk}, type={parsed.property_type}, pooja={parsed.must_have_pooja_room}")

results = search(parsed, top_k=3)
print(f"  Found {len(results)} matching unit(s)")

if results:
    answer = generate_answer(user_query, results)
    print(f"\n  Bot Response:\n")
    # Indent each line
    for line in answer.split("\n"):
        print(f"    {line}")
else:
    print("  No results found.")

print(f"\n{SEPARATOR}")
print("  All tests complete ✓")
print(SEPARATOR)
