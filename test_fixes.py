"""
test_fixes.py — Verifies all 8 query fixes work correctly.
Run: python test_fixes.py
"""
import sys
from query_planner import plan_query, SearchPlan
from search import search, fts5_search, sql_search

PASS = "✅ PASS"
FAIL = "❌ FAIL"
INFO = "ℹ️ INFO"

def check(label, condition, info=""):
    status = PASS if condition else FAIL
    print(f"  {status} — {label}")
    if info:
        print(f"         {info}")
    return condition

print("\n" + "="*60)
print("  PHASE 1: Direct search engine testing (no LLM)")
print("="*60)

# ─── Test FTS5 OR logic ─────────────────────────────────────
print("\n[1] FTS5: Sarkhej OR Vizol/Vinzol")
plan_loc = SearchPlan(
    locations=["Sarkhej", "Vizol", "Vinzol"],
    semantic_query="2 BHK apartment in Sarkhej or Vizol"
)
fts_hits = fts5_search(plan_loc)
check("FTS5 returns ≥1 project for Sarkhej/Vizol", len(fts_hits) >= 1, f"got: {fts_hits}")
check("Sarkhej project found", "AMBER_RESIDENCY_AHMEDABAD_001" in fts_hits, f"hits: {fts_hits}")
check("Vinzol project found", "OumOrbit_Brochure_1" in fts_hits, f"hits: {fts_hits}")

# ─── Test min_bhk SQL filter ─────────────────────────────────
print("\n[2] SQL: min_bhk for 'Villas 3 BHK or larger'")
plan_villa = SearchPlan(
    min_bhk=3,
    property_type="VILLA",
    semantic_query="villa 3 BHK or larger"
)
sql_res = sql_search(plan_villa, limit=50)
check("SQL returns results for min_bhk=3 + VILLA", len(sql_res) >= 1, f"got {len(sql_res)} rows")
bhks_found = set(r["bhk"] for r in sql_res)
check("All returned BHKs are >= 3", all(b >= 3 for b in bhks_found), f"BHKs: {bhks_found}")

# ─── Test score threshold fix (SQL-only results pass) ─────────
print("\n[3] Score threshold: 3 BHK apartments without vector context")
plan_3bhk = SearchPlan(
    bhk=3,
    cities=["Ahmedabad"],
    semantic_query="3 BHK apartment in Ahmedabad"
)
result3 = search(plan_3bhk, top_k=5)
check("Search returns results for 3 BHK Ahmedabad", len(result3["results"]) >= 1,
      f"got {len(result3['results'])} projects")

# ─── Test FTS SQL LIKE fallback ──────────────────────────────
print("\n[4] SQL LIKE fallback: neighbourhood search")
plan_like = SearchPlan(
    locations=["Sarkhej"],
    semantic_query="apartment near Sarkhej"
)
fts_sarkhej = fts5_search(plan_like)
check("Sarkhej found via FTS or SQL LIKE", len(fts_sarkhej) >= 1, f"got: {fts_sarkhej}")

print("\n" + "="*60)
print("  PHASE 2: Full pipeline testing (requires Groq LLM)")
print("="*60)

queries = [
    ("Q1 - Sarkhej OR Vizol 2BHK",
     "I am looking for a 2 BHK apartment strictly in Sarkhej or Vizol.",
     lambda r, p: (len(r["results"]) >= 1 and
                   any("Sarkhej" in (proj.get("neighbourhood") or "") or
                       "Vinzol" in (proj.get("neighbourhood") or "") or
                       "Vizol" in (proj.get("neighbourhood") or "")
                       for proj in r["results"]))),

    ("Q3 - Near Karnavati Club",
     "Find me any property near Karnavati Club.",
     lambda r, p: len(r["results"]) >= 1),

    ("Q5 - Villas 3 BHK or larger",
     "Show me only villas that are 3 BHK or larger.",
     lambda r, p: (len(r["results"]) >= 1 and p.property_type == "VILLA")),

    ("Q8 - List all 3 BHK apartments",
     "List all the 3 BHK apartments in the city.",
     lambda r, p: (r["query_type"] == "SEARCH" and len(r["results"]) >= 1)),
]

scores = []
for label, query, validator in queries:
    print(f"\n[{label}]")
    print(f"  Query: \"{query}\"")
    try:
        plan = plan_query(query)
        print(f"  Plan: type={plan.query_type} bhk={plan.bhk} min_bhk={getattr(plan,'min_bhk',None)} "
              f"ptype={plan.property_type} locs={plan.locations}")
        result = search(plan, top_k=5)
        print(f"  Results: {len(result['results'])} project(s), type={result['query_type']}")
        for proj in result["results"]:
            print(f"    • {proj['project_name']} ({proj.get('neighbourhood')}) — "
                  f"BHKs: {set(u.get('bhk') for u in proj.get('matching_units',[]))}")
        passed = validator(result, plan)
        check("Query returned expected results", passed)
        scores.append(passed)
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()
        scores.append(False)

print("\n" + "="*60)
total = sum(scores)
print(f"  LLM Pipeline: {total}/{len(scores)} passed")
print("="*60 + "\n")
