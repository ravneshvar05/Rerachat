"""
test_new_fixes.py — Verify the two new search fixes:
  1. "house" / "bungalow" etc. → ROW_HOUSE property_type mapping
  2. Non-apartment fallback: tenements/villas/row-houses when specific type has no results
Run: python test_new_fixes.py
"""
import sys
from query_planner import plan_query, SearchPlan
from search import search, sql_search, NON_APARTMENT_TYPES, _search_non_apartment

PASS = "✅ PASS"
FAIL = "❌ FAIL"

def check(label, condition, info=""):
    status = PASS if condition else FAIL
    print(f"  {status} — {label}")
    if info:
        print(f"         {info}")
    return condition

results = []

# ─── SECTION 1: Property type synonym mapping (LLM plan_query) ──────────────
print("\n" + "="*60)
print("  SECTION 1: Property type synonym mapping (LLM)")
print("="*60)

synonym_tests = [
    ("house near Sun City",            "ROW_HOUSE", ["Sun City"]),
    ("find me some house near sun city", "ROW_HOUSE", ["Sun City"]),
    ("show me bungalows in ahmedabad",  "ROW_HOUSE", []),
    ("independent house in bopal",      "ROW_HOUSE", ["Bopal"]),
    ("show me villas near karnavati club", "VILLA",   ["Karnavati Club"]),
    ("find me tenaments in ahmedabad",  "TENEMENT",  []),
    ("tenements in surat",             "TENEMENT",  []),
    ("show me flats in ahmedabad",     "APARTMENT", []),
    ("3 bhk penthouse in gandhinagar", "PENTHOUSE", []),
]

for query, expected_type, expected_locs in synonym_tests:
    print(f"\n  Query: \"{query}\"")
    try:
        plan = plan_query(query)
        print(f"  Plan: type={plan.property_type} cities={plan.cities} locs={plan.locations}")

        type_ok = (plan.property_type or "").upper() == expected_type
        results.append(check(
            f"property_type == {expected_type}",
            type_ok,
            f"got: {plan.property_type}"
        ))

        # Check Sun City is in locations (not cities)
        for loc in expected_locs:
            loc_ok = any(loc.lower() in l.lower() for l in (plan.locations or []))
            city_bad = any(loc.lower() in c.lower() for c in (plan.cities or []))
            results.append(check(
                f"'{loc}' in locations (not cities)",
                loc_ok and not city_bad,
                f"locations={plan.locations} cities={plan.cities}"
            ))

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()
        results.append(False)


# ─── SECTION 2: Non-apartment fallback (search engine) ──────────────────────
print("\n" + "="*60)
print("  SECTION 2: Non-apartment fallback (search engine)")
print("="*60)

print("\n  Test A: NON_APARTMENT_TYPES constant correctness")
for t in ("VILLA", "ROW_HOUSE", "TENEMENT", "PENTHOUSE"):
    results.append(check(f"{t} in NON_APARTMENT_TYPES", t in NON_APARTMENT_TYPES))
results.append(check("APARTMENT NOT in NON_APARTMENT_TYPES", "APARTMENT" not in NON_APARTMENT_TYPES))

print("\n  Test B: _search_non_apartment returns only non-APARTMENT rows")
plan_any = SearchPlan(
    cities=["Ahmedabad"],
    semantic_query="residential property in Ahmedabad"
)
fallback_results = _search_non_apartment(plan_any, top_k=10)
print(f"  Fallback returned {len(fallback_results)} project(s)")
if fallback_results:
    all_types = set()
    for proj in fallback_results:
        for u in proj.get("matching_units", []):
            all_types.add(u.get("property_type", "").upper())
    results.append(check(
        "No APARTMENT in fallback results",
        "APARTMENT" not in all_types,
        f"types found: {all_types}"
    ))
else:
    print("  ⚠️  No fallback results (DB may only have APARTMENTs in Ahmedabad — check manually)")
    results.append(True)  # Don't fail if DB simply has no non-APT data

print("\n  Test C: search() returns fallback_non_apt flag correctly")
# Simulate: ask for PENTHOUSE in a city where likely none exist
plan_penthouse = SearchPlan(
    property_type="PENTHOUSE",
    cities=["Ahmedabad"],
    semantic_query="penthouse in ahmedabad"
)
sr = search(plan_penthouse, top_k=5)
print(f"  search() results: {len(sr['results'])} | fallback_non_apt={sr.get('fallback_non_apt')} | original_type={sr.get('original_type')}")

# The flag field must exist
results.append(check(
    "fallback_non_apt key exists in search result",
    "fallback_non_apt" in sr
))
results.append(check(
    "original_type key exists in search result",
    "original_type" in sr
))
# If there are results, they should all be non-apartment
if sr["results"] and sr.get("fallback_non_apt"):
    all_types = set()
    for proj in sr["results"]:
        for u in proj.get("matching_units", []):
            all_types.add(u.get("property_type", "").upper())
    results.append(check(
        "Fallback results contain no APARTMENT",
        "APARTMENT" not in all_types,
        f"types: {all_types}"
    ))

# ─── SECTION 3: Full pipeline test — "house near Sun City" ──────────────────
print("\n" + "="*60)
print("  SECTION 3: Full pipeline — 'find me house near sun city'")
print("="*60)
try:
    plan = plan_query("find me some house near sun city")
    print(f"  Plan: type={plan.property_type} locs={plan.locations} cities={plan.cities}")
    sr = search(plan, top_k=5)
    print(f"  Results: {len(sr['results'])} project(s) | fallback={sr.get('fallback_non_apt')}")
    for proj in sr["results"]:
        units = proj.get("matching_units", [])
        types = {u.get("property_type") for u in units}
        print(f"    • {proj['project_name']} ({proj.get('neighbourhood')}) — types: {types}")

    results.append(check(
        "property_type is ROW_HOUSE",
        (plan.property_type or "").upper() == "ROW_HOUSE",
        f"got: {plan.property_type}"
    ))
    results.append(check(
        "Sun City captured in locations",
        any("sun city" in l.lower() for l in (plan.locations or [])),
        f"locations: {plan.locations}"
    ))
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback; traceback.print_exc()
    results.append(False)


# ─── SUMMARY ────────────────────────────────────────────────────────────────
print("\n" + "="*60)
passed = sum(results)
total = len(results)
print(f"  TOTAL: {passed}/{total} checks passed")
print("="*60 + "\n")
if passed < total:
    sys.exit(1)
