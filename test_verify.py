"""
test_verify.py -- Verification of synonym mapping and fallback logic.
Uses only ASCII output to avoid Windows charmap issues.
"""
import sys
from query_planner import plan_query, SearchPlan
from search import search, sql_search, NON_APARTMENT_TYPES, _search_non_apartment

results = []

def check(label, condition, info=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    if info:
        print(f"         {info}")
    return condition


# ============================================================
print("\n=== SECTION 1: Property Type Synonym Mapping (LLM) ===")
# ============================================================

synonym_tests = [
    ("house near Sun City",              "", ["Sun City"]),
    ("find me some house near sun city", "", ["Sun City"]),
    ("show me bungalows in ahmedabad",   "ROW_HOUSE", []),
    ("independent house in bopal",       "ROW_HOUSE", ["Bopal"]),
    ("find me tenaments in ahmedabad",   "TENEMENT",  []),
    ("tenements in surat",               "TENEMENT",  []),
    ("show me flats in ahmedabad",       "APARTMENT", []),
    ("3 bhk penthouse in gandhinagar",   "PENTHOUSE", []),
]

for query, expected_type, expected_locs in synonym_tests:
    print(f"\n  Query: [{query}]")
    try:
        plan = plan_query(query)
        print(f"  Plan : type={plan.property_type} cities={plan.cities} locs={plan.locations}")

        type_ok = (plan.property_type or "").upper() == expected_type
        results.append(check(f"property_type == {expected_type}", type_ok,
                             f"got: {plan.property_type}"))

        for loc in expected_locs:
            loc_ok  = any(loc.lower() in l.lower() for l in (plan.locations or []))
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


# ============================================================
print("\n\n=== SECTION 2: NON_APARTMENT_TYPES constant ===")
# ============================================================
for t in ("VILLA", "ROW_HOUSE", "TENEMENT", "PENTHOUSE"):
    results.append(check(f"{t} in NON_APARTMENT_TYPES", t in NON_APARTMENT_TYPES))
results.append(check("APARTMENT NOT in NON_APARTMENT_TYPES",
                     "APARTMENT" not in NON_APARTMENT_TYPES))


# ============================================================
print("\n\n=== SECTION 3: _search_non_apartment returns no APARTMENTs ===")
# ============================================================
plan_any = SearchPlan(
    cities=["Ahmedabad"],
    semantic_query="residential property in Ahmedabad"
)
fb = _search_non_apartment(plan_any, top_k=10)
print(f"  _search_non_apartment returned {len(fb)} project(s)")
if fb:
    all_types = set()
    for p in fb:
        for u in p.get("matching_units", []):
            all_types.add(u.get("property_type", "").upper())
    results.append(check("No APARTMENT in fallback results",
                         "APARTMENT" not in all_types, f"types: {all_types}"))
else:
    print("  [INFO] No non-APT projects in Ahmedabad (DB specific -- not a code error)")
    results.append(True)


# ============================================================
print("\n\n=== SECTION 4: search() fallback_non_apt flag ===")
# ============================================================
plan_pth = SearchPlan(
    property_type="PENTHOUSE",
    cities=["Ahmedabad"],
    semantic_query="penthouse in ahmedabad"
)
sr = search(plan_pth, top_k=5)
n = len(sr["results"])
fb_flag = sr.get("fallback_non_apt")
orig    = sr.get("original_type")
print(f"  search() results={n} | fallback_non_apt={fb_flag} | original_type={orig}")
results.append(check("'fallback_non_apt' key present in result dict", "fallback_non_apt" in sr))
results.append(check("'original_type' key present in result dict",    "original_type"    in sr))
results.append(check("original_type == 'PENTHOUSE'", orig == "PENTHOUSE", f"got: {orig}"))
if n and fb_flag:
    all_types = set()
    for proj in sr["results"]:
        for u in proj.get("matching_units", []):
            all_types.add(u.get("property_type", "").upper())
    results.append(check("Fallback results have no APARTMENT", "APARTMENT" not in all_types,
                         f"types: {all_types}"))


# ============================================================
print("\n\n=== SECTION 5: Full pipeline -- 'find me house near sun city' ===")
# ============================================================
try:
    plan = plan_query("find me some house near sun city")
    print(f"  Plan: type={plan.property_type} locs={plan.locations} cities={plan.cities}")
    sr = search(plan, top_k=5)
    print(f"  Results: {len(sr['results'])} project(s) | fallback={sr.get('fallback_non_apt')}")
    for proj in sr["results"]:
        units = proj.get("matching_units", [])
        types = {u.get("property_type") for u in units}
        print(f"    * {proj['project_name']} ({proj.get('neighbourhood')}) -- types: {types}")

    results.append(check("property_type is NULL (generic search)",
                         not plan.property_type,
                         f"got: {plan.property_type}"))
    results.append(check("Sun City captured in locations",
                         any("sun city" in l.lower() for l in (plan.locations or [])),
                         f"locations: {plan.locations}"))
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback; traceback.print_exc()
    results.append(False)


# ============================================================
print("\n\n=== SUMMARY ===")
# ============================================================
passed = sum(results)
total  = len(results)
print(f"  {passed}/{total} checks passed")
print("=" * 40)
if passed < total:
    sys.exit(1)
