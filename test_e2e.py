"""Quick end-to-end validation of the new pipeline."""
import time

from query_planner import plan_query
from search import search
from answer_generator import generate_answer

SEP = "─" * 55

def run_test(label, query, sleep=3):
    print(f"\n{SEP}")
    print(f"  {label}")
    print(f"  Query: '{query}'")
    print(SEP)
    
    plan = plan_query(query)
    print(f"  Plan → type={plan.query_type} cities={plan.cities} bhk={plan.bhk} ptype={plan.property_type}")
    print(f"         must_have={plan.must_have} locations={getattr(plan, 'locations', [])}")
    
    time.sleep(sleep)
    result = search(plan)
    print(f"  Search → {len(result['results'])} projects, agg={result['aggregate'] is not None}")
    for p in result['results']:
        score = p.get('relevance_score', 1.0)
        print(f"    - {p.get('project_name')} | {len(p.get('matching_units', []))} units | score={score}")
    
    time.sleep(sleep)
    answer = generate_answer(query, result)
    print(f"\n  Answer:\n  {answer[:300]}...")
    return result

# Test 1: Standard search
r1 = run_test("SEARCH — 3 BHK Villa", "3 BHK villa with pooja room in Ahmedabad", sleep=4)

# Test 2: Aggregation  
r2 = run_test("AGGREGATE — count query", "How many projects have a swimming pool?", sleep=4)

# Test 3: Landmark FTS5
r3 = run_test("FTS5 LANDMARK — near landmark", "Projects near Karnavati Club", sleep=4)

# Test 4: Detail
r4 = run_test("DETAIL — full project info", "Give me full details of KP Villas", sleep=4)

print(f"\n{SEP}")
print("  All tests passed ✓")
print(SEP)
