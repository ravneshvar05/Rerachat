import time
from query_planner import plan_query
from search import search

def run(q):
    print(f"\n--- Query: {q} ---")
    plan = plan_query(q)
    print(f"Locations parsed: {getattr(plan, 'locations', [])}")
    print(f"Semantic fallback limit: {plan.semantic_query}")
    
    res = search(plan)
    print(f"Projects found: {len(res['results'])}")
    for p in res['results']:
        score = p.get("relevance_score", 1.0)
        print(f" - {p.get('project_name')} in {p.get('city')}, {p.get('neighbourhood')} (score: {score})")

run("2bhk in sarkhej")
run("2bhk in sarkhej and vizol")
