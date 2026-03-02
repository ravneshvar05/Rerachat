import sys
sys.stdout.reconfigure(encoding='utf-8')
from query_planner import plan_query
from search import search

tests = [
    ('Q1: Sarkhej or Vizol 2BHK', 'I am looking for a 2 BHK apartment strictly in Sarkhej or Vizol.'),
    ('Q3: Near Karnavati Club', 'Find me any property near Karnavati Club.'),
    ('Q5: Villas 3 BHK or larger', 'Show me only villas that are 3 BHK or larger.'),
    ('Q8: List all 3 BHK apartments', 'List all the 3 BHK apartments in the city.'),
]

passed = 0
total = 4

for label, query in tests:
    print(f'\n--- {label} ---')
    print(f'Query: {query}')
    plan = plan_query(query)
    min_bhk = getattr(plan, 'min_bhk', None)
    print(f'Plan: type={plan.query_type} bhk={plan.bhk} min_bhk={min_bhk} ptype={plan.property_type} locs={plan.locations} cities={plan.cities}')
    result = search(plan, top_k=5)
    n = len(result['results'])
    qt = result['query_type']
    print(f'Results: {n} projects, query_type={qt}')
    for p in result['results']:
        bhks = set(u.get('bhk') for u in p.get('matching_units', []))
        print(f'  - {p["project_name"]} ({p.get("neighbourhood")}) BHKs:{bhks}')

    ok = False
    if 'Q1' in label:
        # Should find Sarkhej or Vinzol project
        ok = n >= 1 and any(
            'Sarkhej' in (p.get('neighbourhood') or '') or
            'Vinzol' in (p.get('neighbourhood') or '') or
            'Vizol' in (p.get('neighbourhood') or '')
            for p in result['results']
        )
    elif 'Q3' in label:
        ok = n >= 1
    elif 'Q5' in label:
        ok = n >= 1 and plan.property_type == 'VILLA'
    elif 'Q8' in label:
        ok = qt == 'SEARCH' and n >= 1

    status = 'PASS' if ok else 'FAIL'
    print(f'  => {status}')
    if ok:
        passed += 1

print(f'\n\nSummary: {passed}/{total} LLM pipeline tests passed')
