import sqlite3
conn = sqlite3.connect('db/projects_v2.db')
conn.row_factory = sqlite3.Row

print('=== TABLES ===')
for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
    print(r[0])

print('\n=== PROPERTY TYPES ===')
for r in conn.execute('SELECT DISTINCT property_type, COUNT(*) as cnt FROM units GROUP BY property_type').fetchall():
    print(dict(r))

print('\n=== BHK VALUES ===')
for r in conn.execute('SELECT DISTINCT bhk, COUNT(*) as cnt FROM units GROUP BY bhk ORDER BY bhk').fetchall():
    print(dict(r))

print('\n=== NEIGHBOURHOODS (all unique) ===')
for r in conn.execute('SELECT DISTINCT city, neighbourhood FROM projects ORDER BY city, neighbourhood').fetchall():
    print(dict(r))

print('\n=== FTS5 landmarks sample ===')
for r in conn.execute('SELECT project_id, neighbourhood, landmarks_text FROM projects_fts LIMIT 10').fetchall():
    print(dict(r))

print('\n=== Searching Sarkhej in FTS ===')
try:
    rows = conn.execute('SELECT project_id, neighbourhood FROM projects_fts WHERE projects_fts MATCH ? ORDER BY rank', ('"Sarkhej"',)).fetchall()
    for r in rows:
        print(dict(r))
    print(f'Total: {len(rows)}')
except Exception as e:
    print(f'Error: {e}')

print('\n=== Searching Vizol/Vinzol in FTS ===')
try:
    rows = conn.execute('SELECT project_id, neighbourhood FROM projects_fts WHERE projects_fts MATCH ? ORDER BY rank', ('"Vizol" OR "Vinzol"',)).fetchall()
    for r in rows:
        print(dict(r))
    print(f'Total: {len(rows)}')
except Exception as e:
    print(f'Error: {e}')

print('\n=== Searching Makarba in FTS ===')
try:
    rows = conn.execute('SELECT project_id, neighbourhood FROM projects_fts WHERE projects_fts MATCH ? ORDER BY rank', ('"Makarba"',)).fetchall()
    for r in rows:
        print(dict(r))
    print(f'Total: {len(rows)}')
except Exception as e:
    print(f'Error: {e}')

print('\n=== Searching Ognaj in FTS ===')
try:
    rows = conn.execute('SELECT project_id, neighbourhood FROM projects_fts WHERE projects_fts MATCH ? ORDER BY rank', ('"Ognaj"',)).fetchall()
    for r in rows:
        print(dict(r))
    print(f'Total: {len(rows)}')
except Exception as e:
    print(f'Error: {e}')

print('\n=== Direct neighbourhood search for all areas ===')
for area in ['ognaj', 'makarba', 'sarkhej', 'vizol', 'vinzol', 'karnavati', 'bopal']:
    rows = conn.execute(f"SELECT DISTINCT neighbourhood, address FROM projects WHERE LOWER(neighbourhood) LIKE '%{area}%' OR LOWER(address) LIKE '%{area}%'").fetchall()
    if rows:
        for r in rows:
            print(f'  {area}: {dict(r)}')

print('\n=== VILLAS in DB ===')
for r in conn.execute("SELECT p.project_name, p.neighbourhood, u.bhk, u.property_type FROM units u JOIN projects p ON u.project_id = p.project_id WHERE u.property_type='VILLA' LIMIT 20").fetchall():
    print(dict(r))

print('\n=== ROW_HOUSE in DB ===')
for r in conn.execute("SELECT p.project_name, p.neighbourhood, u.bhk, u.property_type FROM units u JOIN projects p ON u.project_id = p.project_id WHERE u.property_type='ROW_HOUSE' LIMIT 20").fetchall():
    print(dict(r))

print('\n=== TENEMENT in DB ===')
for r in conn.execute("SELECT p.project_name, p.neighbourhood, u.bhk, u.property_type FROM units u JOIN projects p ON u.project_id = p.project_id WHERE u.property_type='TENEMENT' LIMIT 20").fetchall():
    print(dict(r))

print('\n=== 3 BHK Ahmedabad units sample ===')
for r in conn.execute("SELECT p.project_name, p.neighbourhood, u.bhk, u.property_type FROM units u JOIN projects p ON u.project_id = p.project_id WHERE u.bhk=3 AND LOWER(p.city)='ahmedabad' LIMIT 10").fetchall():
    print(dict(r))
