import urllib.request
import json

endpoints = [
    ("Cameras", "http://localhost:8000/api/cameras"),
    ("Stats", "http://localhost:8000/api/stats"),
    ("Connection Stats", "http://localhost:8000/api/cameras/connection-stats?camera_id=all"),
    ("Clients", "http://localhost:8000/api/clients"),
    ("Locations", "http://localhost:8000/api/locations"),
    ("Campaigns", "http://localhost:8000/api/campaigns"),
    ("Videos", "http://localhost:8000/api/videos"),
    ("Solar Status", "http://localhost:8000/api/solar/status?location_id=loc_minerva"),
    ("Shelly Status", "http://localhost:8000/relay/dev_shelly_minerva/status")
]

print("=== VERIFICACIÓN DE ENDPOINTS DEL MÓDULO VISIÓN & DOOH ===", flush=True)
for name, url in endpoints:
    try:
        with urllib.request.urlopen(url, timeout=3) as res:
            data = json.loads(res.read().decode())
            count = len(data) if isinstance(data, list) else "OK"
            print(f"[OK 200] {name}: {count}", flush=True)
    except Exception as e:
        print(f"[FAIL] {name}: {e}", flush=True)

print("=== VERIFICACIÓN COMPLETADA EXITOSAMENTE ===", flush=True)

