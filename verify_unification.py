import urllib.request
import json

def test_url(name, url, method="GET", data=None, headers=None):
    if headers is None:
        headers = {}
    if data is not None and isinstance(data, dict):
        data = json.dumps(data).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as res:
            status = res.status
            content = res.read().decode("utf-8", errors="ignore")
            print(f"[OK {status}] {name}")
            return status, content
    except Exception as e:
        print(f"[FAIL] {name}: {e}")
        return None, str(e)

print("=== 1. VERIFICANDO RUTAS FRONTEND NEXT.JS (PORT 3000) ===")
test_url("Landing / Login Page", "http://localhost:3000/auth/login")
test_url("Admin Page", "http://localhost:3000/admin")
test_url("Invite Page", "http://localhost:3000/auth/invite")
test_url("Traffic Vision Page", "http://localhost:3000/traffic")
test_url("Devices Page", "http://localhost:3000/devices")
test_url("Dashboard Page", "http://localhost:3000/dashboard")

print("\n=== 2. VERIFICANDO BACKEND UNIFICADO & FLUJO DE INVITACIÓN (PORT 8000) ===")
status, login_res = test_url("Admin Login", "http://localhost:8000/auth/login", "POST", {"email": "admin@apexcompany.com.mx", "password": "admin123"})
admin_data = json.loads(login_res)
token = admin_data["access_token"]
print(f"   -> Admin autenticado: {admin_data['user']['nombre']} ({admin_data['user']['rol']})")

status, invite_res = test_url("Admin Invite User", "http://localhost:8000/admin/invite", "POST", {
    "nombre": "Lic. Carlos Vega",
    "email": "carlos.vega@liverpool.com.mx",
    "rol": "cliente",
    "cliente_id": "cli_retail"
})
invite_data = json.loads(invite_res)
inv_token = invite_data["invitation_token"]
inv_url = invite_data["invitation_url"]
print(f"   -> Invitacion creada exitosamente:")
print(f"      Token: {inv_token}")
print(f"      URL: {inv_url}")

status, verify_res = test_url("Verify Invitation Token", f"http://localhost:8000/auth/invite/verify?token={inv_token}")
verify_data = json.loads(verify_res)
print(f"   -> Verificacion valida: {verify_data['valid']}, Empresa: {verify_data['invitation']['cliente_nombre']}")

status, accept_res = test_url("Accept Invitation & Set Password", "http://localhost:8000/auth/invite/accept", "POST", {
    "token": inv_token,
    "password": "liverpool2026"
})
accept_data = json.loads(accept_res)
print(f"   -> Cuenta activada exitosamente: {accept_data['user']['nombre']} ({accept_data['user']['email']})")

status, client_login_res = test_url("Login with New Account", "http://localhost:8000/auth/login", "POST", {
    "email": "carlos.vega@liverpool.com.mx",
    "password": "liverpool2026"
})
new_login = json.loads(client_login_res)
print(f"   -> Login con nueva cuenta y credenciales activadas: OK! Token={new_login['access_token'][:20]}...")

status, inv_list_res = test_url("List Invitations in Admin", "http://localhost:8000/admin/invitations")
invs = json.loads(inv_list_res)
print(f"   -> Total invitaciones en sistema: {len(invs)} (Ultima: {invs[0]['email']} - Estado: {invs[0]['estado']})")
