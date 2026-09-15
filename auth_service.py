"""
auth_service.py — Servicio criptográfico de autenticación, JWT y gestión de invitaciones
para la plataforma unificada Apex Company.
"""

import hmac
import hashlib
import base64
import json
import time
import secrets
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, Dict, Any

SECRET_KEY = "apex-company-secret-jwt-key-dooh-vision-2026"
ACCESS_TOKEN_EXPIRE_SECONDS = 86400 * 7  # 7 días de sesión
INVITATION_EXPIRE_SECONDS = 86400 * 7    # 7 días para aceptar invitación


def hash_password(password: str) -> str:
    """Genera un hash seguro PBKDF2-HMAC-SHA256 con salt aleatorio."""
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000)
    return f"{salt}:{dk.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Verifica si la contraseña coincide con el hash almacenado en tiempo constante."""
    try:
        if ":" not in stored_hash:
            return False
        salt, expected_hex = stored_hash.split(":", 1)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000)
        return hmac.compare_digest(dk.hex(), expected_hex)
    except Exception:
        return False


def _base64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def _base64url_decode(data: str) -> bytes:
    rem = len(data) % 4
    if rem > 0:
        data += "=" * (4 - rem)
    return base64.urlsafe_b64decode(data.encode("utf-8"))


def create_access_token(data: Dict[str, Any], expires_seconds: int = ACCESS_TOKEN_EXPIRE_SECONDS) -> str:
    """Crea un token JWT estándar firmado con HMAC-SHA256."""
    header = {"alg": "HS256", "typ": "JWT"}
    payload = dict(data)
    payload["iat"] = int(time.time())
    payload["exp"] = int(time.time()) + expires_seconds

    header_b64 = _base64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_b64 = _base64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")

    signature = hmac.new(SECRET_KEY.encode("utf-8"), signing_input, hashlib.sha256).digest()
    sig_b64 = _base64url_encode(signature)

    return f"{header_b64}.{payload_b64}.{sig_b64}"


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """Decodifica y valida la firma y expiración de un token JWT."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        header_b64, payload_b64, sig_b64 = parts

        signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
        expected_sig = hmac.new(SECRET_KEY.encode("utf-8"), signing_input, hashlib.sha256).digest()
        actual_sig = _base64url_decode(sig_b64)

        if not hmac.compare_digest(expected_sig, actual_sig):
            return None

        payload_bytes = _base64url_decode(payload_b64)
        payload = json.loads(payload_bytes.decode("utf-8"))

        # Verificar expiración
        exp = payload.get("exp")
        if exp and int(time.time()) > exp:
            return None

        return payload
    except Exception:
        return None


def generate_invitation_token() -> str:
    """Genera un token criptográfico seguro de alta entropía para invitaciones."""
    return secrets.token_urlsafe(32)


def render_invitation_email_html(
    nombre: str,
    email: str,
    rol: str,
    empresa_nombre: str,
    invitation_url: str
) -> str:
    """Genera la plantilla HTML corporativa de Apex Company para la invitación."""
    rol_label = "Administrador de Plataforma" if rol == "admin" else "Cliente / Anunciante DOOH"
    return f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <title>Invitación a la Plataforma Apex Company</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; background-color: #0b0f19; color: #f1f5f9; margin: 0; padding: 30px 15px; }}
    .container {{ max-width: 580px; margin: 0 auto; background: #111827; border: 1px solid #1f2937; border-radius: 12px; overflow: hidden; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }}
    .header {{ background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%); padding: 25px 30px; text-align: left; }}
    .logo {{ display: flex; align-items: center; gap: 10px; }}
    .logo-badge {{ width: 34px; height: 34px; background: #ffffff; color: #0284c7; border-radius: 8px; font-size: 18px; font-weight: 900; line-height: 34px; text-align: center; display: inline-block; }}
    .logo-text {{ font-size: 18px; font-weight: 800; color: #ffffff; letter-spacing: 0.5px; vertical-align: middle; }}
    .content {{ padding: 30px; }}
    h2 {{ color: #ffffff; font-size: 20px; margin-top: 0; font-weight: 700; }}
    p {{ color: #94a3b8; font-size: 14px; line-height: 1.6; margin: 12px 0; }}
    .card {{ background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 16px; margin: 20px 0; }}
    .card-item {{ display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 13px; }}
    .card-item:last-child {{ margin-bottom: 0; }}
    .card-label {{ color: #64748b; font-weight: 600; }}
    .card-val {{ color: #38bdf8; font-weight: 700; }}
    .btn-container {{ text-align: center; margin: 30px 0 20px 0; }}
    .btn {{ background: #0284c7; color: #ffffff !important; padding: 12px 28px; border-radius: 8px; text-decoration: none; font-weight: 700; font-size: 14px; display: inline-block; box-shadow: 0 4px 12px rgba(2,132,199,0.35); }}
    .btn:hover {{ background: #0369a1; }}
    .footer {{ background: #0f172a; padding: 18px 30px; text-align: center; font-size: 11px; color: #64748b; border-top: 1px solid #1f2937; }}
    .link-alt {{ word-break: break-all; color: #0284c7; font-size: 12px; font-family: monospace; }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div class="logo">
        <span class="logo-badge">A</span>
        <span class="logo-text">APEX COMPANY</span>
      </div>
    </div>
    <div class="content">
      <h2>¡Hola, {nombre}!</h2>
      <p>Has sido dado de alta e invitado para acceder al sistema unificado de <strong>Gestión y Analítica de Pantallas DOOH</strong> de <strong>Apex Company</strong>.</p>
      
      <div class="card">
        <div class="card-item">
          <span class="card-label">Correo Registrado:</span>
          <span class="card-val">{email}</span>
        </div>
        <div class="card-item">
          <span class="card-label">Empresa / Cliente:</span>
          <span class="card-val">{empresa_nombre}</span>
        </div>
        <div class="card-item">
          <span class="card-label">Rol de Acceso:</span>
          <span class="card-val">{rol_label}</span>
        </div>
      </div>

      <p>Para comenzar a utilizar tu cuenta, configurar tu contraseña y acceder a tus métricas en tiempo real, haz clic en el siguiente botón:</p>

      <div class="btn-container">
        <a href="{invitation_url}" class="btn">Activar mi Cuenta y Acceder</a>
      </div>

      <p style="font-size: 12px; color: #64748b; margin-top: 25px;">
        Si el botón no funciona, copia y pega este enlace directo en tu navegador:<br>
        <a href="{invitation_url}" class="link-alt">{invitation_url}</a>
      </p>
    </div>
    <div class="footer">
      Tecnología y Plataforma Desarrollada por <strong>apexcompany.com.mx</strong><br>
      Soluciones Avanzadas de Visión Artificial, Analítica Vehicular e Infraestructura Publicitaria DOOH.
    </div>
  </div>
</body>
</html>
"""


def send_invitation_email(
    to_email: str,
    subject: str,
    html_content: str,
    smtp_host: Optional[str] = None,
    smtp_port: int = 587,
    smtp_user: Optional[str] = None,
    smtp_password: Optional[str] = None
) -> Dict[str, Any]:
    """
    Envía el correo de invitación si hay credenciales SMTP configuradas,
    o registra la entrega simulada exitosa para desarrollo.
    """
    if smtp_host and smtp_user and smtp_password:
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = f"Apex Company <{smtp_user}>"
            msg["To"] = to_email
            msg.attach(MIMEText(html_content, "html", "utf-8"))

            with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
                server.starttls()
                server.login(smtp_user, smtp_password)
                server.sendmail(smtp_user, [to_email], msg.as_string())
            return {"sent": True, "method": "smtp", "detail": f"Correo enviado a {to_email}"}
        except Exception as e:
            return {"sent": False, "method": "smtp", "detail": f"Error SMTP: {str(e)}"}
    else:
        # Modo simulación / desarrollo local
        print(f"[AUTH_SERVICE] Simulación de correo de invitación a '{to_email}'.")
        return {
            "sent": True,
            "method": "mock_delivered",
            "detail": f"Invitación generada exitosamente para {to_email}. Enlace listo para ser utilizado."
        }
