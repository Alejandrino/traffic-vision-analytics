"""
check_deps.py — Verifica que todas las dependencias estén instaladas correctamente.
Ejecutar antes de main.py para diagnosticar problemas.
"""
import sys

checks = [
    ("cv2",          "opencv-python"),
    ("numpy",        "numpy"),
    ("ultralytics",  "ultralytics (YOLOv8)"),
    ("supervision",  "supervision (ByteTrack/LineZone)"),
    ("fastapi",      "fastapi"),
    ("uvicorn",      "uvicorn"),
    ("websockets",   "websockets"),
    ("PIL",          "Pillow"),
]

print(f"\nPython {sys.version}")
print("-" * 50)
ok = True
for module, pkg_name in checks:
    try:
        m = __import__(module)
        ver = getattr(m, "__version__", "?")
        print(f"  [OK] {pkg_name:<30} v{ver}")
    except ImportError as e:
        print(f"  [XX] {pkg_name:<30} NO INSTALADO -> pip install {pkg_name}")
        ok = False

print("-" * 50)
if ok:
    print("  ✓ Todas las dependencias están listas. Ejecuta: python main.py\n")
else:
    print("  ✗ Faltan dependencias. Instálalas y vuelve a ejecutar este check.\n")
    sys.exit(1)

# Verificar acceso a cámara 0 (USB)
print("Verificando cámara USB 0...")
try:
    import cv2
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        if ret:
            print(f"  ✓ Cámara USB 0 disponible ({frame.shape[1]}x{frame.shape[0]})")
        else:
            print("  ⚠ Cámara USB 0 abierta pero no devuelve frames")
    else:
        print("  ⚠ Cámara USB 0 no disponible (normal si usas solo RTSP)")
except Exception as e:
    print(f"  ⚠ Error al probar cámara: {e}")

print()
