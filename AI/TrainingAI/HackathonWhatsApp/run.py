"""
Punto de Entrada Principal del Servidor.
Permite iniciar el servidor FastAPI con Uvicorn y opcionalmente abrir un túnel público con ngrok.

Uso:
    python run.py
    python run.py --port 8080
    python run.py --ngrok
"""

import sys
import argparse
import uvicorn

# Asegurar compatibilidad de emojis en consolas Windows (cp1252 -> utf-8)
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from src.config import HOST, PORT, NGROK_AUTHTOKEN

def start_ngrok(port: int):
    """Inicia el túnel de ngrok si está disponible y configurado."""
    try:
        from pyngrok import ngrok
        if NGROK_AUTHTOKEN:
            ngrok.set_auth_token(NGROK_AUTHTOKEN)
        tunnel = ngrok.connect(port)
        public_url = tunnel.public_url
        print("\n" + "=" * 65)
        print(" 🌐 TÚNEL NGROK PÚBLICO ACTIVADO CON ÉXITO")
        print("=" * 65)
        print(f" URL pública:        {public_url}")
        print(f" Callback URL Meta:  {public_url}/webhook")
        print(" Configura esta Callback URL en:")
        print(" Meta for Developers > WhatsApp > Configuración de la API > Webhooks")
        print("=" * 65 + "\n")
        return tunnel
    except Exception as e:
        print(f"\n⚠️ No se pudo iniciar el túnel de ngrok: {e}")
        print("El servidor continuará corriendo en modo local.\n")
        return None

def main():
    parser = argparse.ArgumentParser(description="Servidor Backend Cupcake WhatsApp Bot")
    parser.add_argument("--host", default=HOST, help="Host del servidor (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=PORT, help="Puerto del servidor (default: 8000)")
    parser.add_argument("--ngrok", action="store_true", help="Abrir túnel público con pyngrok")
    parser.add_argument("--reload", action="store_true", help="Habilitar autoreload para desarrollo")
    args = parser.parse_args()

    tunnel = None
    if args.ngrok:
        tunnel = start_ngrok(args.port)

    print("\nIniciando servidor Uvicorn...")
    try:
        uvicorn.run(
            "src.app:app",
            host=args.host,
            port=args.port,
            reload=args.reload
        )
    finally:
        if tunnel:
            from pyngrok import ngrok
            print("Cerrando túnel de ngrok...")
            ngrok.kill()

if __name__ == "__main__":
    main()
