"""
Módulo de Configuración.
Carga variables de entorno para WhatsApp Cloud API, servidor y proveedores LLM.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar archivo .env desde el directorio raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# Configuración de Meta WhatsApp Cloud API
WHATSAPP_ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN", "")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID", "")
WABA_TOKEN_ID = os.getenv("WABA_TOKEN_ID", "")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "cupcakes_webhook_secret_2026")
WHATSAPP_API_VERSION = os.getenv("WHATSAPP_API_VERSION", "v21.0")

# Servidor Web
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
DEBUG = os.getenv("DEBUG", "True").lower() in ("true", "1", "yes")

# Configuración de LLM
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto").lower()  # auto, ollama, groq, openai, fallback
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ngrok
NGROK_AUTHTOKEN = os.getenv("NGROK_AUTHTOKEN", "")

# Base de datos
DATABASE_PATH = os.getenv("DATABASE_PATH", str(BASE_DIR / "data" / "cupcakes.db"))

def is_whatsapp_configured() -> bool:
    """Valida si las credenciales mínimas de WhatsApp Cloud API están presentes."""
    return bool(WHATSAPP_ACCESS_TOKEN and PHONE_NUMBER_ID and WHATSAPP_ACCESS_TOKEN != "test_token_whatsapp")
