"""
Aplicación Principal FastAPI.
Configura middlewares, routers, archivos estáticos y eventos de inicio.
"""

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.api.routes_webhook import router as webhook_router
from src.api.routes_orders import router as orders_router
from src.api.routes_simulator import router as simulator_router
from src.database import init_db
from src.config import DEBUG

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicializa la base de datos y recursos al arrancar el servidor."""
    init_db()
    print("=====================================================")
    print("🧁 Cupcake Wonderland - WhatsApp Bot Backend Iniciado")
    print("📍 Documentación Swagger: http://localhost:8000/docs")
    print("📱 Panel Web y Simulador: http://localhost:8000/")
    print("🔗 Webhook de WhatsApp:  http://localhost:8000/webhook")
    print("=====================================================")
    yield

app = FastAPI(
    title="Cupcake Wonderland - WhatsApp Bot & Order Management",
    description="Sistema inteligente de gestión de pedidos de cupcakes personalizados a través de WhatsApp Cloud API.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configuración de CORS para pruebas locales y dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir Routers
app.include_router(webhook_router)
app.include_router(orders_router)
app.include_router(simulator_router)

# Servir archivos estáticos del dashboard
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_dashboard():
        """Sirve el panel web principal de la pastelería."""
        return FileResponse(str(STATIC_DIR / "index.html"))
