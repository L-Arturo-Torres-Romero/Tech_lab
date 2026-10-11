"""
Rutas para Simulación y Pruebas Locales del Asistente.
Permite interactuar con el bot y consultar bitácoras sin requerir un webhook activo de Meta.
"""

import time
from fastapi import APIRouter
from typing import Dict, Any, List
from src.models.whatsapp import SimulateMessageRequest, SimulateMessageResponse
from src.services.agent_service import (
    procesar_y_responder,
    conversaciones_por_numero,
    reiniciar_memoria_usuario,
    errores_procesamiento
)
from src.services.whatsapp_service import normalizar_numero_mx
from src.database import get_connection

router = APIRouter(prefix="/api", tags=["Simulator & Diagnostics"])

@router.post("/simulate-message", response_model=SimulateMessageResponse)
async def simulate_incoming_message(payload: SimulateMessageRequest):
    """
    Simula un mensaje entrante de WhatsApp y devuelve la respuesta directa del bot.
    Ideal para probar flujos conversacionales desde Swagger UI, Postman o el panel web.
    """
    inicio = time.time()
    phone_norm = normalizar_numero_mx(payload.phone_number)
    
    # Procesar directamente a través del agente
    bot_reply = procesar_y_responder(phone_norm, payload.message)
    tiempo_ms = round((time.time() - inicio) * 1000, 2)

    return SimulateMessageResponse(
        phone_number=phone_norm,
        user_message=payload.message,
        bot_response=bot_reply,
        execution_time_ms=tiempo_ms,
        model_used="agent_orchestrator"
    )

@router.get("/conversations/{phone_number}")
async def get_user_conversation(phone_number: str):
    """Devuelve la memoria conversacional activa en RAM para un número de teléfono."""
    phone_norm = normalizar_numero_mx(phone_number)
    historial = conversaciones_por_numero.get(phone_norm, [])
    return {
        "phone_number": phone_norm,
        "turns_count": len(historial),
        "history": historial
    }

@router.delete("/conversations/{phone_number}")
async def clear_user_conversation(phone_number: str):
    """Reinicia la memoria conversacional de un usuario."""
    phone_norm = normalizar_numero_mx(phone_number)
    reiniciar_memoria_usuario(phone_norm)
    return {"phone_number": phone_norm, "cleared": True}

@router.get("/logs")
async def get_system_logs(limit: int = 50):
    """Devuelve la bitácora de mensajes guardados en la base de datos."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM messages_log ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    
    return {
        "recent_messages": [dict(r) for r in rows],
        "recent_errors": errores_procesamiento[-20:]
    }
