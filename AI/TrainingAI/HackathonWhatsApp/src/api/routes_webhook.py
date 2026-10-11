"""
Rutas del Webhook de Meta WhatsApp Cloud API.
Incluye verificación de webhook (GET) y procesamiento asíncrono con BackgroundTasks (POST).
"""

from fastapi import APIRouter, Request, BackgroundTasks, Response, status
from fastapi.responses import PlainTextResponse, JSONResponse
from typing import Set, Dict, Any, List
from src.config import VERIFY_TOKEN, is_whatsapp_configured
from src.services.whatsapp_service import normalizar_numero_mx
from src.services.agent_service import procesar_y_responder, errores_procesamiento

router = APIRouter(tags=["WhatsApp Webhook"])

# Memoria de control para evitar duplicados cuando Meta reintenta entregas
ids_procesados: Set[str] = set()
mensajes_recibidos: List[Dict[str, Any]] = []

@router.get("/webhook")
async def verificar_webhook(request: Request):
    """
    Endpoint de Verificación de Webhook para Meta for Developers.
    Meta realiza una petición GET con hub.mode, hub.verify_token y hub.challenge.
    """
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        # Meta requiere que se devuelva el challenge tal cual
        return PlainTextResponse(content=challenge, status_code=200)

    return JSONResponse(content={"error": "Token de verificación inválido"}, status_code=403)

@router.post("/webhook")
async def recibir_mensaje(request: Request, background_tasks: BackgroundTasks):
    """
    Endpoint de Recepción de Mensajes de WhatsApp.
    Recibe los payloads enviados por Meta, valida deduplicación y delega
    el procesamiento a BackgroundTasks para responder con status 200 de inmediato.
    """
    try:
        body = await request.json()
        mensajes_recibidos.append(body)

        # Meta envía eventos estructurados en entry[0].changes[0].value
        entries = body.get("entry", [])
        for entry in entries:
            for change in entry.get("changes", []):
                value = change.get("value", {})
                messages = value.get("messages", [])
                
                # Si el payload es una actualización de estado (sent, delivered, read), no contiene 'messages'
                if not messages:
                    continue

                for msg in messages:
                    wamid = msg.get("id")
                    
                    # Deduplicación: si ya procesamos este id, responder OK y no duplicar
                    if wamid in ids_procesados:
                        return {"status": "ok", "message": "already_processed"}
                        
                    ids_procesados.add(wamid)
                    # Mantener un límite razonable en la memoria de IDs
                    if len(ids_procesados) > 5000:
                        ids_procesados.pop()

                    # Solo procesar mensajes de texto
                    msg_type = msg.get("type")
                    if msg_type == "text":
                        from_number = normalizar_numero_mx(msg.get("from", ""))
                        texto = msg.get("text", {}).get("body", "")
                        
                        if from_number and texto:
                            # Delegar procesamiento asíncrono para liberar el hilo de Meta en < 100ms
                            background_tasks.add_task(procesar_y_responder, from_number, texto)

        return {"status": "ok"}

    except Exception as e:
        errores_procesamiento.append({"error_webhook": repr(e)})
        return {"status": "error", "detail": str(e)}

@router.get("/health")
async def health_check():
    """Chequeo de salud del servicio y diagnóstico de credenciales."""
    return {
        "status": "healthy",
        "whatsapp_configured": is_whatsapp_configured(),
        "messages_received_count": len(mensajes_recibidos),
        "errors_count": len(errores_procesamiento)
    }
