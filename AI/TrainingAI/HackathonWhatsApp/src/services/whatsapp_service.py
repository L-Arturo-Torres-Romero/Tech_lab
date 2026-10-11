"""
Servicio de Integración con Meta WhatsApp Cloud API.
Se encarga de la normalización de números telefónicos y del envío de mensajes de texto a la API oficial de Meta.
"""

import re
import requests
from typing import Dict, Any, List
from src.config import (
    WHATSAPP_ACCESS_TOKEN,
    PHONE_NUMBER_ID,
    WHATSAPP_API_VERSION,
    is_whatsapp_configured
)
from src.database import log_message

respuestas_enviadas: List[Dict[str, Any]] = []

def normalizar_numero_mx(numero: str) -> str:
    """
    Normaliza el número telefónico para WhatsApp.
    En México, Meta envía '521' en el webhook (13 dígitos), pero para enviar mensajes
    requiere el formato estándar internacional sin el 1 intermedio ('52' + 10 dígitos).
    """
    if not numero:
        return ""
    # Quitar signos, espacios y guiones
    limpio = re.sub(r"[^\d]", "", str(numero))
    
    # Manejar caso México: 521XXXXXXXXXX (13 dígitos) -> 52XXXXXXXXXX (12 dígitos)
    if limpio.startswith("521") and len(limpio) == 13:
        return "52" + limpio[3:]
        
    # Si viene con 10 dígitos nacionales mexicanos, agregar prefijo 52
    if len(limpio) == 10:
        return "52" + limpio
        
    return limpio

def enviar_respuesta_whatsapp(numero: str, texto: str) -> Dict[str, Any]:
    """
    Envía un mensaje de texto al usuario a través de WhatsApp Cloud API.
    Si las credenciales no están configuradas (modo local/simulación), registra el mensaje sin error.
    """
    num_norm = normalizar_numero_mx(numero)
    registro = {
        "numero": num_norm,
        "texto": texto,
        "status_code": 200,
        "simulado": False
    }

    if not is_whatsapp_configured():
        # Modo simulación local o pruebas
        registro["simulado"] = True
        registro["response"] = "Enviado a simulador (credenciales Meta no configuradas)"
        respuestas_enviadas.append(registro)
        log_message(phone_number=num_norm, direction="OUT", message_text=texto, status="simulado")
        return registro

    url = f"https://graph.facebook.com/{WHATSAPP_API_VERSION}/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": num_norm,
        "type": "text",
        "text": {"body": texto}
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        registro["status_code"] = response.status_code
        registro["response"] = response.text
        respuestas_enviadas.append(registro)

        log_status = "ok" if response.status_code in (200, 201) else f"error_{response.status_code}"
        log_message(phone_number=num_norm, direction="OUT", message_text=texto, status=log_status)
        return registro

    except requests.exceptions.RequestException as e:
        registro["status_code"] = 500
        registro["error"] = str(e)
        respuestas_enviadas.append(registro)
        log_message(phone_number=num_norm, direction="OUT", message_text=texto, status="request_failed")
        return registro
