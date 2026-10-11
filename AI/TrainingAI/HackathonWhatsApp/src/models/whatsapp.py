"""
Modelos de Datos para WhatsApp Cloud API.
Estructuras para deserializar webhooks entrantes y serializar mensajes salientes.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class WhatsAppTextMessage(BaseModel):
    body: str

class WhatsAppIncomingMessage(BaseModel):
    from_user: str = Field(..., alias="from")
    id: str
    timestamp: str
    type: str = "text"
    text: Optional[WhatsAppTextMessage] = None

class WhatsAppValue(BaseModel):
    messaging_product: Optional[str] = "whatsapp"
    metadata: Optional[Dict[str, Any]] = None
    contacts: Optional[List[Dict[str, Any]]] = None
    messages: Optional[List[Dict[str, Any]]] = None
    statuses: Optional[List[Dict[str, Any]]] = None

class WhatsAppChange(BaseModel):
    value: WhatsAppValue
    field: str

class WhatsAppEntry(BaseModel):
    id: str
    changes: List[WhatsAppChange]

class WhatsAppWebhookPayload(BaseModel):
    object: Optional[str] = "whatsapp_business_account"
    entry: List[WhatsAppEntry]

class SimulateMessageRequest(BaseModel):
    phone_number: str = Field(..., description="Número de teléfono simulado del usuario (ej. 525512345678)")
    message: str = Field(..., description="Texto del mensaje a simular")
    customer_name: Optional[str] = Field("Cliente de Prueba", description="Nombre opcional del cliente")

class SimulateMessageResponse(BaseModel):
    phone_number: str
    user_message: str
    bot_response: str
    execution_time_ms: float
    model_used: str
