"""
Modelos de Datos para Pedidos de Cupcakes.
"""

from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field

class OrderStatus(str, Enum):
    COTIZACION = "COTIZACION"
    PENDIENTE_PAGO = "PENDIENTE_PAGO"
    CONFIRMADO = "CONFIRMADO"
    EN_PREPARACION = "EN_PREPARACION"
    LISTO = "LISTO"
    ENTREGADO = "ENTREGADO"
    CANCELADO = "CANCELADO"

class DeliveryType(str, Enum):
    RECOGER = "recoger"
    DOMICILIO = "domicilio"

class OrderCreate(BaseModel):
    phone_number: str = Field(..., description="Número de teléfono de WhatsApp del cliente")
    customer_name: Optional[str] = Field("Cliente", description="Nombre del cliente")
    quantity: int = Field(..., description="Cantidad de cupcakes (4, 6, 12, 24, etc.)")
    flavor: str = Field(..., description="Sabor de la masa (ej. Red Velvet, Chocolate Belga, Vainilla Francesa)")
    filling: Optional[str] = Field("Sin relleno", description="Relleno del cupcake (Nutella, Frutos Rojos, etc.)")
    frosting: Optional[str] = Field("Buttercream tradicional", description="Tipo de cobertura")
    theme_design: Optional[str] = Field("Clásico", description="Temática o diseño (Cumpleaños, Boda, etc.)")
    custom_text: Optional[str] = Field(None, description="Letrero o dedicatoria personalizada")
    delivery_date: str = Field(..., description="Fecha deseada de entrega (formato YYYY-MM-DD)")
    delivery_type: DeliveryType = Field(DeliveryType.RECOGER, description="Tipo de entrega")
    delivery_address: Optional[str] = Field(None, description="Dirección de envío si es a domicilio")
    notes: Optional[str] = Field(None, description="Notas adicionales o especificaciones de alérgenos")

class OrderUpdateStatus(BaseModel):
    status: OrderStatus = Field(..., description="Nuevo estado del pedido")
    notes: Optional[str] = Field(None, description="Nota sobre el cambio de estado")

class OrderResponse(BaseModel):
    id: str
    phone_number: str
    customer_name: Optional[str]
    quantity: int
    flavor: str
    filling: Optional[str]
    frosting: Optional[str]
    theme_design: Optional[str]
    custom_text: Optional[str]
    delivery_date: str
    delivery_type: str
    delivery_address: Optional[str]
    total_price: float
    status: str
    notes: Optional[str]
    created_at: str
    updated_at: str
