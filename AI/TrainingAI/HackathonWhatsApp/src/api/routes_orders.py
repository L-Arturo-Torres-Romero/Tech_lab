"""
Rutas REST para Gestión de Pedidos y Catálogo.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from src.models.order import OrderCreate, OrderUpdateStatus, OrderResponse
from src.services.order_service import (
    crear_pedido,
    listar_pedidos,
    consultar_pedido,
    actualizar_estado_pedido
)
from src.services.catalog_service import (
    SABORES_MASA,
    RELLENOS,
    COBERTURAS,
    PAQUETES_PRECIOS,
    TEMATICAS_PERSONALIZACION,
    PREGUNTAS_FRECUENTES_CUPCAKES
)

router = APIRouter(prefix="/api", tags=["Orders & Catalog"])

@router.get("/orders", response_model=List[OrderResponse])
async def get_orders(
    limit: int = Query(50, ge=1, le=100),
    status: Optional[str] = Query(None, description="Filtrar por estado del pedido")
):
    """Lista todos los pedidos de cupcakes registrados en la pastelería."""
    return listar_pedidos(limit=limit, status=status)

@router.post("/orders", response_model=OrderResponse)
async def create_new_order(order_data: OrderCreate):
    """Crea manualmente un nuevo pedido en el sistema."""
    pedido = crear_pedido(
        phone_number=order_data.phone_number,
        customer_name=order_data.customer_name,
        quantity=order_data.quantity,
        flavor=order_data.flavor,
        filling=order_data.filling,
        frosting=order_data.frosting,
        theme_design=order_data.theme_design,
        custom_text=order_data.custom_text,
        delivery_date=order_data.delivery_date,
        delivery_type=order_data.delivery_type.value,
        delivery_address=order_data.delivery_address,
        notes=order_data.notes
    )
    # Recuperar el pedido completo
    pedidos = listar_pedidos(limit=1)
    if pedidos:
        return pedidos[0]
    return pedido

@router.get("/orders/{order_id}")
async def get_order_detail(order_id: str):
    """Obtiene el detalle formateado o registro de un pedido específico."""
    info_txt = consultar_pedido(order_id)
    return {"order_id": order_id, "summary": info_txt}

@router.patch("/orders/{order_id}/status")
async def update_order_status_endpoint(order_id: str, update: OrderUpdateStatus):
    """Actualiza el estado de un pedido (ej. EN_PREPARACION, LISTO, ENTREGADO)."""
    success = actualizar_estado_pedido(order_id, update.status.value, update.notes)
    if not success:
        raise HTTPException(status_code=404, detail=f"Pedido {order_id} no encontrado")
    return {"order_id": order_id, "new_status": update.status.value, "updated": True}

@router.get("/catalog")
async def get_catalog_data():
    """Retorna los datos completos del catálogo en formato estructurado JSON."""
    return {
        "sabores_masa": SABORES_MASA,
        "rellenos": RELLENOS,
        "coberturas": COBERTURAS,
        "paquetes_precios": PAQUETES_PRECIOS,
        "tematicas": TEMATICAS_PERSONALIZACION,
        "faq": PREGUNTAS_FRECUENTES_CUPCAKES
    }
