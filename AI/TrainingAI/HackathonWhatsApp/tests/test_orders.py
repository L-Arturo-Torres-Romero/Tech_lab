"""
Pruebas del Servicio de Pedidos y Base de Datos.
"""

from datetime import datetime, timedelta
from src.services.order_service import (
    crear_pedido,
    consultar_pedido,
    consultar_disponibilidad,
    actualizar_estado_pedido,
    listar_pedidos
)

def test_consultar_pedido_semilla():
    # El pedido CUP-1001 se crea como semilla
    detalle = consultar_pedido("CUP-1001")
    assert "CUP-1001" in detalle
    assert "Red Velvet" in detalle

def test_crear_y_consultar_nuevo_pedido():
    fecha_futura = (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d")
    nuevo = crear_pedido(
        phone_number="525544332211",
        customer_name="Fernanda Luna",
        quantity=6,
        flavor="Chocolate Belga",
        filling="Nutella",
        frosting="Ganache de Chocolate",
        theme_design="Cumpleaños",
        delivery_date=fecha_futura,
        delivery_type="recoger"
    )
    
    order_id = nuevo["id"]
    assert order_id.startswith("CUP-")
    assert nuevo["total_price"] == 230.0

    # Consultar detalle
    consulta = consultar_pedido(order_id)
    assert order_id in consulta
    assert "Chocolate Belga" in consulta

def test_consultar_disponibilidad():
    # Fecha de hoy debe avisar de la anticipación mínima
    hoy = datetime.now().strftime("%Y-%m-%d")
    resp_hoy = consultar_disponibilidad(hoy)
    assert "24 horas" in resp_hoy

    # Fecha futura debe indicar disponibilidad
    fecha_futura = (datetime.now() + timedelta(days=4)).strftime("%Y-%m-%d")
    resp_futura = consultar_disponibilidad(fecha_futura)
    assert "cupo disponible" in resp_futura.lower()

def test_actualizar_estado_pedido():
    # Actualizar estado de CUP-1002
    ok = actualizar_estado_pedido("CUP-1002", "LISTO", "Empacado en caja de lujo")
    assert ok is True

    detalle = consultar_pedido("CUP-1002")
    assert "Listo para entrega" in detalle or "LISTO" in detalle

def test_listar_pedidos():
    pedidos = listar_pedidos(limit=10)
    assert len(pedidos) >= 2
    assert all("id" in p for p in pedidos)
