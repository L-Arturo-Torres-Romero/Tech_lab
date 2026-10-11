"""
Pruebas del Agente Conversacional y Guardrails.
"""

from src.services.agent_service import (
    procesar_y_responder,
    reiniciar_memoria_usuario
)

TEST_PHONE = "525599887766"

def setup_function():
    reiniciar_memoria_usuario(TEST_PHONE)

def test_guardrail_saludo_inicial():
    resp = procesar_y_responder(TEST_PHONE, "Hola, buen día")
    assert "Cupcake Wonderland" in resp
    assert "menú" in resp.lower() or "catálogo" in resp.lower()

def test_guardrail_urgencia_alergia():
    resp = procesar_y_responder(TEST_PHONE, "Tengo una alergia severa y me intoxiqué con un postre")
    assert "ATENCIÓN DE SEGURIDAD" in resp or "emergencia" in resp.lower()
    assert "lácteos" in resp.lower() or "alérgica" in resp.lower()

def test_guardrail_escalamiento_humano():
    resp = procesar_y_responder(TEST_PHONE, "Quiero hablar con una persona del equipo")
    assert "ASESOR HUMANO" in resp or "reposteros" in resp.lower()

def test_flujo_consultar_pedido():
    # Pedir estatus de CUP-1001
    resp = procesar_y_responder(TEST_PHONE, "Hola, quiero saber el estatus de mi pedido CUP-1001")
    assert "CUP-1001" in resp
    assert "Red Velvet" in resp

def test_flujo_consultar_menu():
    resp = procesar_y_responder(TEST_PHONE, "¿Cuáles son los sabores de pan que tienen?")
    assert "Vainilla Francesa" in resp or "Chocolate Belga" in resp
