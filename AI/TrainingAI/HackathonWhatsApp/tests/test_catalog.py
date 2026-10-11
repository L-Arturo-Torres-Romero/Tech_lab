"""
Pruebas para el Catálogo de Cupcakes y Preguntas Frecuentes.
"""

from src.services.catalog_service import (
    consultar_catalogo,
    buscar_pregunta_frecuente,
    calcular_precio_pedido
)

def test_consultar_catalogo_completo():
    catalogo = consultar_catalogo()
    assert "Red Velvet" in catalogo
    assert "Chocolate Belga" in catalogo
    assert "Paquetes" in catalogo

def test_consultar_catalogo_por_categoria():
    sabores = consultar_catalogo("sabores")
    assert "Vainilla Francesa" in sabores
    assert "Limón & Amapola" in sabores

    rellenos = consultar_catalogo("rellenos")
    assert "Nutella" in rellenos
    assert "Frutos Rojos" in rellenos

    paquetes = consultar_catalogo("paquetes")
    assert "$160" in paquetes or "$420" in paquetes

def test_buscar_pregunta_frecuente_anticipacion():
    faq = buscar_pregunta_frecuente("¿Con cuánto tiempo de anticipación debo pedir?")
    assert "24 a 48 horas" in faq

def test_buscar_pregunta_frecuente_pagos():
    faq = buscar_pregunta_frecuente("¿Qué métodos de pago tienen?")
    assert "transferencia" in faq.lower()
    assert "anticipo" in faq.lower()

def test_buscar_pregunta_frecuente_alergias():
    faq = buscar_pregunta_frecuente("Tengo alergia al gluten")
    assert "trigo" in faq.lower() or "gluten" in faq.lower()

def test_calcular_precio_pedido():
    # 6 cupcakes base
    precio_6 = calcular_precio_pedido(cantidad=6)
    assert precio_6 == 230.0

    # 12 cupcakes base
    precio_12 = calcular_precio_pedido(cantidad=12)
    assert precio_12 == 420.0

    # 12 cupcakes con foto comestible y envío a domicilio
    precio_12_completo = calcular_precio_pedido(cantidad=12, con_foto=True, domicilio=True)
    # 420 + 40 (foto) + 60 (envío) = 520
    assert precio_12_completo == 520.0
