"""
Pruebas de normalización de números telefónicos para WhatsApp en México.
"""

from src.services.whatsapp_service import normalizar_numero_mx

def test_normalizar_numero_mx_con_521():
    # En México, Meta envía '521' en el webhook (13 dígitos)
    numero_inbound = "5215512345678"
    assert normalizar_numero_mx(numero_inbound) == "525512345678"

def test_normalizar_numero_mx_con_10_digitos():
    # 10 dígitos nacionales mexicanos
    numero_10 = "5512345678"
    assert normalizar_numero_mx(numero_10) == "525512345678"

def test_normalizar_numero_con_formato_mas_y_espacios():
    # Números con signos, espacios o guiones
    numero_formateado = "+52 1 (55) 1234-5678"
    assert normalizar_numero_mx(numero_formateado) == "525512345678"

def test_normalizar_numero_ya_estandar():
    numero_ok = "525598765432"
    assert normalizar_numero_mx(numero_ok) == "525598765432"
