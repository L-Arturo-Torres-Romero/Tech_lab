"""
Pruebas de los Endpoints de la API y Webhook.
"""

from fastapi.testclient import TestClient
from src.app import app
from src.config import VERIFY_TOKEN

client = TestClient(app)

def test_webhook_get_verificacion_exitosa():
    challenge = "1158201444"
    params = {
        "hub.mode": "subscribe",
        "hub.verify_token": VERIFY_TOKEN,
        "hub.challenge": challenge
    }
    response = client.get("/webhook", params=params)
    assert response.status_code == 200
    assert response.text == challenge

def test_webhook_get_verificacion_fallida():
    params = {
        "hub.mode": "subscribe",
        "hub.verify_token": "token_incorrecto_123",
        "hub.challenge": "1158201444"
    }
    response = client.get("/webhook", params=params)
    assert response.status_code == 403

def test_webhook_post_recepcion_mensaje():
    # Payload similar al emitido por Meta en los notebooks
    payload = {
        "object": "whatsapp_business_account",
        "entry": [{
            "id": "123456789",
            "changes": [{
                "value": {
                    "messaging_product": "whatsapp",
                    "metadata": {"display_phone_number": "15550269876", "phone_number_id": "12345"},
                    "messages": [{
                        "from": "5215512345678",
                        "id": "wamid.HBgLNTI1NTEyMzQ1Njc4FQIAEhgg",
                        "timestamp": "1728500000",
                        "text": {"body": "Hola, ¿qué sabores tienen?"},
                        "type": "text"
                    }]
                },
                "field": "messages"
            }]
        }]
    }
    response = client.post("/webhook", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_api_simulate_message():
    payload = {
        "phone_number": "525511223344",
        "message": "Hola, quiero saber precios de cupcakes"
    }
    response = client.post("/api/simulate-message", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "bot_response" in data
    assert data["phone_number"] == "525511223344"
    assert data["execution_time_ms"] >= 0

def test_api_get_catalog():
    response = client.get("/api/catalog")
    assert response.status_code == 200
    data = response.json()
    assert "sabores_masa" in data
    assert "paquetes_precios" in data
