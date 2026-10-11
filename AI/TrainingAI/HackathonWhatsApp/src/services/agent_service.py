"""
Servicio del Agente Conversacional para WhatsApp.
Implementa el orquestador con guardrails deterministas (urgencia/alergias, escalamiento, saludos),
memoria conversacional por número y serialización por locks de hilo.
"""

import threading
from typing import Dict, List, Any
from src.services.whatsapp_service import enviar_respuesta_whatsapp
from src.services.llm_service import generar_respuesta_llm
from src.database import log_message

# Reglas fijas deterministas (Guardrails)
FRASES_URGENCIA = [
    "alergia", "anafilaxia", "intoxicacion", "intoxicación", "urgencia",
    "emergencia", "grave", "hospital", "asfixia", "fraude", "cargo no reconocido",
    "cobro duplicado", "me cobraron dos veces"
]

FRASES_ESCALAMIENTO = [
    "hablar con alguien", "hablar con una persona", "agente humano",
    "asistente humano", "humano", "asesor", "persona real", "operador",
    "atencion a clientes", "atención a clientes", "quiero hablar con un humano"
]

FRASES_SALUDO = [
    "hola", "buenas", "buenos dias", "buenos días", "buenas tardes",
    "buenas noches", "que tal", "qué tal", "buen dia", "buen día", "hey", "saludos"
]

# Estructuras en memoria para serialización y conversación
locks_por_numero: Dict[str, threading.Lock] = {}
conversaciones_por_numero: Dict[str, List[Dict[str, str]]] = {}
usuarios_saludados = set()
errores_procesamiento: List[Dict[str, Any]] = []

def obtener_lock(numero: str) -> threading.Lock:
    """Devuelve un candado exclusivo por número de teléfono para evitar condiciones de carrera."""
    if numero not in locks_por_numero:
        locks_por_numero[numero] = threading.Lock()
    return locks_por_numero[numero]

def es_solo_saludo(texto_low: str) -> bool:
    """Comprueba si el mensaje del usuario consiste únicamente en un saludo."""
    texto_limpio = texto_low.strip(" ,.!¡¿?")
    return texto_limpio in FRASES_SALUDO

def enviar_aviso_de_error(numero: str) -> None:
    """Envía un aviso de falla técnica amistoso para que el bot nunca deje al usuario sin respuesta."""
    try:
        enviar_respuesta_whatsapp(
            numero,
            "🧁 Estamos teniendo un pequeño inconveniente técnico con nuestro sistema. "
            "Por favor danos un momento o intenta nuevamente en unos minutos."
        )
    except Exception:
        pass

def procesar_y_responder(numero: str, texto: str) -> str:
    """
    Función central de procesamiento y respuesta:
    1. Asegura el Lock del número telefónico.
    2. Aplica reglas deterministas de seguridad (alergias/urgencias).
    3. Aplica regla de escalamiento a personal humano.
    4. Aplica regla de saludo inicial.
    5. O delega a la inteligencia artificial (LLM/NLU) con memoria y tools.
    """
    with obtener_lock(numero):
        try:
            texto_low = texto.lower()
            log_message(phone_number=numero, direction="IN", message_text=texto)

            # 1. REGLA DE URGENCIA / ALERGIAS ALIMENTARIAS
            if any(frase in texto_low for frase in FRASES_URGENCIA):
                respuesta = (
                    "⚠️ *ATENCIÓN DE SEGURIDAD / URGENCIA:*\n\n"
                    "Si presentas una reacción alérgica o emergencia médica, comunícate inmediatamente a los números de emergencia (911).\n\n"
                    "Nuestros cupcakes regulares se elaboran en instalaciones donde se manejan lácteos, trigo (gluten), huevos y frutos secos. "
                    "Para reportes de pagos o atención prioritaria, llama directamente a nuestra línea directa: *(55) 5555-CUPCAKE*."
                )
                enviar_respuesta_whatsapp(numero, respuesta)
                return respuesta

            # 2. REGLA DE ESCALAMIENTO HUMANO
            if any(frase in texto_low for frase in FRASES_ESCALAMIENTO):
                respuesta = (
                    "👩‍🍳 *TRANSFERENCIA CON UN ASESOR HUMANO:*\n\n"
                    "Con mucho gusto. He canalizado tu conversación con nuestro equipo de reposteros en tienda. "
                    "En unos instantes un miembro del equipo te responderá personalmente por este mismo chat. ¡Gracias por tu paciencia!"
                )
                enviar_respuesta_whatsapp(numero, respuesta)
                return respuesta

            # 3. REGLA DE SALUDO INICIAL
            if es_solo_saludo(texto_low) or (numero not in usuarios_saludados and len(texto.strip()) <= 15):
                usuarios_saludados.add(numero)
                respuesta = (
                    "🧁 ¡Hola! Bienvenido a *Cupcake Wonderland*, tu pastelería artesanal de cupcakes personalizados. ✨\n\n"
                    "¿En qué podemos endulzarte el día hoy?\n"
                    "• Escribe *'menú'* para ver nuestros sabores y precios.\n"
                    "• Pregúntame por *'disponibilidad'* para una fecha.\n"
                    "• O cuéntame qué cupcakes te gustaría cotizar o personalizar."
                )
                enviar_respuesta_whatsapp(numero, respuesta)
                return respuesta

            # 4. PROCESAMIENTO CON LLM / AGENTE Y MEMORIA CONVERSACIONAL
            historial = conversaciones_por_numero.get(numero, [])

            respuesta_bot, modelo_usado = generar_respuesta_llm(numero, texto, historial)

            # Actualizar memoria del usuario (máximo 8 turnos recientes)
            historial.append({"role": "user", "content": texto})
            historial.append({"role": "assistant", "content": respuesta_bot})
            if len(historial) > 16:
                historial = historial[-16:]
            conversaciones_por_numero[numero] = historial

            # Enviar mensaje a WhatsApp
            enviar_respuesta_whatsapp(numero, respuesta_bot)
            return respuesta_bot

        except Exception as e:
            errores_procesamiento.append({
                "numero": numero,
                "texto": texto,
                "error": repr(e)
            })
            enviar_aviso_de_error(numero)
            return "Lo sentimos, ocurrió un error técnico al procesar tu solicitud."

def reiniciar_memoria_usuario(numero: str) -> None:
    """Limpia el historial conversacional de un usuario (para pruebas o reinicios)."""
    if numero in conversaciones_por_numero:
        conversaciones_por_numero[numero] = []
    if numero in usuarios_saludados:
        usuarios_saludados.remove(numero)
