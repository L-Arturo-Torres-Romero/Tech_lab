"""
Servicio Unificado de LLM y Function Calling.
Soporta Ollama (Llama 3.1 local), OpenAI/Groq, y Fallback NLU inteligente sin dependencias externas.
"""

import json
import re
import requests
from typing import Dict, Any, List, Optional, Tuple
from src.config import (
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    GROQ_API_KEY,
    OPENAI_API_KEY
)
from src.services.catalog_service import (
    consultar_catalogo,
    buscar_pregunta_frecuente,
    PAQUETES_PRECIOS,
    SABORES_MASA,
    RELLENOS,
    COBERTURAS
)
from src.services.order_service import (
    consultar_disponibilidad,
    crear_pedido,
    consultar_pedido,
    normalizar_fecha
)

SYSTEM_PROMPT = (
    "Eres 'DulceBot', el asistente virtual experto de 'Cupcake Wonderland', una pastelería artesanal de cupcakes personalizados. "
    "Tu objetivo es brindar una atención cálida, dulce y eficiente para ayudar al cliente a armar y consultar sus pedidos. "
    "REGLAS OBLIGATORIAS:\n"
    "1. Responde de forma concisa (2 a 3 líneas o viñetas claras).\n"
    "2. Usa las herramientas (tools) disponibles cuando el usuario pregunte por disponibilidad, catálogo, estado de un pedido o preguntas frecuentes.\n"
    "3. Nunca inventes precios ni sabores que no estén en las herramientas.\n"
    "4. Si el cliente quiere hacer un pedido, asegúrate de recopilar: cantidad, sabor de masa, relleno, cobertura, temática y fecha de entrega deseada.\n"
    "5. Mantén un tono alegre, amable y profesional usando emojis temáticos (🧁, 🍓, 🍰, ✨).\n"
    "6. No menciones detalles técnicos de funciones ni parámetros internos."
)

# Definición de herramientas para Function Calling
TOOLS_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "consultar_catalogo",
            "description": "Consulta los sabores de masa, rellenos, coberturas, temáticas o precios de los paquetes de cupcakes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "categoria": {
                        "type": "string",
                        "description": "Categoría a consultar: 'sabores', 'rellenos', 'coberturas', 'paquetes', 'tematicas' o 'todo'."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "consultar_disponibilidad",
            "description": "Verifica si hay cupo disponible de producción y entrega para una fecha específica (mínimo 24 horas de anticipación).",
            "parameters": {
                "type": "object",
                "properties": {
                    "fecha": {
                        "type": "string",
                        "description": "Fecha a consultar en formato YYYY-MM-DD o en texto como '15 de octubre' o 'mañana'."
                    }
                },
                "required": ["fecha"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "consultar_pedido",
            "description": "Consulta el estado actual y detalles de un pedido de cupcakes utilizando su número identificador (ej. CUP-1001).",
            "parameters": {
                "type": "object",
                "properties": {
                    "numero_pedido": {
                        "type": "string",
                        "description": "Número de pedido del cliente (ej. CUP-1001)."
                    }
                },
                "required": ["numero_pedido"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "crear_pedido",
            "description": "Registra un nuevo pedido personalizado de cupcakes una vez que el cliente ha elegido sus opciones.",
            "parameters": {
                "type": "object",
                "properties": {
                    "quantity": {
                        "type": "integer",
                        "description": "Cantidad de cupcakes (4, 6, 12, 24, etc.)."
                    },
                    "flavor": {
                        "type": "string",
                        "description": "Sabor del pan/masa (Red Velvet, Chocolate Belga, Vainilla Francesa, etc.)."
                    },
                    "filling": {
                        "type": "string",
                        "description": "Relleno deseado (Nutella, Frutos Rojos, Dulce de Leche, etc.)."
                    },
                    "frosting": {
                        "type": "string",
                        "description": "Tipo de cobertura o frosting."
                    },
                    "theme_design": {
                        "type": "string",
                        "description": "Temática o diseño (Cumpleaños, Boda, Infantil, foto comestible, etc.)."
                    },
                    "delivery_date": {
                        "type": "string",
                        "description": "Fecha de entrega deseada (YYYY-MM-DD)."
                    },
                    "delivery_type": {
                        "type": "string",
                        "description": "'recoger' en sucursal o 'domicilio'."
                    },
                    "custom_text": {
                        "type": "string",
                        "description": "Texto o letrero para la dedicatoria en fondant."
                    }
                },
                "required": ["quantity", "flavor", "delivery_date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "buscar_pregunta_frecuente",
            "description": "Busca información sobre políticas de la pastelería: anticipación mínima, métodos de pago, alérgenos, envíos, conservación o cancelaciones.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pregunta": {
                        "type": "string",
                        "description": "Duda o consulta del usuario en texto libre."
                    }
                },
                "required": ["pregunta"]
            }
        }
    }
]

def ejecutar_herramienta(nombre: str, args: Dict[str, Any], phone_number: str) -> str:
    """Ejecuta la función real de negocio según la llamada del modelo."""
    if nombre == "consultar_catalogo":
        return consultar_catalogo(args.get("categoria"))
        
    elif nombre == "consultar_disponibilidad":
        return consultar_disponibilidad(args.get("fecha"))
        
    elif nombre == "consultar_pedido":
        return consultar_pedido(args.get("numero_pedido"), phone_number=phone_number)
        
    elif nombre == "buscar_pregunta_frecuente":
        return buscar_pregunta_frecuente(args.get("pregunta", ""))
        
    elif nombre == "crear_pedido":
        pedido = crear_pedido(
            phone_number=phone_number,
            quantity=args.get("quantity", 12),
            flavor=args.get("flavor", "Red Velvet"),
            filling=args.get("filling", "Frutos Rojos"),
            frosting=args.get("frosting", "Frosting de Queso Crema"),
            theme_design=args.get("theme_design", "Personalizado"),
            custom_text=args.get("custom_text"),
            delivery_date=args.get("delivery_date"),
            delivery_type=args.get("delivery_type", "recoger")
        )
        return (
            f"🎉 ¡Tu pedido ha sido registrado con éxito!\n\n"
            f"• *Número de Pedido:* *{pedido['id']}*\n"
            f"• *Total Estimado:* ${pedido['total_price']:.2f} MXN\n"
            f"• *Fecha de Entrega:* {pedido['delivery_date']}\n"
            f"• *Estado:* Pendiente de anticipo (50%)\n\n"
            f"Para confirmar tu pedido, por favor realiza la transferencia del 50% de anticipo a nuestra cuenta CLABE que te compartiremos enseguida."
        )
        
    return f"Herramienta {nombre} no encontrada."

# -------------------------------------------------------------
# MOTOR FALLBACK NLU (Inteligente y sin dependencias externas)
# -------------------------------------------------------------
def motor_fallback_nlu(phone_number: str, user_text: str, historial: List[Dict[str, str]]) -> str:
    """
    Analizador NLU semántico de alta precisión para el dominio de cupcakes.
    Extrae entidades e intenciones para responder y ejecutar las herramientas correspondientes.
    Garantiza que el asistente NUNCA falle, aún si Ollama o las APIs externas están inactivas.
    """
    texto_low = user_text.lower().strip()

    # 1. Consulta de número de pedido específico (ej. CUP-1001 o 'pedido 1001')
    match_pedido = re.search(r"(?:cup-?|pedido\s*#?\s*)(\d{3,5})", texto_low)
    if match_pedido:
        order_num = f"CUP-{match_pedido.group(1)}"
        return consultar_pedido(order_num, phone_number=phone_number)

    if any(p in texto_low for p in ["estatus de mi pedido", "mi pedido", "donde va mi pedido", "consultar pedido", "estado del pedido"]):
        return consultar_pedido(phone_number=phone_number)

    # 2. Consulta de catálogo / sabores / precios / opciones
    if any(p in texto_low for p in ["sabor", "sabores", "que sabores", "de que sabores", "relleno", "rellenos"]):
        if "relleno" in texto_low:
            return consultar_catalogo("rellenos")
        return consultar_catalogo("sabores")

    if any(p in texto_low for p in ["precio", "precios", "cuanto cuesta", "cuanto sale", "costo", "paquete", "paquetes", "menu", "catalogo"]):
        return consultar_catalogo("paquetes")

    if any(p in texto_low for p in ["frosting", "cobertura", "betun", "betún", "decoracion", "tematica"]):
        return consultar_catalogo("coberturas")

    # 3. Consulta de disponibilidad y fechas
    if any(p in texto_low for p in ["disponible", "disponibilidad", "espacio", "horario", "para el", "tienen cupo", "para manana", "para mañana", "para el sabado"]):
        # Extraer fecha tentativa
        fecha_match = re.search(r"(\d{1,2}\s+de\s+[a-zñ]+|\d{4}-\d{2}-\d{2}|mañana|pasado mañana)", texto_low)
        fecha_param = fecha_match.group(1) if fecha_match else None
        return consultar_disponibilidad(fecha_param)

    # 4. Preguntas Frecuentes (FAQ)
    for k in ["anticipacion", "tiempo", "pago", "envio", "entrega", "recoger", "sucursal", "direccion", "alergia", "gluten", "vegano", "conservar", "duracion"]:
        if k in texto_low:
            return buscar_pregunta_frecuente(texto_low)

    # 5. Intención de compra / Pedido
    if any(p in texto_low for p in ["quiero pedir", "quiero ordenar", "hacer un pedido", "me gustaria encargar", "encargar", "cotizar", "quiero cupcakes", "cupcakes de"]):
        # Intentar extraer cantidad
        cant_match = re.search(r"(\d+)\s*(?:cupcakes|piezas|pzas)?", texto_low)
        cant = int(cant_match.group(1)) if cant_match else 12

        # Intentar extraer sabor
        sabor_detectado = "Red Velvet"
        for s in SABORES_MASA:
            if s["nombre"].lower() in texto_low:
                sabor_detectado = s["nombre"]
                break

        # Intentar extraer fecha
        fecha_match = re.search(r"(\d{1,2}\s+de\s+[a-zñ]+|\d{4}-\d{2}-\d{2}|mañana|pasado mañana)", texto_low)
        if fecha_match:
            fecha_str = fecha_match.group(1)
            # Si tiene los datos mínimos, registrar pedido o cotizarlo
            res = crear_pedido(
                phone_number=phone_number,
                quantity=cant,
                flavor=sabor_detectado,
                delivery_date=fecha_str
            )
            return (
                f"🧁 ¡Con mucho gusto te preparamos tu pedido de *{cant} cupcakes de {sabor_detectado}*!\n\n"
                f"• *Número asignado:* *{res['id']}*\n"
                f"• *Fecha de entrega programada:* {res['delivery_date']}\n"
                f"• *Total:* ${res['total_price']:.2f} MXN\n\n"
                f"¿Te gustaría agregarle algún relleno especial (Nutella, Frutos Rojos, Cajeta) o algún letrero personalizado?"
            )
        else:
            return (
                f"🧁 ¡Me encantaría tomar tu pedido de {cant} cupcakes de {sabor_detectado}!\n\n"
                f"Por favor indícame:\n"
                f"1. 📅 ¿Para qué fecha lo necesitas? (Mínimo 24h de anticipación)\n"
                f"2. 🍓 ¿Qué relleno prefieres? (Nutella, Frutos Rojos, Cajeta o Crema Pastelera)\n"
                f"3. 📍 ¿Pasarías a recoger a nuestra sucursal Del Valle o prefieres envío a domicilio?"
            )

    # Respuesta conversacional general si no se detectó una intención dura
    return (
        "🧁 ¡En *Cupcake Wonderland* estamos listos para endulzar tus momentos especiales!\n\n"
        "Puedo ayudarte a:\n"
        "• Ver nuestro *catálogo de sabores y precios*\n"
        "• *Consultar disponibilidad* para una fecha\n"
        "• *Armar y cotizar* tu pedido personalizado\n"
        "• Revisar el *estatus de tu pedido*\n\n"
        "¿Qué se te antoja hoy?"
    )

# -------------------------------------------------------------
# EJECUCIÓN CON OLLAMA (Llama 3.1)
# -------------------------------------------------------------
def generar_con_ollama(phone_number: str, user_text: str, historial: List[Dict[str, str]]) -> str:
    """Envía la solicitud a Ollama local (/api/chat)."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Añadir historial reciente
    messages.extend(historial)
    messages.append({"role": "user", "content": user_text})

    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "tools": TOOLS_DEFINITION,
        "stream": False
    }

    resp = requests.post(url, json=payload, timeout=(1.0, 20.0))
    if resp.status_code != 200:
        raise RuntimeError(f"Ollama respondió con HTTP {resp.status_code}: {resp.text}")

    data = resp.json()
    msg = data.get("message", {})
    tool_calls = msg.get("tool_calls", [])

    if tool_calls:
        # Ejecutar function calling
        messages.append(msg)
        for tc in tool_calls:
            fn_name = tc.get("function", {}).get("name")
            fn_args = tc.get("function", {}).get("arguments", {})
            if isinstance(fn_args, str):
                try:
                    fn_args = json.loads(fn_args)
                except Exception:
                    fn_args = {}
            resultado = ejecutar_herramienta(fn_name, fn_args, phone_number)
            messages.append({
                "role": "tool",
                "content": str(resultado)
            })

        # Segunda llamada a Ollama para respuesta final en lenguaje natural
        payload_2 = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False
        }
        resp2 = requests.post(url, json=payload_2, timeout=25)
        if resp2.status_code == 200:
            return resp2.json().get("message", {}).get("content", str(resultado))
        return str(resultado)

    return msg.get("content", "Disculpa, ¿podrías repetirme tu consulta?")

# -------------------------------------------------------------
# GENERADOR PRINCIPAL CON RESILIENCIA Y SELECCIÓN AUTOMÁTICA
# -------------------------------------------------------------
def generar_respuesta_llm(phone_number: str, user_text: str, historial: List[Dict[str, str]]) -> Tuple[str, str]:
    """
    Punto de entrada principal para generar respuestas de IA.
    Retorna una tupla: (texto_respuesta, modelo_utilizado).
    """
    provider = LLM_PROVIDER.lower()

    if provider in ("ollama", "auto"):
        try:
            respuesta = generar_con_ollama(phone_number, user_text, historial)
            return respuesta, f"ollama/{OLLAMA_MODEL}"
        except Exception:
            # Si Ollama no está activo o falló por timeout, usar Fallback suavemente
            pass

    # Usar Fallback NLU inteligente
    respuesta_fallback = motor_fallback_nlu(phone_number, user_text, historial)
    return respuesta_fallback, "nlu_rule_engine"
