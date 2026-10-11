"""
Servicio de Gestión de Pedidos de Cupcakes.
Maneja la creación, consulta, validación de fechas y actualización de estados en base de datos.
"""

import re
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from src.database import get_connection
from src.services.catalog_service import calcular_precio_pedido

MESES_ES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12
}

def normalizar_fecha(fecha_texto: str) -> str:
    """Normaliza texto de fecha (ej. '15 de octubre' o '2026-10-15') a formato YYYY-MM-DD."""
    if not fecha_texto:
        return ""
    texto = fecha_texto.strip().lower()
    
    # Formato ISO ya listo: YYYY-MM-DD
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", texto):
        return texto
        
    # Formato DD/MM/YYYY o DD-MM-YYYY
    match_slash = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", texto)
    if match_slash:
        dia, mes, anio = match_slash.groups()
        return f"{int(anio):04d}-{int(mes):02d}-{int(dia):02d}"

    # Formato texto: '15 de octubre' o '15 de octubre de 2026'
    match_texto = re.search(r"(\d{1,2})\s+de\s+([a-zñ]+)(?:\s+de\s+(\d{4}))?", texto)
    if match_texto:
        dia, mes_str, anio_str = match_texto.groups()
        mes = MESES_ES.get(mes_str)
        if mes:
            anio = int(anio_str) if anio_str else datetime.now().year
            return f"{anio:04d}-{mes:02d}-{int(dia):02d}"

    # Referencias relativas: 'mañana', 'pasado mañana'
    now = datetime.now()
    if "pasado manana" in texto or "pasado mañana" in texto:
        return (now + timedelta(days=2)).strftime("%Y-%m-%d")
    if "manana" in texto or "mañana" in texto:
        return (now + timedelta(days=1)).strftime("%Y-%m-%d")

    return texto

def consultar_disponibilidad(fecha: Optional[str] = None) -> str:
    """
    Verifica si hay cupo disponible para hornear y entregar pedidos en una fecha.
    Requiere al menos 24 horas de anticipación.
    """
    if not fecha:
        manana = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        pasado = (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d")
        return f"📅 Para pedidos personalizados necesitamos mínimo 24h de anticipación. Tenemos fechas abiertas a partir de mañana ({manana}) y pasado mañana ({pasado}). ¿Para qué fecha lo necesitas?"

    fecha_norm = normalizar_fecha(fecha)
    
    # Validar formato de fecha
    try:
        fecha_dt = datetime.strptime(fecha_norm, "%Y-%m-%d")
    except ValueError:
        return f"No pude reconocer la fecha '{fecha}'. Por favor indícamela en formato 'día de mes' (por ejemplo, '15 de octubre')."

    # Validar anticipación mínima (al menos el día siguiente)
    hoy = datetime.now().date()
    if fecha_dt.date() <= hoy:
        return f"⚠️ Los pedidos personalizados requieren al menos 24 horas de anticipación para garantizar la frescura de los cupcakes. La fecha más pronta disponible es {(hoy + timedelta(days=1)).strftime('%Y-%m-%d')}."

    # Consultar cupo en base de datos
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT max_orders, booked_orders FROM daily_capacity WHERE date = ?", (fecha_norm,))
    row = cursor.fetchone()
    
    max_orders = row["max_orders"] if row else 12
    booked_orders = row["booked_orders"] if row else 0
    conn.close()

    cupos_libres = max_orders - booked_orders
    if cupos_libres <= 0:
        return f"❌ Lo sentimos, para el {fecha_norm} nuestra agenda de horneado ya está al 100% de capacidad. Te sugerimos seleccionar el día siguiente."

    return f"✅ ¡Excelente noticia! Tenemos cupo disponible para el {fecha_norm} (nos quedan {cupos_libres} espacios de entrega). ¿Deseas que armemos tu pedido?"

def generar_siguiente_id() -> str:
    """Genera un ID secuencial para el pedido garantizando que sea mayor a todos los existentes."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM orders WHERE id LIKE 'CUP-%'")
    rows = cursor.fetchall()
    conn.close()

    max_num = 1002
    for r in rows:
        match = re.search(r"CUP-(\d+)", r["id"])
        if match:
            max_num = max(max_num, int(match.group(1)))

    return f"CUP-{max_num + 1:04d}"

def crear_pedido(
    phone_number: str,
    customer_name: Optional[str] = "Cliente",
    quantity: int = 12,
    flavor: str = "Red Velvet",
    filling: Optional[str] = "Frutos Rojos",
    frosting: Optional[str] = "Frosting de Queso Crema",
    theme_design: Optional[str] = "Personalizado",
    custom_text: Optional[str] = None,
    delivery_date: Optional[str] = None,
    delivery_type: str = "recoger",
    delivery_address: Optional[str] = None,
    notes: Optional[str] = None
) -> Dict[str, Any]:
    """Crea y registra un nuevo pedido de cupcakes en la base de datos."""
    fecha_norm = normalizar_fecha(delivery_date) if delivery_date else (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d")
    
    con_foto = bool(theme_design and ("foto" in theme_design.lower() or "logo" in theme_design.lower()))
    con_3d = bool(theme_design and ("3d" in theme_design.lower() or "figura" in theme_design.lower()))
    es_domicilio = delivery_type.lower() == "domicilio"
    
    total = calcular_precio_pedido(
        cantidad=quantity,
        con_foto=con_foto,
        con_figura_3d=con_3d,
        domicilio=es_domicilio
    )
    
    order_id = generar_siguiente_id()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
    INSERT INTO orders (
        id, phone_number, customer_name, quantity, flavor, filling, frosting,
        theme_design, custom_text, delivery_date, delivery_type, delivery_address,
        total_price, status, notes, created_at, updated_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        order_id, phone_number, customer_name, quantity, flavor, filling, frosting,
        theme_design, custom_text, fecha_norm, delivery_type, delivery_address,
        total, "PENDIENTE_PAGO", notes, now_str, now_str
    ))

    # Incrementar cupo reservado para esa fecha
    cursor.execute("""
    INSERT INTO daily_capacity (date, max_orders, booked_orders)
    VALUES (?, 12, 1)
    ON CONFLICT(date) DO UPDATE SET booked_orders = booked_orders + 1
    """, (fecha_norm,))

    conn.commit()
    conn.close()

    return {
        "id": order_id,
        "phone_number": phone_number,
        "customer_name": customer_name,
        "quantity": quantity,
        "flavor": flavor,
        "filling": filling,
        "frosting": frosting,
        "theme_design": theme_design,
        "custom_text": custom_text,
        "delivery_date": fecha_norm,
        "delivery_type": delivery_type,
        "delivery_address": delivery_address,
        "total_price": total,
        "status": "PENDIENTE_PAGO"
    }

def consultar_pedido(numero_pedido: Optional[str] = None, phone_number: Optional[str] = None) -> str:
    """Busca y formatea la información de un pedido por su número o por teléfono."""
    conn = get_connection()
    cursor = conn.cursor()

    if numero_pedido:
        num_limpio = str(numero_pedido).strip().upper()
        # Agregar prefijo si el usuario escribió solo los dígitos
        if num_limpio.isdigit():
            num_limpio = f"CUP-{num_limpio}"
        cursor.execute("SELECT * FROM orders WHERE id = ?", (num_limpio,))
    elif phone_number:
        cursor.execute("SELECT * FROM orders WHERE phone_number = ? ORDER BY created_at DESC LIMIT 1", (phone_number,))
    else:
        conn.close()
        return "Por favor proporciona tu número de pedido (ejemplo: *CUP-1001*) para revisarlo en el sistema."

    row = cursor.fetchone()
    conn.close()

    if not row:
        return f"No encontramos ningún pedido registrado con la referencia *{numero_pedido}*. Por favor verifica los datos o solicítanos asistencia."

    pedido = dict(row)
    estados_texto = {
        "COTIZACION": "📝 En Cotización",
        "PENDIENTE_PAGO": "⏳ Pendiente de anticipo (50%)",
        "CONFIRMADO": "✅ Confirmado y programado en cocina",
        "EN_PREPARACION": "👩‍🍳 ¡En el horno y decorándose!",
        "LISTO": "📦 ¡Listo para entrega o recogida!",
        "ENTREGADO": "🎉 Entregado",
        "CANCELADO": "❌ Cancelado"
    }
    
    estado_humano = estados_texto.get(pedido["status"], pedido["status"])

    return (
        f"🧁 *DETALLE DE TU PEDIDO {pedido['id']}*\n\n"
        f"• *Estado:* {estado_humano}\n"
        f"• *Cantidad:* {pedido['quantity']} cupcakes\n"
        f"• *Sabor:* {pedido['flavor']} con relleno de {pedido['filling'] or 'Sin relleno'}\n"
        f"• *Cobertura:* {pedido['frosting']}\n"
        f"• *Diseño / Temática:* {pedido['theme_design']}\n"
        f"• *Fecha de Entrega:* {pedido['delivery_date']}\n"
        f"• *Modalidad:* {'Envío a domicilio' if pedido['delivery_type'] == 'domicilio' else 'Recoger en sucursal'}\n"
        f"• *Total:* ${pedido['total_price']:.2f} MXN\n\n"
        "_Si deseas modificar algún dato o tienes preguntas, estamos a tu orden._"
    )

def actualizar_estado_pedido(order_id: str, nuevo_estado: str, notes: Optional[str] = None) -> bool:
    """Actualiza el estado de un pedido en la base de datos."""
    conn = get_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    cursor.execute("""
    UPDATE orders
    SET status = ?, notes = COALESCE(?, notes), updated_at = ?
    WHERE id = ?
    """, (nuevo_estado, notes, now_str, order_id))
    
    updated = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return updated

def listar_pedidos(limit: int = 50, status: Optional[str] = None) -> List[Dict[str, Any]]:
    """Lista los pedidos registrados en el sistema."""
    conn = get_connection()
    cursor = conn.cursor()
    
    if status:
        cursor.execute("SELECT * FROM orders WHERE status = ? ORDER BY created_at DESC LIMIT ?", (status, limit))
    else:
        cursor.execute("SELECT * FROM orders ORDER BY created_at DESC LIMIT ?", (limit,))
        
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]
