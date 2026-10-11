"""
Servicio del Catálogo y Preguntas Frecuentes de Cupcakes.
Define los sabores, rellenos, coberturas, precios y respuestas a dudas frecuentes.
"""

import unicodedata
from typing import Dict, Any, List, Optional

def quitar_acentos(texto: str) -> str:
    """Elimina acentos y normaliza el texto para búsquedas flexibles."""
    return ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')

# Catálogo oficial de la pastelería
SABORES_MASA = [
    {"nombre": "Vainilla Francesa", "descripcion": "Suave bizcocho tradicional con extracto natural de vainilla Bourbon."},
    {"nombre": "Chocolate Belga", "descripcion": "Bizcocho húmedo e intenso elaborado con cacao de alta pureza."},
    {"nombre": "Red Velvet", "descripcion": "Clásico terciopelo rojo con un toque sutil de cacao y vainilla."},
    {"nombre": "Zanahoria & Nuez", "descripcion": "Masa especiada con canela, nuez pecana troceada y zanahoria fresca."},
    {"nombre": "Limón & Amapola", "descripcion": "Bizcocho cítrico y refrescante con ralladura natural de limón y semillas de amapola."},
    {"nombre": "Café Moka", "descripcion": "Fusión perfecta de café espresso recién molido y cacao oscuro."}
]

RELLENOS = [
    {"nombre": "Sin relleno", "precio_extra": 0},
    {"nombre": "Nutella", "precio_extra": 0},
    {"nombre": "Dulce de Leche / Cajeta artesanal", "precio_extra": 0},
    {"nombre": "Compota de Frutos Rojos", "precio_extra": 0},
    {"nombre": "Crema Pastelera de Vainilla", "precio_extra": 0},
    {"nombre": "Ganache de Chocolate Blanco", "precio_extra": 0}
]

COBERTURAS = [
    {"nombre": "Buttercream Suizo Clásico", "descripcion": "Cremoso, sedoso y dulce en su punto exacto."},
    {"nombre": "Frosting de Queso Crema", "descripcion": "Ligeramente ácido y suave, el acompañante ideal de Red Velvet y Zanahoria."},
    {"nombre": "Ganache de Chocolate", "descripcion": "Cobertura aterciopelada y brillante de chocolate puro."},
    {"nombre": "Merengue Italiano Tostado", "descripcion": "Ligero, esponjoso y flameado artesanalmente."}
]

PAQUETES_PRECIOS = {
    4: {"nombre": "Caja de 4 cupcakes", "precio": 160.0, "descripcion": "Ideal para regalar o un antojo especial."},
    6: {"nombre": "Caja de 6 cupcakes", "precio": 230.0, "descripcion": "Perfecta para compartir en familia o pareja."},
    12: {"nombre": "Caja de 12 cupcakes (1 docena)", "precio": 420.0, "descripcion": "La favorita para reuniones y cumpleaños."},
    24: {"nombre": "Caja de 24 cupcakes (2 docenas)", "precio": 790.0, "descripcion": "Ideal para fiestas, bodas o eventos corporativos."},
    "mini_24": {"nombre": "Caja de 24 Mini Cupcakes", "precio": 380.0, "descripcion": "Bocadillos individuales surtidos para mesas de postres."}
}

TEMATICAS_PERSONALIZACION = [
    "Cumpleaños (con letrero de fondant y velita)",
    "Boda o Aniversario (elegantes con perlas comestibles y flores)",
    "Baby Shower / Revelación de género (tonos pastel con relleno rosa o azul sorpresa)",
    "Graduación (con birretes de chocolate y diplomas)",
    "Infantil / Personajes (figuras 2D o 3D de fondant)",
    "Foto o Logotipo comestible (impresión en oblea de azúcar en alta definición)",
    "Día del Amor / San Valentín (corazones, rosas y dedicatorias)"
]

EXTRAS_PRECIOS = {
    "figura_fondant_3d": 30.0,     # por pieza
    "foto_oblea_comestible": 40.0, # por caja
    "caja_regalo_lujo": 35.0,      # caja con listón de satín y tarjeta
    "envio_domicilio": 60.0        # tarifa base promedio dentro de zona de cobertura
}

PREGUNTAS_FRECUENTES_CUPCAKES = {
    "anticipacion": "Pedimos un mínimo de 24 a 48 horas de anticipación para pedidos personalizados. Para eventos grandes (más de 48 piezas), recomendamos al menos 3 a 5 días de antelación.",
    "tiempo": "Pedimos un mínimo de 24 a 48 horas de anticipación para pedidos personalizados. Para eventos grandes (más de 48 piezas), recomendamos al menos 3 a 5 días de antelación.",
    "pago": "Aceptamos transferencia bancaria (SPEI), tarjetas de crédito/débito y efectivo al recoger. Para confirmar el pedido se solicita el 50% de anticipo y el resto a la entrega.",
    "metodos de pago": "Aceptamos transferencia bancaria (SPEI), tarjetas de crédito/débito y efectivo al recoger. Para confirmar el pedido se solicita el 50% de anticipo y el resto a la entrega.",
    "envio": "Realizamos entregas a domicilio en CDMX y área metropolitana con costo según código postal (desde $60 MXN). También puedes recoger sin costo en nuestra sucursal matriz en Colonia Del Valle.",
    "entrega": "Realizamos entregas a domicilio en CDMX y área metropolitana con costo según código postal (desde $60 MXN). También puedes recoger sin costo en nuestra sucursal matriz en Colonia Del Valle.",
    "recoger": "Puedes recoger tu pedido sin costo en nuestra sucursal matriz ubicada en Colonia Del Valle, CDMX, en horario de 9:00 am a 7:00 pm de lunes a sábado.",
    "direccion": "Nuestra sucursal matriz está ubicada en Colonia Del Valle, Ciudad de México. El horario de atención para recoger pedidos es de 9:00 a 19:00 hrs.",
    "horario": "Abrimos de lunes a sábado de 9:00 am a 7:00 pm, y domingos de 10:00 am a 3:00 pm.",
    "alergia": "Nuestros cupcakes estándar contienen harina de trigo (gluten), huevos y lácteos. Podemos elaborar versiones sin nueces o sin gluten con aviso previo de al menos 48 horas. Avisa siempre si tienes alergias severas.",
    "gluten": "Podemos preparar cupcakes sin gluten con al menos 48 horas de anticipación. Por favor indícalo claramente al hacer tu pedido.",
    "vegano": "Contamos con opciones veganas (sin leche ni huevo) en sabores chocolate y vainilla bajo pedido especial de al menos 12 unidades.",
    "duracion": "Nuestros cupcakes se conservan frescos hasta 3 días a temperatura ambiente en lugar fresco, o hasta 5 días en refrigeración dentro de un contenedor hermético.",
    "conservar": "Nuestros cupcakes se conservan frescos hasta 3 días a temperatura ambiente en lugar fresco, o hasta 5 días en refrigeración dentro de un contenedor hermético.",
    "cancelacion": "Puedes cancelar o reprogramar tu pedido sin penalización con al menos 24 horas de anticipación a la fecha de entrega acordada."
}

def consultar_catalogo(categoria: Optional[str] = None) -> str:
    """
    Retorna el catálogo formateado en texto amigable para el cliente de WhatsApp.
    Si se proporciona una categoría ('sabores', 'rellenos', 'coberturas', 'paquetes', 'tematicas'), filtra por ella.
    """
    cat = categoria.lower().strip() if categoria else "todo"
    
    if "sabor" in cat or "masa" in cat:
        sabores = "\n".join([f"• *{s['nombre']}*: {s['descripcion']}" for s in SABORES_MASA])
        return f"🧁 *NUESTROS SABORES DE BIZCOCHO:*\n\n{sabores}"
    
    if "relleno" in cat:
        rellenos = "\n".join([f"• *{r['nombre']}*" for r in RELLENOS])
        return f"🍓 *RELLENOS DISPONIBLES:*\n\n{rellenos}\n_(¡Todos incluidos sin costo extra!)_"
        
    if "cobertura" in cat or "frosting" in cat:
        frostings = "\n".join([f"• *{c['nombre']}*: {c['descripcion']}" for c in COBERTURAS])
        return f"🍦 *COBERTURAS Y FROSTINGS:*\n\n{frostings}"
        
    if "paquete" in cat or "precio" in cat or "costo" in cat:
        paquetes = "\n".join([f"• *{p['nombre']}*: ${p['precio']:.0f} MXN — _{p['descripcion']}_" for p in PAQUETES_PRECIOS.values()])
        return f"📦 *PAQUETES Y PRECIOS:*\n\n{paquetes}"

    if "tematica" in cat or "diseno" in cat:
        temas = "\n".join([f"• {t}" for t in TEMATICAS_PERSONALIZACION])
        return f"🎨 *TEMÁTICAS DE PERSONALIZACIÓN:*\n\n{temas}\n\n_Podemos incluir letreros con nombres, números y fotos comestibles._"

    # Catálogo resumido completo
    sabores_txt = ", ".join([s["nombre"] for s in SABORES_MASA])
    rellenos_txt = ", ".join([r["nombre"] for r in RELLENOS if r["nombre"] != "Sin relleno"])
    paquetes_txt = "\n".join([f"• *{p['nombre']}*: ${p['precio']:.0f} MXN" for p in PAQUETES_PRECIOS.values()])

    return (
        "🧁 *MENÚ DE CUPCAKES PERSONALIZADOS*\n\n"
        f"🍰 *Sabores de Masa:* {sabores_txt}\n"
        f"🍓 *Rellenos:* {rellenos_txt}\n"
        f"🍦 *Coberturas:* Buttercream Suizo, Queso Crema, Ganache de Chocolate, Merengue\n\n"
        f"📦 *Paquetes:*\n{paquetes_txt}\n\n"
        "🎨 *Personalización:* Cumpleaños, Bodas, Baby Shower, fotos comestibles o figuras de fondant.\n"
        "_Dime qué cantidad y temática tienes en mente y con gusto te ayudo a cotizarlo._"
    )

def buscar_pregunta_frecuente(pregunta: str) -> str:
    """Busca respuestas a preguntas frecuentes sobre la pastelería."""
    pregunta_norm = quitar_acentos(pregunta.lower().strip())
    for clave, respuesta in PREGUNTAS_FRECUENTES_CUPCAKES.items():
        if clave in pregunta_norm:
            return respuesta
    return "No encontré una respuesta específica para tu consulta. Un asesor repostero de nuestro equipo con gusto te atenderá personalmente."

def calcular_precio_pedido(cantidad: int, con_foto: bool = False, con_figura_3d: bool = False, domicilio: bool = False) -> float:
    """Calcula el costo total estimado de un pedido de cupcakes."""
    # Buscar precio base por cantidad
    if cantidad in PAQUETES_PRECIOS:
        base = PAQUETES_PRECIOS[cantidad]["precio"]
    elif cantidad >= 24:
        # Tarifa unitaria preferencial para más de 24 piezas
        base = cantidad * 32.0
    else:
        # Tarifa unitaria regular ($38 c/u)
        base = cantidad * 38.0
        
    extra = 0.0
    if con_foto:
        extra += EXTRAS_PRECIOS["foto_oblea_comestible"]
    if con_figura_3d:
        extra += EXTRAS_PRECIOS["figura_fondant_3d"] * (min(cantidad, 4))
    if domicilio:
        extra += EXTRAS_PRECIOS["envio_domicilio"]
        
    return round(base + extra, 2)
