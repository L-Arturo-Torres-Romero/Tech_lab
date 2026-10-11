"""
Módulo de Base de Datos SQLite.
Gestiona la persistencia de pedidos, bitácora de mensajes y capacidad diaria de producción.
"""

import sqlite3
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from src.config import DATABASE_PATH

def get_connection() -> sqlite3.Connection:
    """Obtiene una conexión a la base de datos SQLite con row_factory como dict y WAL mode."""
    db_file = Path(DATABASE_PATH)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(db_file), timeout=30.0, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=30000;")
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    """Inicializa las tablas de la base de datos y crea datos semilla si está vacía."""
    conn = get_connection()
    cursor = conn.cursor()

    # Tabla de pedidos
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS orders (
        id TEXT PRIMARY KEY,
        phone_number TEXT NOT NULL,
        customer_name TEXT,
        quantity INTEGER NOT NULL,
        flavor TEXT NOT NULL,
        filling TEXT,
        frosting TEXT,
        theme_design TEXT,
        custom_text TEXT,
        delivery_date TEXT NOT NULL,
        delivery_type TEXT DEFAULT 'recoger',
        delivery_address TEXT,
        total_price REAL NOT NULL,
        status TEXT DEFAULT 'COTIZACION',
        notes TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """)

    # Tabla de bitácora de mensajes de WhatsApp
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS messages_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        wamid TEXT UNIQUE,
        direction TEXT NOT NULL,
        phone_number TEXT NOT NULL,
        message_text TEXT NOT NULL,
        response_text TEXT,
        status TEXT,
        timestamp TEXT NOT NULL
    )
    """)

    # Tabla de capacidad diaria de producción
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS daily_capacity (
        date TEXT PRIMARY KEY,
        max_orders INTEGER DEFAULT 12,
        booked_orders INTEGER DEFAULT 0
    )
    """)

    # Verificar si hay pedidos semilla para pruebas inmediatas
    cursor.execute("SELECT COUNT(*) as count FROM orders")
    count = cursor.fetchone()["count"]
    if count == 0:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        seed_orders = [
            (
                "CUP-1001", "525512345678", "María González", 12,
                "Red Velvet", "Frutos Rojos", "Frosting de Queso Crema",
                "Temática Cumpleaños con letrero dorado", "Feliz Cumple Sofía",
                "2026-10-15", "domicilio", "Av. Insurgentes Sur 1200, CDMX",
                450.0, "CONFIRMADO", "Entregar antes de las 3pm", now, now
            ),
            (
                "CUP-1002", "525598765432", "Carlos Pérez", 6,
                "Chocolate Belga", "Nutella", "Ganache de Chocolate",
                "Decoración Aniversario con corazones", "Te amo",
                "2026-10-14", "recoger", "Sucursal Matriz",
                250.0, "EN_PREPARACION", "Sin nueces por precaución", now, now
            )
        ]
        cursor.executemany("""
        INSERT INTO orders (
            id, phone_number, customer_name, quantity, flavor, filling, frosting,
            theme_design, custom_text, delivery_date, delivery_type, delivery_address,
            total_price, status, notes, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, seed_orders)

    conn.commit()
    conn.close()

def log_message(phone_number: str, direction: str, message_text: str, wamid: Optional[str] = None, response_text: Optional[str] = None, status: str = "ok") -> None:
    """Registra un mensaje en la bitácora."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
        INSERT OR IGNORE INTO messages_log (wamid, direction, phone_number, message_text, response_text, status, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (wamid, direction, phone_number, message_text, response_text, status, now))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error al registrar mensaje en BD: {e}")

# Inicializar BD al importar
init_db()
