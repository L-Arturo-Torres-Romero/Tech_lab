"""
Simulador Interactivo de Consola para el Bot de WhatsApp de Cupcakes.
Permite chatear con el asistente directamente en la terminal sin requerir Meta ni túneles.
Uso:
    python simulator.py
"""

import sys

# Asegurar compatibilidad de emojis en consolas Windows (cp1252 -> utf-8)
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from src.services.agent_service import procesar_y_responder, reiniciar_memoria_usuario
from src.services.order_service import listar_pedidos
from src.services.catalog_service import consultar_catalogo

SIMULATED_PHONE = "525512345678"

def print_banner():
    print("\n" + "=" * 65)
    print(" 🧁  CUPCAKE WONDERLAND - SIMULADOR INTERACTIVO DE WHATSAPP  🧁 ")
    print("=" * 65)
    print(" Simula una conversación real de WhatsApp como si fueras un cliente.")
    print(" Comandos especiales:")
    print("   /menu      -> Ver catálogo oficial")
    print("   /pedidos   -> Ver lista de pedidos en base de datos")
    print("   /limpiar   -> Reiniciar memoria conversacional de este chat")
    print("   /salir     -> Salir del simulador")
    print("=" * 65)
    print(f" Teléfono simulado: {SIMULATED_PHONE}")
    print("-" * 65 + "\n")

def main():
    print_banner()
    
    # Mensaje inicial del bot
    saludo = procesar_y_responder(SIMULATED_PHONE, "hola")
    print(f"🤖 [DulceBot WhatsApp]:\n{saludo}\n")

    while True:
        try:
            user_input = input("👤 [Tú]: ").strip()
            if not user_input:
                continue

            cmd = user_input.lower()
            if cmd in ("/salir", "exit", "quit"):
                print("\n👋 ¡Hasta pronto! Que tengas un día muy dulce.\n")
                break

            elif cmd == "/menu":
                print("\n" + consultar_catalogo() + "\n")
                continue

            elif cmd == "/pedidos":
                pedidos = listar_pedidos(limit=10)
                print(f"\n📦 Pedidos registrados ({len(pedidos)}):")
                for p in pedidos:
                    print(f"   • [{p['id']}] {p['customer_name'] or 'Cliente'} - {p['quantity']}x {p['flavor']} (${p['total_price']:.2f}) | Estado: {p['status']}")
                print()
                continue

            elif cmd == "/limpiar":
                reiniciar_memoria_usuario(SIMULATED_PHONE)
                print("\n🔄 Memoria conversacional reiniciada.\n")
                continue

            # Procesar mensaje a través del agente
            respuesta = procesar_y_responder(SIMULATED_PHONE, user_input)
            print(f"\n🤖 [DulceBot WhatsApp]:\n{respuesta}\n")

        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Sesión finalizada.")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")

if __name__ == "__main__":
    main()
