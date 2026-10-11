
from pathlib import Path
import json
import re
import pymupdf

# Directorios
CARPETA_PDF = Path("pdfs")
CARPETA_SALIDA = Path("salida")

# Parámetros iniciales de chunking
CHUNK_SIZE = 3500       # caracteres, no tokens
CHUNK_OVERLAP = 400     # caracteres compartidos


def limpiar_texto(texto):
    """Limpieza conservadora del texto extraído."""
    texto = texto.replace("\x00", "")

    # Normalizar espacios horizontales sin colapsar
    # todos los saltos de línea.
    texto = re.sub(r"[ \t]+", " ", texto)

    # Reducir secuencias largas de líneas vacías.
    texto = re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", texto)

    return texto.strip()


def dividir_texto(texto, tamano=CHUNK_SIZE,
                  overlap=CHUNK_OVERLAP):
    """
    Divide por párrafos cuando es posible.
    Si un párrafo es demasiado largo, lo divide
    en segmentos con solapamiento.
    """
    if not texto:
        return []

    if tamano <= 0 or overlap < 0 or overlap >= tamano:
        raise ValueError("Parámetros de chunking inválidos.")

    parrafos = re.split(r"\n\s*\n", texto)
    fragmentos = []
    actual = ""

    for parrafo in parrafos:
        parrafo = parrafo.strip()
        if not parrafo:
            continue

        # Caso excepcional: un párrafo supera el límite.
        if len(parrafo) > tamano:
            if actual:
                fragmentos.append(actual)
                actual = ""

            inicio = 0
            while inicio < len(parrafo):
                fin = inicio + tamano
                fragmentos.append(parrafo[inicio:fin])

                if fin >= len(parrafo):
                    break

                inicio = fin - overlap
            continue

        candidato = (
            actual + "\n\n" + parrafo if actual else parrafo
        )

        if len(candidato) <= tamano:
            actual = candidato
        else:
            if actual:
                fragmentos.append(actual)

            # Solapamiento con el fragmento anterior.
            prefijo = actual[-overlap:] if actual and overlap else ""
            actual = (prefijo + "\n\n" + parrafo).strip()

            # Garantizar el límite de tamaño.
            if len(actual) > tamano:
                actual = parrafo

    if actual:
        fragmentos.append(actual)

    return fragmentos


def extraer_pdf(ruta_pdf):
    paginas = []

    with pymupdf.open(ruta_pdf) as documento:
        for numero, pagina in enumerate(documento, start=1):
            texto = pagina.get_text("text", sort=True)
            texto = limpiar_texto(texto)

            if texto:
                paginas.append({
                    "pagina": numero,
                    "texto": texto
                })

    return paginas


def procesar_pdf(ruta_pdf):
    paginas = extraer_pdf(ruta_pdf)

    # Cada fragmento pertenece a una sola página.
    # Así conservamos una referencia de página precisa.
    registros = []

    for pagina in paginas:
        fragmentos = dividir_texto(pagina["texto"])

        for indice, fragmento in enumerate(fragmentos):
            registros.append({
                "id": (
                    f"{ruta_pdf.stem}-p{pagina['pagina']}"
                    f"-c{indice}"
                ),
                "texto": fragmento,
                "metadata": {
                    "archivo": ruta_pdf.name,
                    "pagina": pagina["pagina"],
                    "tipo": "manual_tecnico"
                }
            })

    return registros


def main():
    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

    archivos = sorted(CARPETA_PDF.glob("*.pdf"))

    if not archivos:
        print(f"No se encontraron PDF en {CARPETA_PDF.resolve()}")
        return

    destino = CARPETA_SALIDA / "fragmentos.jsonl"
    total = 0

    with destino.open("w", encoding="utf-8") as salida:
        for archivo in archivos:
            try:
                registros = procesar_pdf(archivo)

                for registro in registros:
                    salida.write(
                        json.dumps(registro, ensure_ascii=False)
                        + "\n"
                    )

                total += len(registros)
                print(f"{archivo.name}: {len(registros)} fragmentos")

            except Exception as error:
                print(f"Error en {archivo.name}: {error}")

    print(f"\nTotal: {total} fragmentos")
    print(f"Salida: {destino.resolve()}")


if __name__ == "__main__":
    main()