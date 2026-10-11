# pip install pymupdf

import pymupdf

documento = pymupdf.open("manual.pdf")

with open("manual.txt", "w", encoding="utf-8") as salida:
    for numero, pagina in enumerate(documento, start=1):
        texto = pagina.get_text("text")

        salida.write(f"\n--- Página {numero} ---\n")
        salida.write(texto)