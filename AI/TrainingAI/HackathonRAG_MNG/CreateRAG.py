
# pip install sentence-transformers numpy
import json
from sentence_transformers import SentenceTransformer
import numpy as np
import json
from pathlib import Path

# 1. Cargar el modelo de embeddings local
modelo_embeddings = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)

# 2. Leer los fragmentos del archivo JSONL
documentos = []
metadatos = []
ids = []

with open("salida/fragmentos.jsonl", "r", encoding="utf-8") as archivo:
    for linea in archivo:
        if not linea.strip():
            continue

        registro = json.loads(linea)

        documentos.append(registro["texto"])
        metadatos.append(registro["metadata"])
        ids.append(registro["id"])

# 3. Generar los embeddings
embeddings_documentos = modelo_embeddings.encode(
    documentos,
    convert_to_numpy=True,
    show_progress_bar=True,
    normalize_embeddings=True
)

salida = Path("salida")
salida.mkdir(exist_ok=True)

np.savez_compressed(
    salida / "base_embeddings.npz",
    embeddings=embeddings_documentos,
    textos=np.array(documentos),
    ids=np.array(ids),
    metadatos=np.array([
        json.dumps(m, ensure_ascii=False)
        for m in metadatos
    ])
)

# 4. Comprobar el resultado
print("Número de documentos:", len(documentos))
print("Forma de los embeddings:", embeddings_documentos.shape)
print("Primer ID:", ids[0] if ids else "Sin documentos")
print("Primer texto:", documentos[0] if documentos else "Sin documentos")