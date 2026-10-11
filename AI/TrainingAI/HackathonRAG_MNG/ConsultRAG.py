import json
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Cargar el modelo de embeddings
modelo_embeddings = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)

# 2. Cargar los datos guardados localmente
datos = np.load("salida/base_embeddings.npz")

embeddings = datos["embeddings"]
documentos = datos["textos"].tolist()
ids = datos["ids"].tolist()

metadatos = [
    json.loads(m)
    for m in datos["metadatos"].tolist()
]

# 3. Función de consulta
def consultar_rag(pregunta, top_k=3):
    # Generar embedding de la pregunta
    embedding_pregunta = modelo_embeddings.encode(
        pregunta,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Similitud coseno: los vectores están normalizados
    similitudes = embeddings @ embedding_pregunta

    # Obtener los índices con mayor similitud
    indices = np.argsort(similitudes)[::-1][:top_k]

    # Mostrar resultados
    resultados = []

    for indice in indices:
        resultado = {
            "id": ids[indice],
            "similitud": float(similitudes[indice]),
            "texto": documentos[indice],
            "metadata": metadatos[indice]
        }
        resultados.append(resultado)

    return resultados


if __name__ == "__main__":
    # 4. Probar una consulta
    pregunta = "¿how many techniques are there for risk assessment?"

    resultados = consultar_rag(pregunta, top_k=3)

    for i, resultado in enumerate(resultados, start=1):
        print(f"\n--- Resultado {i} ---")
        print(f"ID: {resultado['id']}")
        print(f"Similitud: {resultado['similitud']:.4f}")
        print(f"Archivo: {resultado['metadata']['archivo']}")
        print(f"Página: {resultado['metadata']['pagina']}")
        print(f"Texto:\n{resultado['texto']}")