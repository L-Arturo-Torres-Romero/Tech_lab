
import sys
from ConsultAI_RAG import preguntar_rag


def AskManagerAssistant(pregunta: str) -> str:
    return preguntar_rag(
        pregunta,
        top_k=3,
        max_new_tokens=1024
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Uso: python AskManagerAssistant.py "Tu pregunta"')
        sys.exit(1)

    pregunta = " ".join(sys.argv[1:])
    respuesta = AskManagerAssistant(pregunta)
    print("\n" + "************************************************************" + "\n")
    print("\n" + respuesta + "\n")

