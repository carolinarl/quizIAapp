import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Prueba de Conocimiento", layout="centered")


PREGUNTAS = [
    {
        "pregunta": "¿Cuál es el rango normal de la temperatura corporal en un adulto?",
        "opciones": ["36.1°C - 37.2°C", "35.0°C - 36.0°C", "37.3°C - 38.0°C", "38.1°C - 39.0°C"],
        "correcta": 0,
        "categoria": "Signos vitales",
    },
    {
        "pregunta": "¿Cuál es el órgano encargado principalmente de filtrar la sangre y producir orina?",
        "opciones": ["Hígado", "Riñón", "Corazón", "Pulmón"],
        "correcta": 1,
        "categoria": "Anatomía",
    },
    {
        "pregunta": "¿Qué vitamina se sintetiza principalmente por la exposición al sol?",
        "opciones": ["Vitamina A", "Vitamina B", "Vitamina C", "Vitamina D"],
        "correcta": 3,
        "categoria": "Nutrición",
    },
    {
        "pregunta": "¿Cuál es el valor normal aproximado de la presión arterial en un adulto sano?",
        "opciones": ["90/60 mmHg", "120/80 mmHg", "140/90 mmHg", "160/100 mmHg"],
        "correcta": 1,
        "categoria": "Signos vitales",
    },
    {
        "pregunta": "¿Qué tipo de diabetes se caracteriza por una destrucción autoinmune de las células beta pancreáticas?",
        "opciones": ["Diabetes tipo 1", "Diabetes tipo 2", "Diabetes gestacional", "Síndrome de resistencia a la insulina"],
        "correcta": 0,
        "categoria": "Endocrinología",
    },
    {
        "pregunta": "¿Cuál es el principal componente de los glóbulos rojos encargado de transportar oxígeno?",
        "opciones": ["Hemoglobina", "Leucocitos", "Plaquetas", "Plasma"],
        "correcta": 0,
        "categoria": "Hematología",
    },
    {
        "pregunta": "¿Cuál de los siguientes síntomas es típico de una hipoglucemia?",
        "opciones": ["Sudoración", "Confusión", "Palpitaciones", "Todas las anteriores"],
        "correcta": 3,
        "categoria": "Endocrinología",
    },
    {
        "pregunta": "¿Qué órgano produce la insulina?",
        "opciones": ["Hígado", "Riñón", "Páncreas", "Corazón"],
        "correcta": 2,
        "categoria": "Endocrinología",
    },
    {
        "pregunta": "¿Cuál de las siguientes enfermedades es causada por un virus?",
        "opciones": ["Tuberculosis", "Influenza", "Diabetes", "Hipertensión"],
        "correcta": 1,
        "categoria": "Infectología",
    },
    {
        "pregunta": "¿Qué prueba mide los niveles promedio de glucosa de los últimos 2 a 3 meses?",
        "opciones": ["Hemoglobina glucosilada (HbA1c)", "Glucosa capilar", "Insulina sérica", "Curva de tolerancia."],
        "correcta": 0,
        "categoria": "Endocrinología",
    },
]


@st.cache_resource(show_spinner="Cargando modelo de IA...")
def cargar_modelo():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        device=-1
    )


def generar_evaluacion_ia(correctas, total, porcentaje, temas_fallados):
    generador = cargar_modelo()

    temas_texto = ", ".join(temas_fallados) if temas_fallados else "ninguno, respondió todo bien"

    prompt = (
        f"You are a medical education evaluator. A student answered {correctas} out of {total} "
        f"questions correctly ({porcentaje:.0f}%). Topics they got wrong: {temas_texto}. "
        f"Write a short evaluation in Spanish (2 sentences) of their knowledge level "
        f"(choose one: Debil, Intermedio, Avanzado, Dominado) and one brief study recommendation."
    )

    resultado = generador(prompt, max_new_tokens=80, do_sample=False)
    return resultado[0]["generated_text"]


def nivel_base_por_porcentaje(porcentaje):
    if porcentaje >= 90:
        return "Dominado"
    elif porcentaje >= 70:
        return "Intermedio"
    else:
        return "Débil"


st.title("Prueba de Conocimiento")
st.write("Responde las 10 preguntas para determinar tu nivel de conocimiento en temas de salud y medicina.")

respuestas_usuario = []

with st.form("quiz_form"):
    for i, item in enumerate(PREGUNTAS):
        st.subheader(f"{i + 1}. {item['pregunta']}")
        respuesta = st.radio(
            "Selecciona una opción:",
            item["opciones"],
            key=f"pregunta_{i}",
            index=None,
        )
        respuestas_usuario.append(respuesta)

    enviado = st.form_submit_button("Enviar respuestas")

if enviado:
    if None in respuestas_usuario:
        st.warning("Por favor responde todas las preguntas antes de enviar.")
    else:
        correctas = 0
        temas_fallados = []

        for i, item in enumerate(PREGUNTAS):
            seleccion_idx = item["opciones"].index(respuestas_usuario[i])
            if seleccion_idx == item["correcta"]:
                correctas += 1
            else:
                temas_fallados.append(item["categoria"])

        temas_fallados = list(dict.fromkeys(temas_fallados))  # sin duplicados, conserva orden
        total = len(PREGUNTAS)
        porcentaje = (correctas / total) * 100

        st.divider()
        st.header(f"Resultado: {correctas}/{total} ({porcentaje:.0f}%)")

        nivel_objetivo = nivel_base_por_porcentaje(porcentaje)
        st.subheader(f"Calificación: {nivel_objetivo}")

        with st.spinner("Evaluación IA..."):
            try:
                evaluacion_ia = generar_evaluacion_ia(correctas, total, porcentaje, temas_fallados)
                st.info(f"**Evaluación de la IA:** {evaluacion_ia}")
            except Exception as e:
                st.warning(
                    " "
                )

        with st.expander("Ver detalle de respuestas"):
            for i, item in enumerate(PREGUNTAS):
                correcta_texto = item["opciones"][item["correcta"]]
                usuario_texto = respuestas_usuario[i]
                if usuario_texto == correcta_texto:
                    st.success(f"{i+1}. {item['pregunta']} Tu respuesta: {usuario_texto}")
                else:
                    st.error(
                        f"{i+1}. {item['pregunta']} Tu respuesta: {usuario_texto} "
                        f"| Correcta: {correcta_texto} | Tema: {item['categoria']}"
                    )