import os

import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


def _get_secret(name):
    """Read a key from Streamlit secrets first, then from the environment (.env)."""
    try:
        val = st.secrets.get(name)
        if val:
            return val
    except Exception:
        pass
    return os.getenv(name)


GROQ_API_KEY = _get_secret("GROQ_API") or _get_secret("GROQ_API_KEY")
LANGCHAIN_API_KEY = _get_secret("LANGCHAIN_API_KEY")

# LangSmith tracing (optional; app still works without it)
if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = "Imans-Chatbot-Groq"

PROVIDER_GROQ = "Groq Cloud (recommended for Streamlit)"
PROVIDER_OLLAMA = "Ollama (local server)"

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
    "qwen/qwen3.6-27b",
    "phi3:mini",
    "gemma:2b",
    "tinyllama",
    "moondream",
]

OLLAMA_MODELS = ["phi3:mini", "gemma:2b", "tinyllama", "moondream"]

# The 4 models below are Ollama models that Groq doesn't host by those exact
# names. In the cloud they are served by the closest Groq-hosted open model so
# the app keeps working with no local server (Groq's free tier is stable).
GROQ_EQUIVALENTS = {
    "phi3:mini": "llama-3.1-8b-instant",
    "gemma:2b": "llama-3.3-70b-versatile",
    "tinyllama": "llama-3.1-8b-instant",
    "moondream": "groq/compound-mini",
}

## prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful and informative assistant designed for question answering.
                    Please provide detailed and insightful responses to user queries.
                    If a user asks for help, offer a comprehensive overview of the chatbot's features, including:

                    * **How to ask questions:** Explain how users can phrase their questions effectively. For instance, mention using clear and concise language, and the types of information the bot is best equipped to handle.
                    * **Available commands:** If there are specific keywords or commands, list them and describe their functionality. This could be things like "help," "examples," or any other custom commands you might want to add.
                    * **Supported topics:** Briefly mention the chatbot's knowledge domain. Is it focused on a specific area like data science or general knowledge? This helps users to know what kind of questions to ask.
                    * **Examples:** Provide a few sample questions to demonstrate how to interact with the chatbot effectively.

                    Always be friendly, patient, and aim to provide the most useful information possible.""",
        ),
        ("user", "Question: {question}"),
    ]
)

output_parser = StrOutputParser()


def generate_response(question, provider, model, temperature, max_tokens):
    try:
        # Ollama-only models picked in the cloud are served by the closest
        # Groq-hosted open model with the same role/size intent.
        if provider == PROVIDER_GROQ and model in GROQ_EQUIVALENTS:
            if not GROQ_API_KEY:
                return (
                    "Groq API key is missing. Add it locally in the `.env` file as "
                    "`GROQ_API=...`, or in Streamlit Cloud go to Settings > Secrets and add "
                    "`GROQ_API = \"...\"`."
                )
            from langchain_groq import ChatGroq

            llm = ChatGroq(
                model=GROQ_EQUIVALENTS[model],
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=GROQ_API_KEY,
            )
        elif provider == PROVIDER_GROQ:
            if not GROQ_API_KEY:
                return (
                    "Groq API key is missing. Add it locally in the `.env` file as "
                    "`GROQ_API=...`, or in Streamlit Cloud go to Settings > Secrets and add "
                    "`GROQ_API = \"...\"`."
                )
            from langchain_groq import ChatGroq

            llm = ChatGroq(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=GROQ_API_KEY,
            )
        else:
            from langchain_ollama import OllamaLLM

            llm = OllamaLLM(
                model=model,
                temperature=temperature,
                num_predict=max_tokens,
            )

        chain = prompt | llm | output_parser
        return chain.invoke({"question": question})
    except Exception as exc:
        if provider == PROVIDER_GROQ and "does not exist" in str(exc).lower():
            return (
                f"`{model}` isn't available on the selected cloud provider. "
                "Try another model from the list, or switch to the Ollama provider "
                "if you're running the app locally."
            )
        return f"Something went wrong: {exc}"


## Title of the app
st.title("🤖 Iman Your Personal AI 🍓🌠")

## Sidebar settings
st.sidebar.header("Model Settings")
provider = st.sidebar.selectbox(
    "Model Provider",
    [PROVIDER_GROQ, PROVIDER_OLLAMA],
)
if provider == PROVIDER_GROQ:
    model = st.sidebar.selectbox("Select Groq Model", GROQ_MODELS)
    if model in GROQ_EQUIVALENTS:
        st.sidebar.info(
            f"`{model}` is a local Ollama model; in the cloud it's served by "
            f"Groq's `{GROQ_EQUIVALENTS[model]}` so it works automatically."
        )
    else:
        status = (
            "Groq API key configured."
            if GROQ_API_KEY
            else "Groq API key is NOT set."
        )
        st.sidebar.info(status)
else:
    model = st.sidebar.selectbox("Select Local Ollama Model", OLLAMA_MODELS)
    st.sidebar.warning(
        "Ollama requires a running local server (http://localhost:11434). "
        "It will NOT work on Streamlit Cloud. Use Groq for the deployed app."
    )

temperature = st.sidebar.slider(
    "Temperature", min_value=0.0, max_value=1.0, value=0.7
)
max_tokens = st.sidebar.slider(
    "Max Tokens", min_value=50, max_value=300, value=150
)

## Main interface for using input
st.write("🌈 I'm curious! What wonders do you have for me today? 🤔 Ask away! 🌟")
user_input = st.text_input("You: ")

if user_input:
    with st.spinner("Thinking..."):
        response = generate_response(user_input, provider, model, temperature, max_tokens)
    st.write(response)
else:
    st.write("Please provide the user input")