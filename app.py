import os
import re
import uuid
import streamlit as st
from dotenv import load_dotenv
from supabase import create_client
from fastembed import TextEmbedding
from groq import Groq
import speech_recognition as sr
from gtts import gTTS
import io

load_dotenv()

st.set_page_config(page_title="Agentic RAG Assistant", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0f0f0f; }
    .block-container { max-width: 780px; padding-top: 3rem; padding-bottom: 8rem; }

    .welcome-text {
        text-align: center;
        font-size: 2rem;
        font-weight: 600;
        color: #ececec;
        margin-top: 12vh;
        margin-bottom: 2rem;
    }

    .user-msg { display: flex; justify-content: flex-end; margin: 14px 0; }
    .user-bubble {
        background-color: #2f2f2f;
        color: #ececec;
        padding: 10px 16px;
        border-radius: 20px;
        max-width: 70%;
        font-size: 1rem;
        line-height: 1.5;
    }
    .assistant-msg { display: flex; justify-content: flex-start; margin: 14px 0; }
    .assistant-text {
        color: #ececec;
        max-width: 85%;
        font-size: 1rem;
        line-height: 1.7;
    }

    section[data-testid="stSidebar"] {
        background-color: #171717;
        border-right: 1px solid #2d2d2d;
    }
    section[data-testid="stSidebar"] .stButton button {
        background-color: transparent;
        border: none;
        text-align: left;
        color: #d1d1d1;
        font-size: 0.9rem;
    }
    section[data-testid="stSidebar"] .stButton button:hover {
        background-color: #2a2a2a;
        color: #fff;
    }

    div[data-testid="stChatInput"] {
        max-width: 780px;
        margin: 0 auto;
    }
    div[data-testid="stChatInput"] textarea {
        background-color: #2f2f2f !important;
        border-radius: 24px !important;
        color: #ececec !important;
        border: 1px solid #3d3d3d !important;
    }

    .voice-call-btn button {
        background: linear-gradient(135deg, #6366f1, #a855f7) !important;
        border-radius: 50% !important;
        width: 52px !important;
        height: 52px !important;
        border: none !important;
        font-size: 1.3rem !important;
    }

    .attach-btn button {
        background-color: #2f2f2f !important;
        border-radius: 50% !important;
        width: 44px !important;
        height: 44px !important;
        border: 1px solid #3d3d3d !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Backend Setup ----------
@st.cache_resource
def init_clients():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    supabase = create_client(url, key)
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return supabase, groq_client, embed_model

supabase, groq_client, embed_model = init_clients()
recognizer = sr.Recognizer()

# ---------- Core RAG Functions ----------
def retrieve_chunks(question, match_count=3):
    question_embedding = list(embed_model.embed([question]))[0].tolist()
    response = supabase.rpc("match_chunks", {
        "query_embedding": question_embedding,
        "match_threshold": 0.3,
        "match_count": match_count
    }).execute()
    return response.data

def has_urdu_script(text):
    return bool(re.search(r'[\u0600-\u06FF]', text))

def generate_answer(question, chunks, history):
    context = "\n\n".join([c["content"] for c in chunks]) if chunks else "No relevant documents found."

    system_prompt = """You are a warm, friendly assistant talking to a close friend named Abbas. Speak naturally and casually — react genuinely, use his name occasionally. Detect the language of his question and respond with TWO versions separated by "|||SPEECH|||":

1. First version: warm, conversational answer in the SAME script he used (Roman Urdu stays Roman Urdu, English stays English, Urdu script stays Urdu script).
2. Second version: the exact same answer, but if it was Urdu or Roman Urdu, written in proper Urdu script for correct pronunciation. If English, repeat the same text.

Use the provided context if relevant, otherwise use your own general knowledge. Keep it natural and not robotic.

Format strictly as: <display answer>|||SPEECH|||<speech answer>"""

    user_prompt = f"""Context:
{context}

Question: {question}"""

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_prompt})

    response = groq_client.chat.completions.create(model="openai/gpt-oss-20b", messages=messages)
    raw = response.choices[0].message.content

    if "|||SPEECH|||" in raw:
        display_text, speech_text = raw.split("|||SPEECH|||", 1)
    else:
        display_text = speech_text = raw

    return display_text.strip(), speech_text.strip()

def text_to_speech_bytes(text):
    lang = "ur" if has_urdu_script(text) else "en"
    tts = gTTS(text=text, lang=lang)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf

def transcribe_audio(audio_bytes):
    audio_file = sr.AudioFile(io.BytesIO(audio_bytes))
    with audio_file as source:
        audio_data = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio_data, language="en-US")
    except (sr.UnknownValueError, sr.RequestError):
        return None

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def ingest_document(title, content):
    doc_response = supabase.table("documents").insert({"title": title, "content": content}).execute()
    document_id = doc_response.data[0]["id"]
    chunks = chunk_text(content)
    embeddings = list(embed_model.embed(chunks))
    for chunk, embedding in zip(chunks, embeddings):
        supabase.table("chunks").insert({
            "document_id": document_id,
            "content": chunk,
            "embedding": embedding.tolist()
        }).execute()
    return len(chunks)

# ---------- Multi-chat session state ----------
if "chats" not in st.session_state:
    first_id = str(uuid.uuid4())
    st.session_state.chats = {first_id: {"title": "New Chat", "messages": []}}
    st.session_state.active_chat = first_id

def new_chat():
    new_id = str(uuid.uuid4())
    st.session_state.chats[new_id] = {"title": "New Chat", "messages": []}
    st.session_state.active_chat = new_id

# ---------- Sidebar (New chat fully on top, then everything else below, one by one) ----------
with st.sidebar:
    st.markdown("### 🧠 Agentic RAG")

    if st.button("➕  New chat", use_container_width=True):
        new_chat()
        st.rerun()

    st.divider()

    for chat_id, chat_data in reversed(list(st.session_state.chats.items())):
        is_active = chat_id == st.session_state.active_chat
        prefix = "💬 " if is_active else "　 "
        if st.button(f"{prefix}{chat_data['title']}", key=f"chat_{chat_id}", use_container_width=True):
            st.session_state.active_chat = chat_id
            st.rerun()

# ---------- Main Chat Area ----------
active_chat = st.session_state.chats[st.session_state.active_chat]
messages = active_chat["messages"]

if not messages:
    st.markdown('<div class="welcome-text">What\'s on your mind today?</div>', unsafe_allow_html=True)
else:
    for msg in messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-msg"><div class="user-bubble">{msg["content"]}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-msg"><div class="assistant-text">{msg["content"]}</div></div>', unsafe_allow_html=True)
            if "audio" in msg:
                st.audio(msg["audio"], format="audio/mp3")

def handle_question(question, voice_reply=False):
    messages.append({"role": "user", "content": question})
    if active_chat["title"] == "New Chat":
        active_chat["title"] = question[:30] + ("..." if len(question) > 30 else "")

    with st.spinner("Thinking..."):
        chunks = retrieve_chunks(question)
        history = [{"role": m["role"], "content": m["content"]} for m in messages[:-1]]
        display_text, speech_text = generate_answer(question, chunks, history)

    audio_bytes = None
    if voice_reply:
        audio_buf = text_to_speech_bytes(speech_text)
        audio_bytes = audio_buf.read()

    assistant_msg = {"role": "assistant", "content": display_text}
    if audio_bytes:
        assistant_msg["audio"] = audio_bytes
    messages.append(assistant_msg)

# ---------- Bottom input row: [+ attach] [text input] [🎙️ voice call] ----------
st.markdown("<br>", unsafe_allow_html=True)

col_attach, col_input, col_voice = st.columns([1, 8, 1])

with col_attach:
    st.markdown('<div class="attach-btn">', unsafe_allow_html=True)
    with st.popover("➕"):
        st.caption("Add Knowledge")
        uploaded_file = st.file_uploader("Upload a text file", type=["txt"], label_visibility="collapsed")
        if uploaded_file is not None:
            if st.button("Ingest this file"):
                content = uploaded_file.read().decode("utf-8")
                with st.spinner("Processing document..."):
                    num_chunks = ingest_document(uploaded_file.name, content)
                st.success(f"Added {num_chunks} chunks")
    st.markdown('</div>', unsafe_allow_html=True)

with col_input:
    question = st.chat_input("Ask anything")

with col_voice:
    st.markdown('<div class="voice-call-btn">', unsafe_allow_html=True)
    with st.popover("🎙️"):
        st.caption("🎧 Voice mode — tap to record, I'll reply out loud")
        audio_input = st.audio_input("Record", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

if audio_input is not None:
    audio_bytes_raw = audio_input.getvalue()
    audio_hash = hash(audio_bytes_raw)
    if st.session_state.get("last_audio_hash") != audio_hash:
        st.session_state.last_audio_hash = audio_hash
        with st.spinner("Transcribing..."):
            transcribed = transcribe_audio(audio_bytes_raw)
        if transcribed:
            handle_question(transcribed, voice_reply=True)
            st.rerun()
        else:
            st.warning("Samajh nahi aaya, dobara try karein.")

if question:
    handle_question(question, voice_reply=False)
    st.rerun()