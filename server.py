import os
import re
import uuid
import logging
import io
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from dotenv import load_dotenv
from supabase import create_client
from fastembed import TextEmbedding
from groq import Groq
from gtts import gTTS
import base64
from pypdf import PdfReader
from docx import Document

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agentic-rag")

app = FastAPI()

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
if not url or not key:
    raise RuntimeError("SUPABASE_URL or SUPABASE_KEY missing in .env file")
supabase = create_client(url, key)

groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    raise RuntimeError("GROQ_API_KEY missing in .env file")
groq_client = Groq(api_key=groq_api_key)

print("Loading embedding model...")
embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
print("Ready!")

def has_urdu_script(text):
    return bool(re.search(r'[\u0600-\u06FF]', text))

def retrieve_chunks(question, match_count=3):
    try:
        question_embedding = list(embed_model.embed([question]))[0].tolist()
        response = supabase.rpc("match_chunks", {
            "query_embedding": question_embedding,
            "match_threshold": 0.3,
            "match_count": match_count
        }).execute()
        return response.data
    except Exception as e:
        logger.error(f"retrieve_chunks failed: {e}")
        return []

def generate_answer(question, chunks, history):
    context = "\n\n".join([c["content"] for c in chunks]) if chunks else "No relevant documents found."

    system_prompt = """You are a warm, friendly assistant talking to a close friend named Abbas. Speak naturally and casually — react genuinely.

Do NOT start every reply with his name or a greeting like "Hey Abbas". Only greet him by name if he greets you first (e.g. "hi", "hello", "kya haal hai" as an opener). For all other questions, just answer directly and naturally, the way a friend would mid-conversation — no repeated greetings.

CRITICAL RULE: Detect the exact language AND script the user used in their question, and respond ONLY in that same language and script:
- If he wrote in Roman Urdu (Urdu words spelled with English letters, e.g. "kya haal hai"), reply ONLY in Roman Urdu. Do not switch to Urdu script.
- If he wrote in English, reply ONLY in English.
- If he wrote in native Urdu script (اردو), reply ONLY in native Urdu script.
- If he wrote in any other language, reply in that same language.

Use the provided context if relevant, otherwise use your own general knowledge. Keep it natural, warm, and not robotic. Return ONLY the answer text — no labels, no extra formatting, no separators."""

    user_prompt = f"""Context:
{context}

Question: {question}"""

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_prompt})

    try:
        response = groq_client.chat.completions.create(model="openai/gpt-oss-20b", messages=messages)
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Groq API failed: {e}")
        return "Maaf kijiye, is waqt jawab generate nahi kar pa raha. Dobara try karein."

def get_speech_version(display_text):
    if has_urdu_script(display_text):
        return display_text
    try:
        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{
                "role": "user",
                "content": f"""If the following text is written in Roman Urdu (Urdu words spelled with English letters), transliterate it into native Urdu script (اردو رسم الخط), preserving the exact meaning. If it's already in English, return it completely unchanged. Return ONLY the resulting text, nothing else, no explanation.

Text: {display_text}"""
            }]
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return display_text

def text_to_speech_base64(text):
    try:
        lang = "ur" if has_urdu_script(text) else "en"
        tts = gTTS(text=text, lang=lang)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    except Exception as e:
        logger.error(f"TTS failed: {e}")
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

def get_chat_history(chat_id):
    response = supabase.table("chat_messages").select("role, content").eq("chat_id", chat_id).order("created_at").execute()
    return response.data

def save_message(chat_id, role, content):
    supabase.table("chat_messages").insert({"chat_id": chat_id, "role": role, "content": content}).execute()

def extract_text_from_file(filename, raw_bytes):
    """Filename ke extension ke hisaab se text nikalta hai. None return karta hai agar unsupported ho."""
    lower_name = filename.lower()

    if lower_name.endswith(".txt"):
        for encoding in ["utf-8", "utf-16", "windows-1252", "latin-1"]:
            try:
                return raw_bytes.decode(encoding)
            except (UnicodeDecodeError, UnicodeError):
                continue
        return None

    if lower_name.endswith(".pdf"):
        try:
            reader = PdfReader(io.BytesIO(raw_bytes))
            text_parts = [page.extract_text() or "" for page in reader.pages]
            return "\n\n".join(text_parts)
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            return None

    if lower_name.endswith(".docx"):
        try:
            doc = Document(io.BytesIO(raw_bytes))
            return "\n\n".join(p.text for p in doc.paragraphs)
        except Exception as e:
            logger.error(f"DOCX extraction failed: {e}")
            return None

    return None

# ---------- API Routes ----------

@app.get("/")
def serve_index():
    return FileResponse("static/index.html")

@app.post("/api/chat")
async def chat(chat_id: str = Form(...), message: str = Form(...), want_voice: bool = Form(False)):
    if not message.strip():
        return JSONResponse({"answer": "Kuch to likho ya bolo!", "audio_base64": None})

    try:
        history = get_chat_history(chat_id)
        chunks = retrieve_chunks(message)
        display_text = generate_answer(message, chunks, history)

        save_message(chat_id, "user", message)
        save_message(chat_id, "assistant", display_text)

        chat_row = supabase.table("user_chats").select("title").eq("id", chat_id).execute()
        if chat_row.data and chat_row.data[0]["title"] == "New Chat":
            new_title = message[:30] + ("..." if len(message) > 30 else "")
            supabase.table("user_chats").update({"title": new_title}).eq("id", chat_id).execute()

        audio_b64 = None
        if want_voice:
            speech_text = get_speech_version(display_text)
            audio_b64 = text_to_speech_base64(speech_text)

        return JSONResponse({"answer": display_text, "audio_base64": audio_b64})
    except Exception as e:
        logger.error(f"/api/chat failed: {e}")
        return JSONResponse({
            "answer": "Kuch masla ho gaya hai, thori dair mein dobara try karein.",
            "audio_base64": None
        }, status_code=500)

@app.post("/api/new_chat")
def new_chat(anon_id: str = Form(...)):
    response = supabase.table("user_chats").insert({"anon_id": anon_id, "title": "New Chat"}).execute()
    return {"chat_id": response.data[0]["id"]}

@app.get("/api/chats")
def list_chats(anon_id: str):
    response = supabase.table("user_chats").select("id, title, created_at").eq("anon_id", anon_id).order("created_at", desc=True).execute()
    return {"chats": response.data}

@app.get("/api/chats/{chat_id}/messages")
def get_messages(chat_id: str):
    return {"messages": get_chat_history(chat_id)}

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        allowed_extensions = (".txt", ".pdf", ".docx")
        if not file.filename.lower().endswith(allowed_extensions):
            return JSONResponse(
                {"error": "Sirf .txt, .pdf, ya .docx files allowed hain."},
                status_code=400
            )

        raw_bytes = await file.read()
        content = extract_text_from_file(file.filename, raw_bytes)

        if content is None:
            return JSONResponse({"error": "File padhi nahi ja saki."}, status_code=400)
        if not content.strip():
            return JSONResponse({"error": "File mein koi text nahi mila."}, status_code=400)

        num_chunks = ingest_document(file.filename, content)
        return {"chunks_added": num_chunks}
    except Exception as e:
        logger.error(f"/api/upload failed: {e}")
        return JSONResponse({"error": "File process nahi ho saki."}, status_code=500)

@app.get("/api/documents")
def list_documents():
    try:
        response = supabase.table("documents").select("id, title, created_at").order("created_at", desc=True).execute()
        docs = response.data
        for doc in docs:
            chunk_count = supabase.table("chunks").select("id", count="exact").eq("document_id", doc["id"]).execute()
            doc["chunk_count"] = chunk_count.count
        return {"documents": docs}
    except Exception as e:
        logger.error(f"/api/documents failed: {e}")
        return JSONResponse({"documents": [], "error": "Documents load nahi ho sake."}, status_code=500)

@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    try:
        supabase.table("chunks").delete().eq("document_id", doc_id).execute()
        supabase.table("documents").delete().eq("id", doc_id).execute()
        return {"deleted": True}
    except Exception as e:
        logger.error(f"delete_document failed: {e}")
        return JSONResponse({"deleted": False}, status_code=500)

app.mount("/static", StaticFiles(directory="static"), name="static")