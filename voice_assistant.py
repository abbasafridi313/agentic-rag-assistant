import os
import re
from dotenv import load_dotenv
from supabase import create_client
from fastembed import TextEmbedding
from groq import Groq
import speech_recognition as sr
from gtts import gTTS
import pygame

load_dotenv()

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase = create_client(url, key)

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

print("Loading embedding model...")
model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

recognizer = sr.Recognizer()
pygame.mixer.init()
conversation_history = []

def has_urdu_script(text):
    return bool(re.search(r'[\u0600-\u06FF]', text))

def speak(text):
    lang = "ur" if has_urdu_script(text) else "en"
    tts = gTTS(text=text, lang=lang)
    tts.save("response.mp3")
    pygame.mixer.music.load("response.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.music.unload()
    os.remove("response.mp3")

def listen():
    with sr.Microphone() as source:
        print("\nSun raha hoon... (bolo)")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        audio = recognizer.listen(source)
    try:
        print("Samajh raha hoon...")
        text = recognizer.recognize_google(audio, language="en-US")
        print(f"Aapne kaha: {text}")
        return text
    except sr.UnknownValueError:
        print("Maaf kijiye, samajh nahi aaya. Dobara try karein.")
        return None
    except sr.RequestError:
        print("Internet connection check karein.")
        return None

def retrieve_chunks(question, match_count=3):
    question_embedding = list(model.embed([question]))[0].tolist()
    response = supabase.rpc("match_chunks", {
        "query_embedding": question_embedding,
        "match_threshold": 0.3,
        "match_count": match_count
    }).execute()
    return response.data

def generate_answer(question, chunks):
    context = "\n\n".join([c["content"] for c in chunks]) if chunks else "No relevant documents found."

    system_prompt = """You are a helpful assistant. Detect the language of the user's question (English, Roman Urdu, or Urdu script), then respond with TWO versions separated by "|||SPEECH|||":

1. First version (before the separator): answer in the SAME script the user used (Roman Urdu stays Roman Urdu, English stays English, Urdu script stays Urdu script).
2. Second version (after the separator): the exact same answer, but if the question's language was Urdu or Roman Urdu, write it in proper Urdu script (اردو رسم الخط) for correct pronunciation. If the question was in English, just repeat the same English text.

Use the provided context if relevant, otherwise use your own general knowledge. Keep answers clear and conversational.

Format strictly as: <display answer>|||SPEECH|||<speech answer>"""

    user_prompt = f"""Context:
{context}

Question: {question}"""

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_prompt})

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages
    )
    raw = response.choices[0].message.content

    if "|||SPEECH|||" in raw:
        display_text, speech_text = raw.split("|||SPEECH|||", 1)
    else:
        display_text = raw
        speech_text = raw

    display_text = display_text.strip()
    speech_text = speech_text.strip()

    conversation_history.append({"role": "user", "content": question})
    conversation_history.append({"role": "assistant", "content": display_text})

    return display_text, speech_text

def ask(question, via_voice):
    chunks = retrieve_chunks(question)
    display_text, speech_text = generate_answer(question, chunks)
    print(f"\nJawab: {display_text}\n")
    if via_voice:
        speak(speech_text)

if __name__ == "__main__":
    print("Assistant ready!")
    print("- Bolne ke liye: bas Enter dabao (kuch likhe baghair)")
    print("- Type karne ke liye: apna sawal likh kar Enter dabao")
    print("- Band karne ke liye: 'q' likho\n")

    while True:
        user_input = input("Sawal (ya khali chhod kar bolo): ")
        if user_input.lower() == "q":
            break
        if user_input.strip() == "":
            question = listen()
            if question:
                ask(question, via_voice=True)
        else:
            ask(user_input, via_voice=False)