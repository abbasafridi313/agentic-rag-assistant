import os
from dotenv import load_dotenv
from supabase import create_client
from fastembed import TextEmbedding
from groq import Groq

load_dotenv()

# Supabase connect
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase = create_client(url, key)

# Groq connect
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Embedding model load karo
print("Loading embedding model...")
model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Conversation history yahan store hogi (memory ke liye)
conversation_history = []

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

    system_prompt = """You are a helpful assistant. ALWAYS respond in Roman Urdu (Urdu written in English/Latin letters), regardless of what language the question is asked in. Use the provided context if relevant, otherwise use your own general knowledge. Keep the conversation history in mind for follow-up questions."""

    user_prompt = f"""Context:
{context}

Question: {question}

Answer in Roman Urdu:"""

    # Conversation history + naya sawal saath bhejo
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_prompt})

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages
    )
    answer = response.choices[0].message.content

    # History update karo (agla sawal iska context use karega)
    conversation_history.append({"role": "user", "content": question})
    conversation_history.append({"role": "assistant", "content": answer})

    return answer

def ask(question):
    chunks = retrieve_chunks(question)
    answer = generate_answer(question, chunks)
    print(f"\nJawab: {answer}\n")

if __name__ == "__main__":
    print("Agentic RAG Assistant — 'exit' likh kar band karo\n")
    while True:
        question = input("Aapka sawal: ")
        if question.lower() == "exit":
            break
        ask(question)