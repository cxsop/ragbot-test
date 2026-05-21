import os
import json
import logging
from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv
from db import search_chunks
from llm import get_llm_response, LLM_PROVIDERS

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "bond-rag-secret-2024")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a knowledgeable and empathetic Customer Support Assistant for an online bond brokerage platform in India.

Your role is to help support agents answer customer queries clearly and accurately about bond taxation — including TDS, capital gains, interest income, Form 26AS, ITR filing, and related topics.

Guidelines:
- Be empathetic, polite, and professional at all times.
- Use simple Indian English that is easy to understand.
- Answer ONLY based on the provided context chunks. Do NOT hallucinate or make up information.
- If the context does not have enough information, clearly say: "I'm sorry, I don't have enough information on this topic in my knowledge base. Please escalate to the tax team for further assistance."
- Format your reply in clean Markdown: use headings, bullet points, and bold text where needed.
- Always end with a helpful note or next step for the support agent.
- Keep the tone warm and reassuring — the customer may be anxious about taxes.
"""

@app.route("/")
def index():
    return render_template("index.html", providers=LLM_PROVIDERS)

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    query = data.get("query", "").strip()
    provider = data.get("provider", "ollama")
    history = data.get("history", [])

    if not query:
        return jsonify({"error": "Query cannot be empty."}), 400

    # Step 1: Retrieve relevant chunks from Postgres
    try:
        chunks = search_chunks(query, top_k=4)
    except Exception as e:
        logger.error(f"DB search error: {e}")
        return jsonify({"error": f"Database error: {str(e)}"}), 500

    if not chunks:
        context_text = "No relevant information found in the knowledge base."
    else:
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"### [{i}] {chunk['title']}\n"
                f"**Source:** {chunk['source_file']} | **Tags:** {', '.join(chunk.get('tags', []))}\n\n"
                f"{chunk['content']}"
            )
        context_text = "\n\n---\n\n".join(context_parts)

    # Step 2: Build the prompt
    user_prompt = f"""## Customer Support Query

**Question from Support Agent:** {query}

---

## Relevant Knowledge Base Context

{context_text}

---

## Instructions

Based ONLY on the context above, please provide a clear, accurate, and empathetic reply that the support agent can use to respond to the customer.
Format your answer in clean Markdown."""

    # Step 3: Call the LLM
    try:
        answer = get_llm_response(
            provider=provider,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            history=history
        )
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return jsonify({"error": f"AI model error: {str(e)}"}), 500

    # Step 4: Return answer + source chunks
    sources = [
        {
            "title": c["title"],
            "source_file": c["source_file"],
            "tags": c.get("tags", []),
            "chunk_id": c["chunk_id"]
        }
        for c in chunks
    ]

    return jsonify({
        "answer": answer,
        "sources": sources,
        "provider": provider
    })


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint"""
    try:
        chunks = search_chunks("test", top_k=1)
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"

    return jsonify({
        "status": "ok",
        "db": db_status,
        "providers": list(LLM_PROVIDERS.keys())
    })


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=1707)
