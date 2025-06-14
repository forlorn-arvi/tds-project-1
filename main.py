from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from fastapi.responses import JSONResponse
import os
import requests
import json
import numpy as np
import base64
from PIL import Image
import pytesseract
import io
import re
# Constants
EMBEDDING_URL = "https://aipipe.org/openai/v1/embeddings"
COMPLETION_URL = "https://aipipe.org/openai/v1/chat/completions"
EMBEDDED_JSON_PATH = "embedded_posts.json"
SIMILARITY_THRESHOLD = 0.50

AIPIPE_API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjI0ZjIwMDg1NTZAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.szxtYl8jv9LRnO-7DiIUCL0V-Zwg3vhf_OhbYS1XP-4"

HEADERS = {
    "Authorization": f"Bearer {AIPIPE_API_KEY}",
    "Content-Type": "application/json"
}

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str
    image: Optional[str] = None

class Link(BaseModel):
    url: str
    text: str

class QuestionResponse(BaseModel):
    answer: str
    links: List[Link]

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def extract_text_from_image(base64_string):
    try:
        image_bytes = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception:
        return ""

def call_embedding_service(text):
    payload = {
        "model": "text-embedding-3-small",
        "input": [text]
    }
    r = requests.post(EMBEDDING_URL, headers=HEADERS, json=payload)
    r.raise_for_status()
    return r.json()['data'][0]['embedding']

def call_completion_service(question_combined, context):
    prompt = f"Student query: {question_combined}\n\nRelevant reference from forum: {context}\n\nBased on the above, respond clearly and helpfully."
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant for IITM TDS students."},
            {"role": "user", "content": prompt}
        ]
    }
    r = requests.post(COMPLETION_URL, headers=HEADERS, json=payload)
    r.raise_for_status()
    return r.json()['choices'][0]['message']['content']
def find_top_k_posts(question_embedding, k=3):
    with open(EMBEDDED_JSON_PATH, 'r', encoding='utf-8') as f:
        posts = json.load(f)

    scored_posts = []
    for post in posts:
        if 'embedding' not in post or not post['embedding']:
            continue
        score = cosine_similarity(question_embedding, post['embedding'])
        scored_posts.append((post, score))

    scored_posts.sort(key=lambda x: x[1], reverse=True)
    return scored_posts[:k]


@app.post("/api/", response_model=QuestionResponse)
async def handle_question(data: QuestionRequest):
    try:
        def generate_slug(title: str) -> str:
            slug = title.lower()
            slug = re.sub(r'[^a-z0-9]+', '-', slug)
            slug = re.sub(r'^-|-$', '', slug)
            return slug

        combined_text = data.question
        if data.image:
            image_text = extract_text_from_image(data.image)
            combined_text += "\n" + image_text

        question_embedding = call_embedding_service(combined_text)
        top_posts = find_top_k_posts(question_embedding, k=3)

        # Filter top_posts by similarity threshold
        top_posts = [(post, score) for post, score in top_posts if score >= SIMILARITY_THRESHOLD]

        if not top_posts:
            return JSONResponse(status_code=404, content={"error": "No relevant posts found."})

        combined_context = "\n---\n".join(post['cleaned_text'] for post, _ in top_posts)
        final_answer = call_completion_service(combined_text, combined_context)

        return {
            "answer": final_answer,
            "links": [
                {
                    "url": f"https://discourse.onlinedegree.iitm.ac.in/t/{generate_slug(post['topic_title'])}/{post['post_number']}",
                    "text": post['topic_title']
                } for post, _ in top_posts
            ]
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
