
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Load the model ONCE when the server starts (fast for every request after).
chat = pipeline("text-generation", model="merged_model")

# The shape of the incoming request: {"question": "..."}
class Question(BaseModel):
    question: str

@app.post("/generate")
def generate(item: Question):
    messages = [{"role": "user", "content": item.question}]
    out = chat(messages, max_new_tokens=50)
    answer = out[0]["generated_text"][-1]["content"]
    return {"answer": answer}
