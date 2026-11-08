import uuid

import requests
from fastapi import FastAPI
from fastapi.responses import JSONResponse

OVMS_URL = "http://localhost:8000/v3/chat/completions"

app = FastAPI()


@app.post("/v1/chat/completions")
def chat(body: dict):
    # pass Docling's request straight through to OVMS
    r = requests.post(OVMS_URL, json=body, timeout=600)
    data = r.json()

    # OVMS leaves out "id", but Docling insists on it, so add one
    if isinstance(data, dict) and "choices" in data and "id" not in data:
        data["id"] = "chatcmpl-" + uuid.uuid4().hex

    return JSONResponse(content=data, status_code=r.status_code)