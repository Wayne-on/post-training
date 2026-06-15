from __future__ import annotations

import os
import time
import uuid
from typing import Any

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM

from post_training.common import build_quantization_config, load_tokenizer, torch_dtype


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = Field(default=512, ge=1)
    temperature: float = Field(default=0.7, ge=0.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    do_sample: bool | None = None


MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
TRUST_REMOTE_CODE = os.getenv("TRUST_REMOTE_CODE", "true").lower() == "true"
ATTN_IMPLEMENTATION = os.getenv("ATTN_IMPLEMENTATION", "sdpa")
TORCH_DTYPE = os.getenv("TORCH_DTYPE", "float16")
LOAD_IN_4BIT = os.getenv("LOAD_IN_4BIT", "false").lower() == "true"
LOAD_IN_8BIT = os.getenv("LOAD_IN_8BIT", "false").lower() == "true"

app = FastAPI(title="Local OpenAI-compatible LLM service")

tokenizer = load_tokenizer(MODEL_ID, TRUST_REMOTE_CODE)
quantization_config = build_quantization_config(
    {
        "load_in_4bit": LOAD_IN_4BIT,
        "load_in_8bit": LOAD_IN_8BIT,
        "bnb_4bit_quant_type": os.getenv("BNB_4BIT_QUANT_TYPE", "nf4"),
        "bnb_4bit_use_double_quant": os.getenv("BNB_4BIT_USE_DOUBLE_QUANT", "true").lower() == "true",
        "bnb_4bit_compute_dtype": os.getenv("BNB_4BIT_COMPUTE_DTYPE", "float16"),
    }
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=TRUST_REMOTE_CODE,
    torch_dtype=torch_dtype(TORCH_DTYPE),
    attn_implementation=ATTN_IMPLEMENTATION,
    quantization_config=quantization_config,
    device_map="auto",
)
model.eval()


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model": MODEL_ID}


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest) -> dict[str, Any]:
    messages = [message.model_dump() for message in request.messages]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    do_sample = request.do_sample if request.do_sample is not None else request.temperature > 0
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=max(request.temperature, 1e-5),
            top_p=request.top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    completion_ids = generated[0][inputs.input_ids.shape[-1] :]
    content = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    created = int(time.time())
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": created,
        "model": request.model or MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": int(inputs.input_ids.shape[-1]),
            "completion_tokens": int(completion_ids.shape[-1]),
            "total_tokens": int(inputs.input_ids.shape[-1] + completion_ids.shape[-1]),
        },
    }
