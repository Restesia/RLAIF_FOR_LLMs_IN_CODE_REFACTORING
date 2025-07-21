import os
import json
import asyncio
import random
from pathlib import Path
from tqdm.auto import tqdm
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextGenerationPipeline,
)
import aiohttp
from peft import PeftModel
from aiohttp import (
    ClientPayloadError,
    ClientConnectionError,
    ClientResponseError,
)

load_dotenv()

# ------------- CONFIGURACIÓN -----------------
BASE_MODEL    = Path("./Modelos/llama2-7b")   # pesos originales
ADAPTER_PATH  = Path("checkpoints/llama2-7b-mss4j")
INPUT_JSON    = Path("RLAIF_estimation_code.json")
PROGRESS_PATH   = Path("RLAIF_DPO_dset_progress.json")
OUTPUT_JSONL = Path("RLAIF_DPO_dset.json")
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE        = 4
NUM_CANDIDATES    = 3
MAX_NEW_TOKENS    = 256
TOP_P             = 0.95
TEMPERATURE       = 0.7
MAX_PROMPT_TOKENS = 1024
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"

DEEPSEEK_API_KEY  = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise RuntimeError("DEEPSEEK_API_KEY no definida (añádela a .env)")
DEEPSEEK_API_URL  = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL   = "deepseek-chat"  # modelo con mejor ratio coste/calidad
MAX_CONCURRENT    = 8  # controlar saturación
MAX_RETRIES       = 3
RETRY_BACKOFF_S   = 2

# ------------------------------------------------
# Model pipeline
# ------------------------------------------------

def load_pipeline() -> TextGenerationPipeline:
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model = model.merge_and_unload()
    return TextGenerationPipeline(model=model, tokenizer=tokenizer)

# ------------------------------------------------
# Candidate generation
# ------------------------------------------------

@torch.no_grad()
def generate_candidates(pipeline: TextGenerationPipeline, prompts: List[str]) -> List[List[str]]:
    raw_out = pipeline(
        prompts,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        top_p=TOP_P,
        temperature=TEMPERATURE,
        num_return_sequences=NUM_CANDIDATES,
        return_full_text=False,
        clean_up_tokenization_spaces=True,
    )
    if isinstance(raw_out[0], list):
        return [[(c.get("generated_text", c)).strip() for c in sub] for sub in raw_out]
    grouped, idx = [], 0
    for _ in prompts:
        grouped.append([(c.get("generated_text", c)).strip() for c in raw_out[idx: idx + NUM_CANDIDATES]])
        idx += NUM_CANDIDATES
    return grouped

# ------------------------------------------------
# DeepSeek scoring (con reintentos)
# ------------------------------------------------

async def _call_deepseek(session: aiohttp.ClientSession, payload: Dict[str, Any]) -> str:
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    async with session.post(DEEPSEEK_API_URL, json=payload, headers=headers, timeout=120) as resp:
        if resp.status != 200:
            text = await resp.text()
            raise ClientResponseError(resp.request_info, resp.history, status=resp.status, message=text)
        raw = await resp.text()
        for line in raw.splitlines():
            if line.startswith("data: "):
                data_str = line[len("data: ") :].strip()
                if data_str == "[DONE]":
                    break
                return data_str
        raise ValueError("Respuesta DeepSeek sin línea data")

async def score_candidate(session: aiohttp.ClientSession, original_prompt: str, candidate: str) -> float:
    system_msg = (
        "Eres un revisor sénior de código Java. Devuelve JSON {\"score\": N} (1-10) evaluando la refactorización."  # noqa: E501
    )
    payload = {
        "model": DEEPSEEK_MODEL,
        "response_format": {"type": "json_object"},
        "stream": True,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"### Código original\n{original_prompt}\n\n### Refactorización\n{candidate}"},
        ],
        "temperature": 0.0,
        "max_tokens": 32,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            data_str = await _call_deepseek(session, payload)
            return float(json.loads(data_str).get("score", 0))
        except (
            ClientPayloadError,
            ClientConnectionError,
            ClientResponseError,
            asyncio.TimeoutError,
            json.JSONDecodeError,
            ValueError,
        ):
            if attempt == MAX_RETRIES:
                return 0.0
            await asyncio.sleep(RETRY_BACKOFF_S * attempt + random.uniform(0, 1))

async def score_batch_async(prompts: List[str], candidates: List[List[str]]) -> List[float]:
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    async with aiohttp.ClientSession() as session:
        tasks = []
        for pr, cand_list in zip(prompts, candidates):
            for cd in cand_list:
                async def sem_task(p=pr, c=cd):
                    async with semaphore:
                        return await score_candidate(session, p, c)
                tasks.append(asyncio.create_task(sem_task()))
        return await asyncio.gather(*tasks)

# ------------------------------------------------
# Prompt helpers
# ------------------------------------------------
def build_prompt(code: str, issues: list[dict]) -> str:
    def format_issue(i: dict) -> str:
        # 1️⃣ Intenta 'line'
        if "line" in i and i["line"] is not None:
            location = f"line {i['line']}"
            # 2️⃣ Si no, usa textRange (start-end)
        elif "textRange" in i and i["textRange"]:
            tr = i["textRange"]
            location = f"lines {tr['startLine']}-{tr['endLine']}"
        else:
            location = "(file level)"  # 3️⃣ Sin referencia de línea

        return f"- (rule {i.get('rule', '?')}) {location}: {i.get('message', '')}"

    issue_summary = "\n".join(format_issue(i) for i in issues)

    prompt = (
        "### Instructions:\n"
        "You are an expert Java assistant. Rewrite the code to fix the "
        "problems listed below while preserving its original functionality. Returns only the corrected code.\n\n"
        "### Original code:\n"
        f"{code}\n\n"
        "### List of problems:\n"
        f"{issue_summary}\n\n"
        "### Corrected code:\n"
    )
    return prompt


def prompt_token_length(tokenizer, code: str, issues: list[dict]) -> int:
    """
    Devuelve cuántos tokens ocupará el prompt sin generarlo entero.
    Se maneja que 'line' o 'textRange' puedan faltar.
    """

    def short_location(issue: dict) -> str:
        if (ln := issue.get("line")):  # línea concreta
            return f"line {ln}"
        tr = issue.get("textRange") or {}
        if tr.get("startLine") and tr.get(
                "endLine"):  # rango de líneas
            return f"lines {tr['startLine']}-{tr['endLine']}"
        return ""  # ámbito de archivo

    issue_text = "\n".join(
        f"- (rule {issue.get('rule', '?')}) "
        f"{short_location(issue)}: "
        f"{issue.get('message', '')}"
        for issue in issues
    )

    to_encode = f"{code}\n{issue_text}"
    return len(tokenizer(to_encode, add_special_tokens=False)["input_ids"])

# ------------------------------------------------
# Utilidades de checkpoint
# ------------------------------------------------

def load_progress() -> int:
    if PROGRESS_PATH.exists():
        try:
            return json.load(PROGRESS_PATH.open())['processed']
        except Exception:
            return 0
    return 0

def save_progress(processed: int):
    with PROGRESS_PATH.open("w") as f:
        json.dump({"processed": processed}, f)

# ------------------------------------------------
# MAIN (con reanudación)
# ------------------------------------------------

def main():
    pipeline = load_pipeline()
    tokenizer = pipeline.tokenizer

    examples: List[Dict[str, Any]] = json.load(INPUT_JSON.open("r", encoding="utf-8"))
    keep = [ex for ex in examples if prompt_token_length(tokenizer, ex["content"], ex["issues"]) <= MAX_PROMPT_TOKENS]
    print(len(keep))
    start_idx = load_progress()
    if start_idx > 0:
        print(f"🔄 Reanudando desde el índice {start_idx} …")
    remaining = keep[(start_idx*BATCH_SIZE):]

    print(len(remaining), start_idx)
    dataloader = DataLoader(remaining, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: x)

    # Abrimos archivo de salida en append; si existe continuamos
    out_mode = "a" if OUTPUT_JSONL.exists() else "w"
    out_f = OUTPUT_JSONL.open(out_mode, encoding="utf-8")

    processed_global = start_idx

    for batch in tqdm(dataloader, desc="Generando y puntuando", total=len(dataloader)):
        prompts = [build_prompt(it["content"], it["issues"]) for it in batch]
        candidates = generate_candidates(pipeline, prompts)
        scores = asyncio.run(score_batch_async(prompts, candidates))

        idx = 0
        for p, cands in zip(prompts, candidates):
            sc_slice = scores[idx: idx + NUM_CANDIDATES]
            idx += NUM_CANDIDATES
            ranked = sorted(zip(cands, sc_slice), key=lambda x: x[1], reverse=True)
            best, _ = ranked[0]
            for worst, _ in ranked[1:]:
                pair = {"prompt": p, "better": best, "worse": worst}
                out_f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        processed_global += 1
        save_progress(processed_global)

    out_f.close()
    print(f"✅ Proceso completado. Pairs totales: {processed_global}")
    PROGRESS_PATH.unlink(missing_ok=True)  # borra progreso al acabar


if __name__ == "__main__":
    main()

