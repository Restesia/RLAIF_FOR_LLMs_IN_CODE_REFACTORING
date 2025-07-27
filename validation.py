from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import requests
import time
import uuid
import subprocess
import re
import tempfile
from typing import Optional, Dict, List, Tuple

from peft import PeftModel

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextGenerationPipeline,
)
from tqdm import tqdm

# ------------- CONFIGURACIÓN -----------------
MODEL_PATH = Path("./Modelos/llama2-7b")   # cámbialo para tu FT
BASE_MODEL   = Path("./Modelos/llama2-7b")             # pesos originales
#ADAPTER_PATH = Path("checkpoints/llama2-7b-mss4j") # STF
ADAPTER_PATH = Path("dpo-llama2-refactor") # STF
BATCH_SIZE = 4 # ajusta (≤ 6) según VRAM y prompt len
MAX_NEW_TOKENS = 256                       # tope de salida
ANALYSIS_WORKERS = 8                       # nº procesos SonarQube
INPUT_JSON = Path("java_validation_code.json")
OUTPUT_JSON = Path("RLAIF_veryfied.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_PROMPT_TOKENS = 1024     # ignora ejemplos más largos
SKIPPED_LOG = Path("skipped.txt")
# ---------------------------------------------


OUTPUT_PATH = "java_validation_code.json"

SONAR_HOST = "http://localhost:9000"
SONAR_TOKEN = "squ_7be58e5f386c00dbdfc8e1a3df15ad1934426e66"          # ← exporta antes de ejecutar
SCANNER_CMD = ["/home/david/sonar-scanner-7.1.0.4889-linux-x64/bin/sonar-scanner"]                # ← o ['docker','run','--rm', ...]
SCAN_TIMEOUT_SEC = 300                             # seg. máx. por snippet
POLL_INTERVAL   = 2                               # seg. entre chequeos CE
MAX_PARALLEL     = 4        # núm. de SonarScanners concurrentes
USE_DOCKER_SCANNER = False  # True ⇒ usa la imagen sonarsource/sonar-scanner-cli


# Ruta al binario local (solo si USE_DOCKER_SCANNER = False)
LOCAL_SCANNER_BIN = "/home/david/sonar-scanner-7.1.0.4889-linux-x64/bin/sonar-scanner"

# ---------  FUNCIÓN DE PROMPTING -------------
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
        if (ln := issue.get("line")):                 # línea concreta
            return f"line {ln}"
        tr = issue.get("textRange") or {}
        if tr.get("startLine") and tr.get("endLine"): # rango de líneas
            return f"lines {tr['startLine']}-{tr['endLine']}"
        return ""                                     # ámbito de archivo

    issue_text = "\n".join(
        f"- (rule {issue.get('rule', '?')}) "
        f"{short_location(issue)}: "
        f"{issue.get('message', '')}"
        for issue in issues
    )

    to_encode = f"{code}\n{issue_text}"
    return len(tokenizer(to_encode, add_special_tokens=False)["input_ids"])
# ---------------------------------------------

# ----  CARGA MODELO Y PREPARA PIPELINE -------
# def load_pipeline(model_path: Path = MODEL_PATH) -> TextGenerationPipeline:
#     tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
#
#     if tokenizer.pad_token_id is None:
#         tokenizer.pad_token = tokenizer.eos_token
#         tokenizer.pad_token_id = tokenizer.eos_token_id
#     model = AutoModelForCausalLM.from_pretrained(
#         model_path,
#         torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
#         device_map="auto",
#         low_cpu_mem_usage=True,
#     )
#     return TextGenerationPipeline(
#         model=model,
#         tokenizer=tokenizer,
#         torch_dtype=model.dtype,
#         batch_size=BATCH_SIZE,
#     )

def load_pipeline() -> TextGenerationPipeline:
    # Usa el tokenizer guardado junto al adapter,
    # así aprovecha los mismos tokens especiales que añadaste al FT.
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, use_fast=True)

    # Garantiza pad_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 1️⃣  carga los pesos base en GPU con Accelerate
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    # 2️⃣  le inyecta el adapter LoRA
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

    # (opcional) fusiona las matrices LoRA y libera memoria
    model = model.merge_and_unload()     # ↩️ quítalo si prefieres mantener el adapter separado

    # 3️⃣  crea la pipeline SIN el argumento `device=`
    return TextGenerationPipeline(model=model, tokenizer=tokenizer)
# ---------------------------------------------

# -------------  BATCH INFERENCE --------------
@torch.inference_mode()
def infer_batch(
    pipe: TextGenerationPipeline,
    snippets: List[Dict[str, Any]],
) -> List[str]:
    prompts = [build_prompt(s["content"], s["issues"]) for s in snippets]
    # transformers ≥ 4.40 devuelve list[dict]; compatibilidad con versiones previas
    raw_outputs = pipe(
        prompts,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        return_full_text=False,
    )
    # raw_outputs puede ser list[str] o list[list[dict]]
    if isinstance(raw_outputs[0], str):
        return raw_outputs
    return [out[0]["generated_text"] for out in raw_outputs]


# -------------  BATCH INFERENCE -------------- (Issue per issue)
@torch.inference_mode()
def infer_batch(
    pipe: TextGenerationPipeline,
    snippets: List[Dict[str, Any]],
) -> List[str]:
    
    prompts = []
    idxL = []
    code = [s["content"] for s in snippets]
    while all(len(s["issues"]) == 0 for s in snippets):
        for idx, s in snippets:
            if s["issues"] != []:
                prompts.append(build_prompt(code[idx], [s["issues"].pop(0)]))
                idxL.append(idx)

        raw_outputs = pipe(
        prompts,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        return_full_text=False,
        )
        if isinstance(raw_outputs[0], str):
            for idx2, s in raw_outputs:
                code[idxL[idx2]] = s
        else:
            filtred_raw_code = [out[0]["generated_text"] for out in raw_outputs]
            for idx2, s in filtred_raw_code:
                code[idxL[idx2]] = s

        prompts = []
        idxL = []

    return code
    
# ---------------------------------------------

def run(cmd: List[str], cwd: Path) -> None:
    """Ejecuta un sub-proceso y lanza excepción si sale con código ≠ 0"""
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(res.stderr or res.stdout)

def wait_for_ce(task_id: str) -> str:
    """Espera a que el Compute Engine termine y devuelve analysisId"""
    url  = f"{SONAR_HOST}/api/ce/task?id={task_id}"
    auth = (SONAR_TOKEN, "")
    t0   = time.time()
    while time.time() - t0 < SCAN_TIMEOUT_SEC:
        status = requests.get(url, auth=auth, timeout=10).json()["task"]
        if status["status"] in ("SUCCESS", "FAILED", "CANCELED"):
            if status["status"] != "SUCCESS":
                raise RuntimeError(f"CE task {task_id} acabó en {status['status']}")
            return status["analysisId"]
        time.sleep(POLL_INTERVAL)
    raise TimeoutError(f"CE task {task_id} excedió {SCAN_TIMEOUT_SEC}s")

def fetch_issues(project_key: str) -> List[Dict]:
    """Devuelve todas las issues abiertas del proyecto"""
    issues, page, ps = [], 1, 500
    auth = (SONAR_TOKEN, "")
    while True:
        url = (f"{SONAR_HOST}/api/issues/search"
               f"?componentKeys={project_key}&resolved=false&ps={ps}&p={page}")
        data = requests.get(url, auth=auth, timeout=15).json()
        issues.extend(data["issues"])
        if page * ps >= data["paging"]["total"]:
            break
        page += 1
    return issues

def nombre_archivo_java(src: str, fallback: str = "Main") -> str:
    """
    Devuelve el nombre que debería tener el archivo que contenga el código para compilar correctamente en java.
    :param src: Código
    :param fallback: Nombre por defecto en caso de no haber un ente Public en el nivel 0
    :return: nombre del archivo
    """

    # 1) quitar comentarios (línea y bloque)
    sin_coment = re.sub(r'//.*?$|/\*.*?\*/', '', src,
                        flags=re.MULTILINE | re.DOTALL)

    # 2) tokenizer muy simple: palabras, llaves, y cualquier otro char suelto
    token_pat = re.compile(r'\w+|[{}]', re.MULTILINE | re.DOTALL)

    depth = 0
    it = iter(token_pat.finditer(sin_coment))

    for tok in it:
        lex = tok.group(0)
        if lex == '{':
            depth += 1
            continue
        if lex == '}':
            depth = max(0, depth - 1)
            continue

        # Solo nos interesa lo que esté en el nivel superior
        if depth == 0 and lex == 'public':
            # colecciona modificadores intermedios (abstract, final, etc.)
            mods: list[str] = []
            while True:
                nxt: Optional[str] = next(it, None)
                if nxt is None:
                    return f"{fallback}.java"
                lex2 = nxt.group(0)
                if lex2 in {'class', 'interface', 'enum', 'record'}:
                    # El siguiente token debería ser el nombre
                    name_tok = next(it, None)
                    if name_tok and re.fullmatch(r'\w+', name_tok.group(0)):
                        return f"{name_tok.group(0)}.java"
                    break  # forma inesperada → usa fallback
                elif lex2 in {
                    'abstract', 'final', 'sealed', 'non-sealed', 'strictfp',
                    'static', '@interface'
                }:
                    mods.append(lex2)
                    continue
                else:
                    break  # algo raro; aborta y sigue buscando

    # 3) Sin public top-level → nombre libre
    return f"{fallback}.java"

def analyze_snippet(snippet: str) -> Tuple[List[Dict], str]:
    """
    Analiza un snippet y devuelve (issues, error_msg). Si error_msg != "", falló.
    Se ejecuta en un proceso hijo para permitir paralelismo verdadero.
    """
    project_key = f"snippet-{uuid.uuid4().hex[:12]}"

    try:
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)

            # 1) escribir código Java y properties
            (tmpdir / nombre_archivo_java(snippet)).write_text(snippet, encoding="utf-8")
            (tmpdir / "sonar-project.properties").write_text(
                "\n".join([
                    f"sonar.projectKey={project_key}",
                    "sonar.projectName=Batch Snippet",
                    "sonar.sources=.",
                    "sonar.language=java",
                    "sonar.sourceEncoding=UTF-8",
                ]),
                encoding="utf-8",
            )

            # 2) comando scanner
            if USE_DOCKER_SCANNER:
                cmd = [
                    "docker", "run", "--rm",
                    "-e", f"SONAR_HOST_URL={SONAR_HOST}",
                    "-e", f"SONAR_TOKEN={SONAR_TOKEN}",
                    "-v", f"{tmpdir}:/usr/src",
                    "sonarsource/sonar-scanner-cli:7.1.0.4889",
                    "-Dsonar.projectKey=" + project_key,
                    "-Dsonar.sources=/usr/src",
                ]
            else:
                cmd = [
                    LOCAL_SCANNER_BIN,
                    f"-Dsonar.host.url={SONAR_HOST}",
                    f"-Dsonar.login={SONAR_TOKEN}",
                ]

            # 3) lanzar análisis
            run(cmd, cwd=tmpdir)

            # 4) leer report-task.txt
            task_file = tmpdir / ".scannerwork" / "report-task.txt"
            ce_task_id = next(
                l.split("=", 1)[1] for l in task_file.read_text().splitlines()
                if l.startswith("ceTaskId=")
            )

            # 5) esperar CE y obtener issues
            wait_for_ce(ce_task_id)
            issues = fetch_issues(project_key)
            return issues, ""   # sin error

    except Exception as exc:
        return [], str(exc)


def main() -> None:
    data: List[Dict[str, Any]] = json.loads(INPUT_JSON.read_text(encoding="utf-8"))
    pipe = load_pipeline()
    tokenizer = pipe.tokenizer

    keep, skip = [], []
    for snippet in data:
        tks = prompt_token_length(
            tokenizer, snippet["content"], snippet["issues"]
        )
        if tks <= MAX_PROMPT_TOKENS:
            keep.append(snippet)
        else:
            skip.append((snippet["path"] or "<in-memory>", tks))

    # Guarda una lista de los ejemplos ignorados (opcional)
    if skip:
        SKIPPED_LOG.write_text(
            "\n".join(f"{p}\t{t} tokens" for p, t in skip), encoding="utf-8"
        )
        print(f"Ignorados {len(skip)} snippets (> {MAX_PROMPT_TOKENS} tks). "
              f"Detalle en {SKIPPED_LOG}")

    results: List[Dict[str, Any]] = []
    analysis_pool = ProcessPoolExecutor(max_workers=ANALYSIS_WORKERS)
    pending_futures: Dict[Any, Dict[str, Any]] = {}

    with tqdm(total=len(keep), desc="Inferencia", leave=True) as pbar:
        for i in range(0, len(keep), BATCH_SIZE):
            batch = keep[i : i + BATCH_SIZE]
            generated = infer_batch(pipe, batch)

            # Lanza análisis SonarQube en paralelo
            for snippet, fixed_code in zip(batch, generated):
                fut = analysis_pool.submit(analyze_snippet, fixed_code)
                pending_futures[fut] = {
                    "code_before": snippet["content"],
                    "issues_before": snippet["issues"],
                    "code_after": fixed_code,
                }
            pbar.update(len(batch))

    # Recolecta resultados de SonarQube
    for fut in tqdm(
        as_completed(pending_futures),
        total=len(pending_futures),
        desc="Análisis SonarQube",
        leave=True,
    ):
        rec = pending_futures.pop(fut)
        rec["issues_after"] = fut.result()
        results.append(rec)

    analysis_pool.shutdown(wait=True)

    OUTPUT_JSON.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n✔ Resultado guardado en {OUTPUT_JSON.resolve()}\n")


if __name__ == "__main__":
    main()


