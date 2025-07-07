from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextGenerationPipeline,
)
from tqdm import tqdm

# ------------- CONFIGURACIÓN -----------------
MODEL_PATH = Path("./Modelos/llama2-7b")   # cámbialo para tu FT
BATCH_SIZE = 4                             # ajusta (≤ 6) según VRAM y prompt len
MAX_NEW_TOKENS = 256                       # tope de salida
ANALYSIS_WORKERS = 8                       # nº procesos SonarQube
INPUT_JSON = Path("input.json")
OUTPUT_JSON = Path("output.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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
def build_prompt(code: str, issues: List[Dict[str, Any]]) -> str:
    """
    Convierte el snippet + issues en un prompt para Llama-2.
    Modifica si tu fine-tune espera otro formato.
    """
    issue_summary = "\n".join(
        f"- (rule {i['rule']}) line {i['line']}: {i['message']}"
        for i in issues
    )
    prompt = (
        "### Instructions:\n"
        "You are an expert Java assistant. Rewrite the code to fix the "
        "problems listed below while preserving its original functionality.\n\n"
        "### Original code:\n"
        f"{code}\n\n"
        "### List of problems:\n"
        f"{issue_summary}\n\n"
        "### Corrected code:\n"
    )
    return prompt
# ---------------------------------------------

# ----  CARGA MODELO Y PREPARA PIPELINE -------
def load_pipeline(model_path: Path = MODEL_PATH) -> TextGenerationPipeline:
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    return TextGenerationPipeline(
        model=model,
        tokenizer=tokenizer,
        device=0 if DEVICE == "cuda" else -1,
        torch_dtype=model.dtype,
        batch_size=BATCH_SIZE,
    )
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

def analyze_file(tmpdir: Path = None) -> Tuple[List[Dict], str]:
    """
    Analiza un snippet y devuelve (issues, error_msg). Si error_msg != "", falló.
    Se ejecuta en un proceso hijo para permitir paralelismo verdadero.
    """
    project_key = f"snippet-{uuid.uuid4().hex[:12]}"

    try:

        # 1) escribir código Java y properties
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

    results: List[Dict[str, Any]] = []
    analysis_pool = ProcessPoolExecutor(max_workers=ANALYSIS_WORKERS)
    pending_futures: Dict[Any, Dict[str, Any]] = {}

    with tqdm(total=len(data), desc="Inferencia", leave=True) as pbar:
        for i in range(0, len(data), BATCH_SIZE):
            batch = data[i : i + BATCH_SIZE]
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


git commit -m "Estado actual del proyecto. \n Primer commit en remoto. \n\n El proyecto consiste en 7 scripts de python alguno de los cuale serán eliminados más adelante. \n Los scripts más importantes son Fine_tuning_LLama2_Mss4j.py que permite hacer fine-tuning de Llama2-7b con un dataset dado. \n El segundo es FT-validation.py, script en construcción cuyo objetivo es validar la mejora en la refactorización realizada por los distintos modelos. \n El tercero es batch_sonar.py el cual realiza una serie de Request en paralelo a SonarQube para analizar el código utilizado para la validación. \n Lo otros cuatro scripts son para descargar y procesar distintos datasets que se han probado o se están utilizando. El resto de los archivos json contienen datos procesados de distintos datasets, algunos no se usan y serán eliminados conforme avance el proyecto." 