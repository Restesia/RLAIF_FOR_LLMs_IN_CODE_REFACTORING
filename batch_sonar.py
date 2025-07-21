#!/usr/bin/env python3
"""
batch_sonar.py  – Analiza en masa snippets Java con SonarQube
"""

import json
import os
import subprocess
import tempfile
import time
import re
from typing import Optional
import uuid
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import requests

# ---------------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------------
SONAR_HOST = "http://localhost:9000"
SONAR_TOKEN = "squ_7be58e5f386c00dbdfc8e1a3df15ad1934426e66"          # ← exporta antes de ejecutar
INPUT_JSON  = "Base_veryfied.json"                  # lista de dicts con "content"
OUTPUT_JSON = "Base_veryfied.json"
SCANNER_CMD = ["/home/david/sonar-scanner-7.1.0.4889-linux-x64/bin/sonar-scanner"]                # ← o ['docker','run','--rm', ...]
SCAN_TIMEOUT_SEC = 300                             # seg. máx. por snippet
POLL_INTERVAL   = 2                               # seg. entre chequeos CE
MAX_PARALLEL     = 4        # núm. de SonarScanners concurrentes
USE_DOCKER_SCANNER = False  # True ⇒ usa la imagen sonarsource/sonar-scanner-cli


# Ruta al binario local (solo si USE_DOCKER_SCANNER = False)
LOCAL_SCANNER_BIN = "/home/david/sonar-scanner-7.1.0.4889-linux-x64/bin/sonar-scanner"

if not SONAR_TOKEN:
    raise SystemExit("❌  Define primero la variable de entorno SONAR_TOKEN")


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
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
    data: List[Dict] = json.loads(Path(INPUT_JSON).read_text(encoding="utf-8"))

    with ProcessPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        futures = {pool.submit(analyze_snippet, item["code_after"]): idx
                   for idx, item in enumerate(data)}

        for fut in tqdm(as_completed(futures),
                        total=len(futures),
                        desc="Snippets",
                        unit="snippet",
                        leave=True):
            idx = futures[fut]
            issues, err = fut.result()
            if err:
                data[idx]["issues_after"] = err
                tqdm.write(f"ERROR: {err}")
            else:
                data[idx]["issues_after"] = issues
                tqdm.write(f"Issues en el snippet: {len(issues)}")

    print("✅ Procesados los fixed")

    Path(OUTPUT_JSON).write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"\n✅  Resultado escrito en {OUTPUT_JSON}")

if __name__ == "__main__":
    main()