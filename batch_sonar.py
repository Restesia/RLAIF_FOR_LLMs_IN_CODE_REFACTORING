#!/usr/bin/env python3
"""
batch_sonar.py  – Analiza en masa snippets Java con SonarQube
"""

import json
import os
import subprocess
import tempfile
import time
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
INPUT_JSON  = "code_refinement_dataset.json"                  # lista de dicts con "content"
OUTPUT_JSON = "snippets_with_issues.json"
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

def analyze_snippet(snippet: str) -> Tuple[List[Dict], str]:
    """
    Analiza un snippet y devuelve (issues, error_msg). Si error_msg != "", falló.
    Se ejecuta en un proceso hijo para permitir paralelismo verdadero.
    """
    project_key = f"snippet-{uuid.uuid4().hex[:12]}"

    snippet ="public class Snippet {\n" + snippet + "\n}"

    try:
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)

            # 1) escribir código Java y properties
            (tmpdir / "Snippet.java").write_text(snippet, encoding="utf-8")
            print((tmpdir / "Snippet.java").read_text())
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
        futures = {pool.submit(analyze_snippet, item["buggy"]): idx
                   for idx, item in enumerate(data[:1600])}

        for fut in tqdm(as_completed(futures),
                        total=len(futures),
                        desc="Snippets",
                        unit="snippet",
                        leave=True):
            idx = futures[fut]
            issues, err = fut.result()
            if err:
                data[idx]["issues_error"] = err
                tqdm.write(f"ERROR: {err}")
            else:

                data[idx]["buggy_issues"] = issues
                tqdm.write(f"Issues en el snippet: {len(issues)}")
    print("✅ Procesados los buggy")

    with ProcessPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        futures = {pool.submit(analyze_snippet, item["fixed"]): idx
                   for idx, item in enumerate(data[:1600])}

        for fut in tqdm(as_completed(futures),
                        total=len(futures),
                        desc="Snippets",
                        unit="snippet",
                        leave=True):
            idx = futures[fut]
            issues, err = fut.result()
            if err:
                data[idx]["issues_error"] = err
                tqdm.write(f"ERROR: {err}")
            else:
                data[idx]["fixed_issues"] = issues
                tqdm.write(f"Issues en el snippet: {len(issues)}")

    print("✅ Procesados los fixed")

    Path(OUTPUT_JSON).write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"\n✅  Resultado escrito en {OUTPUT_JSON}")

if __name__ == "__main__":
    main()