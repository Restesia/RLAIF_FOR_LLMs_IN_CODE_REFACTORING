#!/usr/bin/env python3
"""
extrae_snippets_stack.py
--------------------------------
Descarga N fragmentos de código Java desde The Stack y los guarda en JSON.
Requiere:  pip install datasets tqdm
"""

import json
import hashlib
from tqdm import tqdm
from datasets import load_dataset
import subprocess
import tempfile
from typing import Optional, Dict, List, Tuple
import os
from pathlib import Path
import re
import pickle
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
import requests
from multiprocessing import Manager



OUTPUT_PATH = "RLAIF_estimation_code.json"

SONAR_HOST = "http://localhost:9000"
SONAR_TOKEN = "squ_7be58e5f386c00dbdfc8e1a3df15ad1934426e66"          # ← exporta antes de ejecutar
INPUT_JSON  = "code_refinement_dataset.json"                  # lista de dicts con "content"
SCANNER_CMD = ["/home/david/sonar-scanner-7.1.0.4889-linux-x64/bin/sonar-scanner"]                # ← o ['docker','run','--rm', ...]
SCAN_TIMEOUT_SEC = 300                             # seg. máx. por snippet
POLL_INTERVAL   = 2                               # seg. entre chequeos CE
MAX_PARALLEL     = 8        # núm. de SonarScanners concurrentes
USE_DOCKER_SCANNER = False  # True ⇒ usa la imagen sonarsource/sonar-scanner-cli


# Ruta al binario local (solo si USE_DOCKER_SCANNER = False)
LOCAL_SCANNER_BIN = "/home/david/sonar-scanner-7.1.0.4889-linux-x64/bin/sonar-scanner"

MIN_OFFSET = 1_500_000  # requisito ❷
PROGRESS_FILE = Path("progress.txt")
HASHES_FILE = Path("hashes.pickle")  # hashes ya vistos (dedup)
CHECK_EVERY = 1_000  # guarda progreso cada N filas


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

def compila_java(file: str = None, src_path: Path = None, tmpdir: str = None) ->bool:



    proc = subprocess.run(
        ["javac", str(src_path)+"/"+file],
        capture_output=True,
        text=True,
        cwd=tmpdir,
    )

    ok = proc.returncode == 0
    return ok

def eliminar_comentarios_java(codigo: str) -> str:

    patron = re.compile(
        r"""
        //.*?$           |   # Comentario de línea
        /\*.*?\*/            # Comentario de bloque (no anidados)
        """,
        re.MULTILINE | re.DOTALL | re.VERBOSE,
    )
    return re.sub(patron, "", codigo)

def corpus(row, seen_hashes):

    code = row["content"]

    code = eliminar_comentarios_java(code)

    num_lineas = code.count('\n')
    num_token_estimacion = int(
        len(code) / 4.2 + 0.5)  # A groso modo, es una regla de estimación con un 10% / 20% de error

    if num_lineas < 250 and num_token_estimacion < 1400:
        with tempfile.TemporaryDirectory() as tmpdir:
            file = nombre_archivo_java(code)
            src_path = Path(tmpdir)
            file_path = Path(tmpdir, file)
            file_path.write_text(code, encoding="utf-8")

            if compila_java(file, src_path, tmpdir):
                tqdm.write("Compiló java")
                issues = analyze_file(src_path)
                tqdm.write(f"Fue analizado por SonarQube, {len(issues[0])} issues detectadas")

                if len(issues) > 1:
                    h = hashlib.md5(code.encode()).hexdigest()
                    if not h in seen_hashes:


                        seen_hashes[h] = True

                        snippet = {
                                "content": code,
                                "path": row.get("path"),
                                "repository": row.get("repository_name"),
                                "license": row.get("licenses"),
                                "issues": issues[0],
                                "Num_token_estimado": num_token_estimacion,
                                "Nombre del archivo": file
                            }
                        return snippet

    return None



def main():

    def checkpoint(current_idx: int) -> None:
        PROGRESS_FILE.write_text(str(current_idx))
        with open(HASHES_FILE, "wb") as fh:
            pickle.dump(dict(seen_hashes), fh)
        if snippets:
            with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
                json.dump(existing_snippets + snippets, fh,
                          ensure_ascii=False, indent=2)
        print(f"✓ Checkpoint en idx {current_idx:,}  "
              f"({len(existing_snippets)+len(snippets):,} fragmentos totales)")

    try:
        OFFSET = max(MIN_OFFSET, int(PROGRESS_FILE.read_text().strip()))
    except (FileNotFoundError, ValueError):
        OFFSET = MIN_OFFSET

    # --- carga de The Stack en modo streaming -----------------------------
    print("Conectando con The Stack (Java)…")
    dset = load_dataset(
        "bigcode/the-stack",
        data_dir="data/java",
        split="train",
        streaming=True,
        use_auth_token=None,
    ).skip(OFFSET)

    manager      = Manager()
    seen_hashes  = manager.dict()

    print(f"→ Reanudando desde índice global {OFFSET:,}")

    # ❸ Carga del JSON con snippets ya extraídos
    existing_snippets: list[dict] = []
    if Path(OUTPUT_PATH).exists():
        try:
            with open(OUTPUT_PATH, "r", encoding="utf-8") as fh:
                existing_snippets = json.load(fh)
            print(f"  {len(existing_snippets):,} fragmentos previos encontrados")
        except json.JSONDecodeError:
            print("⚠️  El JSON existente está corrupto; se ignora")

    print("# Pickle load")

    if HASHES_FILE.exists():
        with open(HASHES_FILE, "rb") as fh:
            persisted_hashes = pickle.load(fh)
    else:
        persisted_hashes = {}

    print("# MD5 computing")
    # incluye los hashes de los snippets ya guardados
    for snip in existing_snippets:
        h = hashlib.md5(snip["content"].encode()).hexdigest()
        persisted_hashes[h] = True


    manager      = Manager()
    seen_hashes  = manager.dict(persisted_hashes)

    # contenedor para NUEVOS fragmentos de esta sesión
    snippets     = []

    # `idx` es el índice global (se inicia en OFFSET)
    idx = OFFSET

    print("# Iniciando descarga")

    try:
        with ProcessPoolExecutor(max_workers=MAX_PARALLEL) as pool:
            in_flight = set()
            for row in tqdm(dset, desc="Descargando"):
                fut = pool.submit(corpus, row, seen_hashes)
                in_flight.add(fut)

                idx += 1

                # Mantén solo un número razonable de tareas en vuelo
                if len(in_flight) >= MAX_PARALLEL * 4:
                    for done in as_completed(in_flight, timeout=None):
                        in_flight.remove(done)
                        res = done.result()
                        if res:
                            snippets.append(res)
                    tqdm.write(f"Guardados {len(snippets + existing_snippets)} fragmentos")

                if idx % CHECK_EVERY == 0:
                    checkpoint(idx)

            # drena lo que quede
            for done in as_completed(in_flight):
                res = done.result()
                if res:
                    snippets.append(res)

    except KeyboardInterrupt:
        # --- guardar -----------------------------------------------------------
        print(f"\nGuardados {len(snippets + existing_snippets)} fragmentos")

        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            if snippets:
                with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
                    json.dump(existing_snippets + snippets, fh,
                              ensure_ascii=False, indent=2)
    finally:
        print(f"\nGuardados {len(snippets + existing_snippets)} fragmentos")
        print("idx: ", idx)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            if snippets:
                with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
                    json.dump(existing_snippets + snippets, fh,
                              ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
