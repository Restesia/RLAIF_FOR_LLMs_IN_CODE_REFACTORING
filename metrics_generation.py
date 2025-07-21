
import json
import sys
import tempfile
import subprocess
import os
import re
from tqdm import tqdm
from collections import defaultdict


def extract_package_and_filename(code_str, default_name):
    """Return the package (if any) and a reasonable filename for the main type."""
    package_match = re.search(r'^\s*package\s+([\w.]+)\s*;', code_str, re.MULTILINE)
    package = package_match.group(1) if package_match else None

    # Guess the first declared type to build the filename
    type_match = re.search(r'\b(class|interface|enum)\s+([A-Za-z_][A-Za-z0-9_]*)', code_str)
    typename = type_match.group(2) if type_match else default_name
    filename = f"{typename}.java"
    return package, filename


def write_and_compile(code_str, workdir):
    """Write the snippet to disk respecting its package and invoke javac."""
    package, filename = extract_package_and_filename(code_str, "Snippet")

    target_dir = workdir
    if package:
        target_dir = os.path.join(workdir, *package.split('.'))
        os.makedirs(target_dir, exist_ok=True)

    source_path = os.path.join(target_dir, filename)
    with open(source_path, 'w', encoding='utf-8') as fp:
        fp.write(code_str)

    result = subprocess.run(["javac", source_path], capture_output=True)
    return result.returncode == 0, result.stderr.decode()


def issues_as_set(issue_list):
    """Represent each issue as a tuple we can diff."""
    s = set()
    for issue in issue_list:
        s.add((issue.get('rule'), issue.get('line'), issue.get('message')))
    return s


def main(path):
    with open(path, 'r', encoding='utf-8') as fh:
        data = json.load(fh)

    total_fixed = 0
    total_added = 0
    not_compiling = 0
    per_snippet = []

    errors = 0

    for idx, itm in tqdm(enumerate(data, 1)):
        before = issues_as_set(itm.get('issues_before', []))

        raw_after = itm.get('issues_after', [])
        # Normalizar por si viene doblemente anidado [[...], ""]
        if isinstance(raw_after, list) and len(raw_after) == 2 and isinstance(raw_after[0], list):
            after = issues_as_set(raw_after[0])
            fixed = len(before - after)
            added = len(after - before)
            total_fixed += fixed
            total_added += added
        elif not isinstance(raw_after, str):
            after = issues_as_set(raw_after)
            fixed = len(before - after)
            added = len(after - before)
            total_fixed += fixed
            total_added += added
        else:
            errors = errors + 1
            after = None



        if after != None:
            # Compilar el código refactorizado
            with tempfile.TemporaryDirectory() as tmp:
                ok, _ = write_and_compile(itm.get('code_after', ''), tmp)
            if not ok:
                not_compiling += 1

            per_snippet.append({'idx': idx, 'fixed': fixed, 'added': added})

    # Salida
    print("=== Summary ===")
    print(f"Total issues fixed     : {total_fixed}")
    print(f"Total issues introduced: {total_added}")
    print(f"Snippets not compiling : {not_compiling}\n")
    print(f"Total SonarQube Errors: {errors}")


    print("=== Details per snippet ===")
    for sn in per_snippet:
        print(f"[{sn['idx']}] fixed={sn['fixed']} added={sn['added']}")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python evaluate_refactor_sonarqube.py <results.json>')
        sys.exit(1)
    main(sys.argv[1])
