import argparse
from pathlib import Path
import sys


def merge_jsonl(path_a: Path, path_b: Path, out_path: Path) -> None:
    """Copia las líneas de los dos ficheros en 'out_path'."""
    # Abrimos los tres archivos con manejo de contexto
    with path_a.open('r', encoding='utf-8') as fa, \
            path_b.open('r', encoding='utf-8') as fb, \
            out_path.open('w', encoding='utf-8') as fout:

        # Escribimos cada línea tal como está:
        # (asumimos que cada línea ya es JSON válido)
        for line in fa:
            fout.write(line.rstrip('\n') + '\n')
        for line in fb:
            fout.write(line.rstrip('\n') + '\n')

def main(args=None):
    parser = argparse.ArgumentParser(
        description="Fusiona dos ficheros JSONL en uno solo, conservando el orden."
    )
    parser.add_argument("input1", type=Path, help="Primer archivo JSONL")
    parser.add_argument("input2", type=Path, help="Segundo archivo JSONL")
    parser.add_argument("output", type=Path, help="Archivo JSONL resultante")
    ns = parser.parse_args(args)

    # Comprobaciones básicas
    for p in (ns.input1, ns.input2):
        if not p.is_file():
            parser.error(f"El archivo {p} no existe o no es un fichero regular.")

    try:
        merge_jsonl(ns.input1, ns.input2, ns.output)
        print(f"Archivo fusionado escrito en: {ns.output}")
    except Exception as exc:
        print(f"Error al fusionar archivos: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()