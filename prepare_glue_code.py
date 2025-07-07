
import json
from pathlib import Path
from datasets import load_dataset

# ────────────────────────────────────
# 1. CONFIGURACIÓN GLOBAL
# ────────────────────────────────────
DATASET_NAME = "google/code_x_glue_cc_code_refinement"
CONFIG       = "medium"                    # "small" o "medium"
OUT_DIR      = Path("dataset/glue_code")    # Carpeta destino

INST_OPEN  = "<s>"
INST_CLOSE = "</s>"
NL         = "\n"

PROMPT_TMPL = (
    "{inst_open} Eres un experto en Java. Corrige el siguiente fragmento de código:"
    "{nl}```java{nl}{before}{nl}```{nl}{inst_close}{nl}"
)

SPLITS = {                    # alias → split HF
    "train": "train",
    "val"  : "validation",
    "test" : "test",
}
# ────────────────────────────────────


def make_prompt(code_buggy: str) -> str:
    """Devuelve el prompt con el código buggy embebido."""
    return PROMPT_TMPL.format(
        inst_open=INST_OPEN,
        inst_close=INST_CLOSE,
        nl=NL,
        before=code_buggy.rstrip(),   # quita \n final si existe
    )


def write_jsonl(ds, out_path: Path) -> None:
    """Escribe el Dataset → JSONL con prompt en 'buggy'."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in ds:
            json.dump(
                {
                    "buggy": make_prompt(rec["buggy"]),
                    "fixed": rec["fixed"],
                },
                f,
                ensure_ascii=False,
            )
            f.write("\n")
    print(f"  ↳ {out_path}  ({out_path.stat().st_size/1024:.1f} KB)")


def main() -> None:
    print(f"⬇️  Descargando '{DATASET_NAME}' ({CONFIG}) …")
    for alias, hf_split in SPLITS.items():
        print(f"• Procesando split '{hf_split}' …")
        ds = load_dataset(DATASET_NAME, CONFIG, split=hf_split)
        write_jsonl(ds, OUT_DIR / f"{alias}.jsonl")

    print("\n✅  Dataset preparado con prompts. Carpeta:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
