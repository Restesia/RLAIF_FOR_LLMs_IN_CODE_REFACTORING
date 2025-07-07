import json
from datasets import load_dataset

def main(output_path: str = "code_refinement_dataset.json"):
    # Carga todos los splits disponibles (train/validation/test)
    dataset = load_dataset("code_x_glue_cc_code_refinement", "medium")

    # Unificamos los ejemplos de todos los splits en una lista
    rows = []
    for split_name, split_ds in dataset.items():
        for record in split_ds:
            rows.append({
                "buggy": record["buggy"],
                "fixed": record["fixed"],
            })
    # Guardamos en disco con codificación UTF-8 y pretty-print
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"✅ Dataset guardado en “{output_path}” con {len(rows)} ejemplos.")

if __name__ == "__main__":
    main()