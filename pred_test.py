from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import json, torch, tqdm

tok = AutoTokenizer.from_pretrained("./Modelos/llama2-7b", use_fast=False)
tok.pad_token = tok.eos_token
tok.padding_side = "right"
bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
base   = AutoModelForCausalLM.from_pretrained("./Modelos/llama2-7b", quantization_config=bnb_cfg, device_map="auto")
ft     = PeftModel.from_pretrained(base, "./checkpoints/llama2-7b-mss4j")


BATCH = 16
GEN_KW = dict(max_new_tokens=64, num_beams=1, do_sample=False)

with open("/home/david/Documentos/ISIA/TFM/Código/dataset/procesed_dataset/test.jsonl") as fh:
    rows = [json.loads(l) for l in fh]


prompts = [f"<s>[INST] Corrige:\n```java\n{r['text']}\n```[/INST]"
           for r in rows]

def batched(prompts, model, tag: str):
    bar = tqdm.tqdm(range(0, len(prompts), BATCH), desc=f"✏️  Generando {tag}")
    for i in bar:
        batch_enc = tok(prompts[i:i + BATCH],
                        return_tensors="pt",
                        padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**batch_enc, **GEN_KW)
        decoded = tok.batch_decode(out, skip_special_tokens=True)
        yield from decoded
        bar.set_postfix(done=min(i + BATCH, len(prompts)))

for tag, mdl in [("base", base), ("ft", ft)]:
    out_path = f"pred_{tag}.java"
    with open(out_path, "w") as fh:
        for txt in batched(prompts, mdl, tag):
            fixed = txt.split("```java")[-1].rstrip("`").strip()
            fh.write(fixed + "\n")
    print(f"✓ {out_path} listo")