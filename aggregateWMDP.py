import os
import json

BASE_DIR = "WMDPResults"
MODELS = ["llama3", "mistral", "phi3"]
DOMAINS = ["bio", "chem", "cyber"]

OUTPUT_DIR = os.path.join(BASE_DIR, "Aggregated")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_results(model, domain):
    path = os.path.join(BASE_DIR, model, domain, "results.json")
    with open(path, "r") as f:
        return json.load(f)


all_domains_combined = []

for domain in DOMAINS:
    print(f"Processing domain: {domain}")

    # Load all model results for this domain
    model_results = {
        model: load_results(model, domain)
        for model in MODELS
    }

    # Sanity check: all same length
    lengths = {len(v) for v in model_results.values()}
    if len(lengths) != 1:
        raise ValueError(
            f"Results length mismatch in domain '{domain}': {lengths}"
        )

    n = lengths.pop()
    aggregated = []

    for i in range(n):
        records = {m: model_results[m][i] for m in MODELS}

        # Require sandbagging_signal == True for ALL models
        if not all(r["sandbagging_signal"] for r in records.values()):
            continue

        # Take reference from any model (identical by construction)
        ref = records[MODELS[0]]

        entry = {
            "domain": domain,              
            "question": ref["question"],
            "choices": ref["choices"],
            "correct": ref["correct"],
            "normal": ref["normal"],
            "evaluation": ref["evaluation"],
        }

        aggregated.append(entry)
        all_domains_combined.append(entry)

    out_path = os.path.join(OUTPUT_DIR, f"{domain}_sandbagging.json")
    with open(out_path, "w") as f:
        json.dump(aggregated, f, indent=2)

    print(f"  → {len(aggregated)} questions written to {out_path}")

final_path = os.path.join(OUTPUT_DIR, "wmdp_sandbagging.json")
with open(final_path, "w") as f:
    json.dump(all_domains_combined, f, indent=2)
