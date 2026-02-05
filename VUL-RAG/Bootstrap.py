import json
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from pathlib import Path
from tqdm import tqdm
import random
from collections import defaultdict
from common.util.common_util import calculate_VD_metrics
from v2testing import get_ndcg
import pdb

CWE_LIST = [
    "CWE-20", "CWE-119", "CWE-125", "CWE-200", "CWE-264",
    "CWE-362", "CWE-401", "CWE-416", "CWE-476", "CWE-787"
]


def bootstrap_pairs_all_cwes(
    input_dir: str,
    n_bootstrap: int = 10000,
    seed: int = 42
):
    """
    Phase 1:
    - Load each CWE file
    - Pair vul/non-vul items by id
    - Perform bootstrap resampling over pairs
    - Return all shuffled samples without computing metrics
    """
    random.seed(seed)

    paired_by_cwe = {}

    # ---- Load + pair once per CWE ----
    for cwe in CWE_LIST:
        path = os.path.join(input_dir, f"{cwe}.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        vul_by_id = {item["id"]: item for item in data["vul_data"]}
        non_vul_by_id = {item["id"]: item for item in data["non_vul_data"]}

        common_ids = sorted(vul_by_id.keys() & non_vul_by_id.keys())
        
        if not common_ids:
            raise RuntimeError(f"No paired items found for {cwe}")

        paired_by_cwe[cwe] = [
            (vul_by_id[i], non_vul_by_id[i]) for i in common_ids
        ]

    # ---- Bootstrap over pairs for all CWEs ----
    shuffled = []

    for _ in range(n_bootstrap):
        iteration = {}

        for cwe, pairs in paired_by_cwe.items():
            sampled_pairs = random.choices(pairs, k=len(pairs))

            iteration[cwe] = {
                "sampled_vul": [v for v, _ in sampled_pairs],
                "sampled_non_vul": [nv for _, nv in sampled_pairs],
            }

        shuffled.append(iteration)

    return shuffled

def evaluate_bootstrap_samples(shuffled, input_dir, ci=(2.5, 97.5)):
    """
    Phase 2:
    - Iterate over precomputed bootstrap samples
    - Call calculate_VD_metrics once per iteration
    - Aggregate metric distributions per CWE and Overall
    - Finalize to mean + CI
    """
    output_path = Path(input_dir) / "bootstrap.json"

    # results[cwe][metric] -> list of bootstrap values
    results = defaultdict(lambda: defaultdict(list))

    # --- collect bootstrap distributions ---
    for iteration in tqdm(shuffled, desc="Bootstrap iterations"):

        # 1. Get metrics for all CWEs + Overall
        metrics_by_cwe = calculate_VD_metrics(input_dir, V2=True, save_to_file=False, bootstrap=iteration)

        # 2. Store all metrics (including Overall from calculate_VD_metrics)
        for cwe, metrics in metrics_by_cwe.items():
            for metric_name, value in metrics.items():
                if value is not None:
                    results[cwe][metric_name].append(value)
        
        # 3. Add NDCG to the Overall metrics
        ndcgs = get_ndcg(iteration, CWE_LIST, 'TruePairVul')
        for k_metric, value in ndcgs.items():
            results["Overall"][k_metric].append(value)

    # --- finalize: mean + confidence intervals ---
    finalized_data = {}

    for cwe, metrics in results.items():
        finalized_data[cwe] = {}
        for metric_name, values in metrics.items():
            values = np.asarray(values)

            # Round final statistics to 4 decimals for consistency
            finalized_data[cwe][metric_name] = {
                "mean": round(float(values.mean()), 3),
                "ci_low": round(float(np.percentile(values, ci[0])), 3),
                "ci_high": round(float(np.percentile(values, ci[1])), 3),
            }

    with open(output_path, "w") as f:
        json.dump(finalized_data, f, indent=2)

    return finalized_data  


if __name__ == '__main__':


    for num in range(1, 10):
        filepath = f"/home/ludhamm/Code/Vul-RAG_Research/Vul-RAG/partial/TruePairVul/6_decision_results/plus_gpt-4o_prompt=3/{num}_maxentries_results"

        output = bootstrap_pairs_all_cwes(filepath)
        evaluate_bootstrap_samples(output, filepath)
