import os
import numpy as np
import json
from tqdm import tqdm
import math

def analyze_dape_all_tokens(
    base_dir,
    activation_threshold=0.0,
    specificity_threshold=0.5,
    dape_percentile=1,
    save_path="dape_results.json"
):
    domains = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    domain_to_id = {d: i for i, d in enumerate(domains)}
    id_to_domain = {i: d for d, i in domain_to_id.items()}
    k = len(domains)

    activation_counts = None
    token_counts = np.zeros(k, dtype=int)
    initialized = False

    print("Loading and counting activations...")
    for domain in tqdm(domains):
        domain_id = domain_to_id[domain]
        domain_path = os.path.join(base_dir, domain)
        samples = [s for s in os.listdir(domain_path) if s.startswith("sample_") and os.path.isdir(os.path.join(domain_path, s))]

        for sample in samples:
            npz_path = os.path.join(domain_path, sample, "states_and_logits.npz")
            if not npz_path.endswith(".npz"):
                continue
            try:
                data = np.load(npz_path, allow_pickle=True)
            except Exception as e:
                print(f"[Warning] Skipping corrupted file: {npz_path}\nReason: {e}")
                continue

            all_hidden_states = data["all_hidden_states"]

            if not initialized:
                num_layers = len(all_hidden_states)
                hidden_dim = all_hidden_states[0].shape[1]
                activation_counts = np.zeros((num_layers, hidden_dim, k), dtype=int)
                initialized = True

            for layer_idx, layer_matrix in enumerate(all_hidden_states):  # shape: (seq_len, hidden_dim)
                active_mask = layer_matrix > activation_threshold
                activation_counts[layer_idx, :, domain_id] += np.sum(active_mask, axis=0)
                token_counts[domain_id] += layer_matrix.shape[0]

    print("Computing DAPE ...")
    dape_results = []
    for l in range(num_layers):
        for neuron in range(hidden_dim):
            M_u = activation_counts[l, neuron, :]  # shape: (k,)
            N_i = token_counts + 1e-8
            p_u = M_u / N_i
            norm = np.sum(p_u) + 1e-8
            p_u_norm = p_u / norm

            entropy = -np.sum([p * math.log2(p) if p > 0 else 0 for p in p_u_norm])
            dape_results.append({
                "layer": l,
                "number_in_layer": neuron,
                "neuron_id": l * hidden_dim + neuron,
                "DAPE": float(entropy),
                "raw_probs": p_u.tolist()
            })

    # === Compute summary statistics
    dape_scores = [x["DAPE"] for x in dape_results]
    dape_min = float(np.min(dape_scores))
    dape_max = float(np.max(dape_scores))
    dape_mean = float(np.mean(dape_scores))
    dape_std = float(np.std(dape_scores))
    total_neurons = len(dape_results)

    # === Select domain-specific neurons
    dape_results.sort(key=lambda x: x["DAPE"])
    cutoff = max(1, int(len(dape_results) * dape_percentile / 100))
    top_neurons = dape_results[:cutoff]

    output_neurons = []
    for entry in top_neurons:
        probs = entry["raw_probs"]
        domain_id = int(np.argmax(probs))
        max_prob = probs[domain_id]
        specific_domain = id_to_domain[domain_id] if max_prob > specificity_threshold else "ambiguous"

        # Convert domain-indexed probs to named dict
        named_probs = {id_to_domain[i]: float(p) for i, p in enumerate(probs)}

        output_neurons.append({
            "neuron_id": entry["neuron_id"],
            "layer": entry["layer"],
            "number_in_layer": entry["number_in_layer"],
            "DAPE": entry["DAPE"],
            "specific_domain": specific_domain,
            "activation_probability": max_prob,
            "activation_probabilities": named_probs
        })

    # === Final JSON structure
    result = {
        "meta": {
            "dape_min": dape_min,
            "dape_max": dape_max,
            "dape_mean": dape_mean,
            "dape_std": dape_std,
            "total_neurons": total_neurons,
            "num_domain_specific_neurons": len(output_neurons),
            "activation_threshold": activation_threshold,
            "specificity_threshold": specificity_threshold,
            "domains": domains,
            "dape_percentile": dape_percentile
        },
        "domain_specific_neurons": output_neurons
    }

    with open(save_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n DAPE analysis complete. Saved to: {save_path}")
    print("Top neurons:")
    for x in output_neurons[:10]:
        print(x)

    return result
