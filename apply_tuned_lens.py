import torch
import torch.nn as nn
import numpy as np
import json
import os
from transformers import AutoProcessor

# === Configuration ===
SAMPLE_PATH = "ffn_states_output/rs/sample_1"
LENS_FILE = "tuned_lens.pt"
TOP_K = 5

def apply_tuned_lens(sample_path, lens_file):
    """
    Applies a pre-trained Tuned Lens to a single sample for analysis.
    """
    metadata_path = os.path.join(sample_path, "metadata.json")
    states_path = os.path.join(sample_path, "states_and_logits.npz")

    if not all(os.path.exists(p) for p in [metadata_path, states_path, lens_file]):
        print(f"Error: Missing one of the required files.")
        return

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        
    saved_data = np.load(states_path)
    all_hidden_states = torch.from_numpy(saved_data['all_hidden_states']).to(torch.float32)

    # --- Load the Tuned Lenses ---
    print("Loading processor and tuned lenses...")
    processor = AutoProcessor.from_pretrained(metadata['model_id']);print(1)

    # Recreate the lens architecture to load the state dict
    num_layers = metadata['num_layers'];print(2)
    lens_state_dict = torch.load(lens_file);print(3)
    hidden_dim = lens_state_dict['0.weight'].shape[1];print(4)
    vocab_size = lens_state_dict['0.weight'].shape[0];print(5)
    
    lenses = nn.ModuleList([nn.Linear(hidden_dim, vocab_size) for _ in range(num_layers)]);print(6)
    lenses.load_state_dict(lens_state_dict);print(7)
    lenses.eval();print(8)

    print("--- Tuned Lens Analysis ---")
    print(f"Model: {metadata['model_id']}, Sample Index: {metadata['index']}")
    print("-" * 25)

    # --- Apply each lens to its corresponding layer's state ---
    with torch.no_grad():
        for i, (hidden_state, lens) in enumerate(zip(all_hidden_states, lenses)):
            tuned_logits = lens(hidden_state)
            
            top_k = torch.topk(tuned_logits, TOP_K)
            tokens = processor.tokenizer.convert_ids_to_tokens(top_k.indices)
            
            layer_name = "Embeddings" if i == 0 else f"Layer {i}"
            print(f"{layer_name}:")
            print(f"  > Predictions: {tokens}")
            
    # --- Compare with Final Model Output ---
    final_logits = torch.from_numpy(saved_data['final_logits'])
    print("\n" + "-" * 25)
    print("Final Layer (Actual Model Output):")
    top_k_final = torch.topk(final_logits, TOP_K)
    tokens_final = processor.tokenizer.convert_ids_to_tokens(top_k_final.indices)
    print(f"  > Predictions: {tokens_final}")

if __name__ == "__main__":
    apply_tuned_lens(SAMPLE_PATH, LENS_FILE)
