import torch
import numpy as np
import json
import os
from transformers import AutoProcessor, LlavaForConditionalGeneration


def run_logit_lens_analysis(sample_path, top_k=5):
    """
    Performs Logit Lens analysis on a single saved sample.
    """
    metadata_path = os.path.join(sample_path, "metadata.json")
    states_path = os.path.join(sample_path, "states_and_logits.npz")

    if not (os.path.exists(metadata_path) and os.path.exists(states_path)):
        print(f"Error: Could not find required files in {sample_path}")
        return

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        
    saved_data = np.load(states_path)
    all_hidden_states = torch.from_numpy(saved_data['all_hidden_states'])
    final_logits = torch.from_numpy(saved_data['final_logits'])

    print("--- Analysis Details ---")
    print(f"Model: {metadata['model_id']}")
    print(f"Domain: {metadata['domain']}, Sample Index: {metadata['index']}")
    print(f"Ground Truth: {metadata['ground_truth']}")
    print(f"Model Prediction: {metadata['prediction']}")
    print("-" * 25)

    # --- Load Model Components ---
    print("Loading model for analysis...")
    model = LlavaForConditionalGeneration.from_pretrained(
        metadata['model_id'],
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    processor = AutoProcessor.from_pretrained(metadata['model_id'])
    
    # Access final layers with the correct paths
    print("Accessing final model layers for analysis...")
    # The final normalization layer is an attribute of the base LlamaModel.
    final_norm = model.language_model.norm
    
    # The language model head is reliably accessed via get_output_embeddings().
    lm_head = model.get_output_embeddings()
    print("Successfully accessed final layers.")

    # --- Perform Logit Lens ---
    print("\n--- Logit Lens Results (Top 5 Tokens) ---\n")
    
    for i, hidden_state in enumerate(all_hidden_states):
        hidden_state = hidden_state.to(torch.float32)
        
        with torch.no_grad():
        
            logits_at_layer_i = lm_head(final_norm(hidden_state))

        top_k = torch.topk(logits_at_layer_i, top_k)
        
        # Use the processor's tokenizer to decode the token ids
        tokens = processor.tokenizer.convert_ids_to_tokens(top_k.indices)
        
        layer_name = f"Layer {i} (Embeddings)" if i == 0 else f"Layer {i}"
        print(f"{layer_name}:")
        print(f"  > Predictions: {tokens}")
    
    # --- Compare with Final Model Output ---
    print("\n" + "-" * 25)
    print("Final Layer (Actual Model Output):")
    top_k_final = torch.topk(final_logits, top_k)
    tokens_final = processor.tokenizer.convert_ids_to_tokens(top_k_final.indices)
    print(f"  > Predictions: {tokens_final}")


if __name__ == "__main__":
    
    SAMPLE_PATH = "ffn_states_output/rs/sample_0"
    TOP_K = 5
    
    run_logit_lens_analysis(SAMPLE_PATH, top_k=TOP_K)
