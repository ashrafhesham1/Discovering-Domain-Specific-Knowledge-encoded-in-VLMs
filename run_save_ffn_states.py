import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from load_bench import load_pair
import os
from tqdm import tqdm
import json
import numpy as np

# === Config ===
model_id = "llava-hf/llava-1.5-7b-hf"
device = "cuda" if torch.cuda.is_available() else "cpu"
n_samples = 1000  # Number of samples to process per domain
save_dir = "ffn_states_output"
domains = ["com", "med", "rs", "ad", "doc"]

print(f"Using device: {device}")
os.makedirs(save_dir, exist_ok=True)

# === Load Model & Processor ===


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    output_hidden_states=True,
    quantization_config=quantization_config,
    device_map=device,
)#.to(device)

processor = AutoProcessor.from_pretrained(model_id)

for domain in domains:
    print(f"\nProcessing domain: {domain}")
    domain_save_dir = os.path.join(save_dir, domain)
    os.makedirs(domain_save_dir, exist_ok=True)

    loader = load_pair(domain, split_flag="train", simple_flag=False)
    
    for i, (prompt, image, answer, index) in tqdm(enumerate(loader), total=n_samples, desc=f"Processing {domain}"):
        if i >= n_samples:
            break

        sample_save_path = os.path.join(domain_save_dir, f"sample_{index}")
        os.makedirs(sample_save_path, exist_ok=True)
        
        if os.path.exists(os.path.join(sample_save_path, "metadata.json")):
            continue

        inputs = processor(prompt, image, return_tensors="pt").to(device, torch.float16)

        with torch.no_grad():
            output_fwd = model(**inputs, output_hidden_states=True)

        all_hidden_states = output_fwd.hidden_states
        
        all_hidden_states_np = [
            layer_hs[0, -1].detach().cpu().numpy() for layer_hs in all_hidden_states
        ]

        final_logits_np = output_fwd.logits[0, -1].detach().cpu().numpy()

        np.savez_compressed(
            os.path.join(sample_save_path, "states_and_logits.npz"),
            all_hidden_states=all_hidden_states_np,
            final_logits=final_logits_np,
        )

        metadata = {
            "index": index,
            "domain": domain,
            "model_id": model_id,
            "prompt": prompt,
            "ground_truth": str(answer),
            "num_layers": len(all_hidden_states_np),
        }
        with open(os.path.join(sample_save_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

print("\nProcessing complete.")
