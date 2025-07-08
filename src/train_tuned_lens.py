import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm

# === Configuration ===
BASE_DATA_DIR = "ffn_states_output" 
DOMAINS = ["rs", "com", "doc", "ad", "med"]
OUTPUT_LENS_FILE = "tuned_lens.pt"
LEARNING_RATE = 1e-3
EPOCHS = 10
BATCH_SIZE = 32

def train_lenses():
    """
    Trains a set of linear probes (tuned lenses) for each layer using data from all specified domains.
    """
    print(f"Loading data from base directory: {BASE_DATA_DIR}")
    all_states = []
    all_logits = []
    
    all_sample_dirs = []
    for domain in DOMAINS:
        domain_path = os.path.join(BASE_DATA_DIR, domain)
        if not os.path.isdir(domain_path):
            print(f"Warning: Domain directory not found, skipping: {domain_path}")
            continue
        
        print(f"-> Found domain: {domain}")
        sample_dirs_in_domain = [os.path.join(domain_path, d) for d in os.listdir(domain_path) if os.path.isdir(os.path.join(domain_path, d))]
        all_sample_dirs.extend(sample_dirs_in_domain)

    if not all_sample_dirs:
        print(f"Error: No sample data found in any of the specified domain directories under {BASE_DATA_DIR}.")
        return

    for sample_path in tqdm(all_sample_dirs, desc="Loading samples from all domains"):
        states_path = os.path.join(sample_path, "states_and_logits.npz")
        if os.path.exists(states_path):
            data = np.load(states_path)
            if 'all_hidden_states' in data and 'final_logits' in data:
                all_states.append(data['all_hidden_states'])
                all_logits.append(data['final_logits'])


    all_states = torch.from_numpy(np.array(all_states, dtype=np.float32)) # (num_samples, num_layers, hidden_dim)
    all_logits = torch.from_numpy(np.array(all_logits, dtype=np.float32)) # (num_samples, vocab_size)


    
    num_samples, num_layers, hidden_dim = all_states.shape
    vocab_size = all_logits.shape[1]
    
    print(f"Loaded {num_samples} total samples across {len(DOMAINS)} domains. Training lenses for {num_layers} layers.")
    print(f"Hidden Dim: {hidden_dim}, Vocab Size: {vocab_size}")

    lenses = nn.ModuleList()
    loss_fn = nn.MSELoss() # Mean Squared Error is a common choice

    for i in tqdm(range(num_layers), desc="Training Lenses"):
        X = all_states[:, i, :]
        y = all_logits          
        
        lens = nn.Linear(hidden_dim, vocab_size)
        optimizer = optim.Adam(lens.parameters(), lr=LEARNING_RATE)
        
        for epoch in range(EPOCHS):
            permutation = torch.randperm(num_samples)
            for j in range(0, num_samples, BATCH_SIZE):
                indices = permutation[j:j+BATCH_SIZE]
                batch_X, batch_y = X[indices], y[indices]

                optimizer.zero_grad()
                predictions = lens(batch_X)
                loss = loss_fn(predictions, batch_y)
                loss.backward()
                optimizer.step()
        
        lenses.append(lens)

    torch.save(lenses.state_dict(), OUTPUT_LENS_FILE)
    print(f"\nSuccessfully trained {len(lenses)} universal lenses.")
    print(f"Saved tuned lens weights to '{OUTPUT_LENS_FILE}'")


if __name__ == "__main__":
    train_lenses()