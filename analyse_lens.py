import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import json
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoProcessor, LlavaForConditionalGeneration

class AnalysisRunner:
    """
    A class to orchestrate aggregate analysis of model hidden states.

    """

    def __init__(self, lens_type, data_dir='ffn_states_output', output_dir='analysis_results', lens_path=None):
        """
        Initializes the AnalysisRunner.

        Args:
            lens_type (str): Type of analysis: 'language', 'vision', or 'logit'.
            data_dir (str): Base directory where saved sample data is located.
            output_dir (str): Directory to save plots and results.
        """
        self.lens_type = lens_type
        self.data_dir = data_dir
        self.output_dir = os.path.join(output_dir, self.lens_type)
        self.lens_path = lens_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.results_df = None

        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Initialized Analyzer for '{self.lens_type}' lens type.")
        print(f"Results will be saved in: {self.output_dir}")

        self._setup()

    def _setup(self):
        """
        Performs initial setup like scanning files and loading models/lenses.
        This is called automatically during initialization.
        """
        # Scan for all sample files.
        path_pattern_multi_domain = os.path.join(self.data_dir, "*", "sample_*") #data_dir/*domain*/sample_*
        self.sample_dirs = glob.glob(path_pattern_multi_domain)
        
        if not self.sample_dirs:
            path_pattern_single_domain = os.path.join(self.data_dir, "sample_*") #data_dir/dample_*
            self.sample_dirs = glob.glob(path_pattern_single_domain)

        if not self.sample_dirs:
            raise FileNotFoundError(f"Error: No 'sample_*' directories found in '{self.data_dir}' using either multi-domain ('*/sample_*') or single-domain ('sample_*') patterns.")

        print(f"Found {len(self.sample_dirs)} samples.")

        # Load metadata and determine dimensions from the first sample
        with open(os.path.join(self.sample_dirs[0], "metadata.json"), 'r') as f:
            metadata = json.load(f)
        
        self.model_id = metadata['model_id']
        self.processor = AutoProcessor.from_pretrained(self.model_id)

        if self.lens_type in ['language', 'logit']:
            self.states_key = 'all_hidden_states'
            self.num_layers = metadata['num_layers']
        else: # vision
            self.states_key = 'vision_hidden_states'
            self.num_layers = metadata['num_vision_layers']

        data = np.load(os.path.join(self.sample_dirs[0], "states_and_logits.npz"))
        hidden_dim = data[self.states_key].shape[-1]
        vocab_size = data['final_logits'].shape[-1]

        # Load Lenses OR Model Head
        self.lenses, self.lm_head, self.final_norm = None, None, None
        if self.lens_type in ['language', 'vision']:
            lens_file = 'tuned_lens.pt' if self.lens_type == 'language' else 'vision_lens.pt'
            #lens_path = os.path.join(self.data_dir, '..', lens_file)
            lens_path = self.lens_path
            if not os.path.exists(lens_path):
                raise FileNotFoundError(f"Error: Lens file not found at {lens_path}")
            print(f"Loading {self.lens_type} tuned lenses...")
            self.lenses = self._load_lenses(lens_path, self.num_layers, hidden_dim, vocab_size)
        
        elif self.lens_type == 'logit':
            print("Loading full model for Logit Lens analysis...")
            model = LlavaForConditionalGeneration.from_pretrained(
                self.model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map='auto'
            )
            self.final_norm = model.language_model.norm
            self.lm_head = model.get_output_embeddings()
            print("Model loaded.")

    def _load_lenses(self, lens_path, num_layers, hidden_dim, vocab_size):
        """Helper to load a state dict into a ModuleList of lenses."""
        lenses = nn.ModuleList([nn.Linear(hidden_dim, vocab_size) for _ in range(num_layers)])
        lenses.load_state_dict(torch.load(lens_path, map_location=self.device))
        lenses.to(self.device)
        lenses.eval()
        return lenses

    def run_and_save_all_metrics(self, max_n_samples=float('inf')):
        """
        Processes all samples to calculate every metric.

        This is the main computational loop. It iterates through all found samples,
        applies the appropriate lens, calculates all analysis metrics, and stores
        them in a pandas DataFrame. The final DataFrame is saved to a CSV file.
        """
        results = []
        for idx, sample_dir in enumerate(tqdm(self.sample_dirs, desc=f"Analyzing with {self.lens_type} lens")):
            if idx>max_n_samples:
                break
            try:
                with open(os.path.join(sample_dir, "metadata.json"), 'r') as f:
                    meta = json.load(f)
                data = np.load(os.path.join(sample_dir, "states_and_logits.npz"))

                if self.states_key not in data:
                    continue

                hidden_states = torch.from_numpy(data[self.states_key]).to(self.device)
                final_logits = torch.from_numpy(data['final_logits']).to(self.device)

                with torch.no_grad():
                    if self.lenses: # For 'language' or 'vision' tuned lenses
                        # The lenses are float32, so cast the hidden states to float32
                        lens_logits = torch.stack([lens(hs.to(torch.float32)) for hs in hidden_states])
                    else: # For 'logit' lens
                        # The model's layers are float16, so cast the hidden states to float16
                        norm_states = self.final_norm(hidden_states.to(torch.float16))
                        lens_logits = self.lm_head(norm_states)

                gt_tokens = self.processor.tokenizer(str(meta['ground_truth']), add_special_tokens=False).input_ids
                gt_token_id = gt_tokens[0] if gt_tokens else None

                result_row = {
                    'domain': meta['domain'],
                    'flip_layer': self._calculate_flip_layer(lens_logits, final_logits),
                    'agreement_layer': self._calculate_ground_truth_agreement(lens_logits, gt_token_id),
                }
                kl_divs = self._calculate_kl_divergence(lens_logits, final_logits)
                top_k_accs = self._calculate_top_k_accuracy(lens_logits, gt_token_id, k=5)
                for i in range(self.num_layers):
                    result_row[f'kl_div_layer_{i}'] = kl_divs[i]
                    result_row[f'top_k_acc_layer_{i}'] = top_k_accs[i]
                results.append(result_row)

            except Exception as e:
                print(f"Warning: Failed to process {sample_dir}. Error: {e}")
                continue
        
        self.results_df = pd.DataFrame(results)
        output_path = os.path.join(self.output_dir, "full_analysis_results.csv")
        self.results_df.to_csv(output_path, index=False)
        print(f"\nFull analysis complete. Results data saved to {output_path}")
        return self.results_df

    # --- Individual Metric Calculation Methods ---
    def _get_top_token_id(self, logits):
        return torch.argmax(logits, dim=-1).item()

    def _calculate_flip_layer(self, lens_logits, final_logits):
        final_pred_id = self._get_top_token_id(final_logits)
        for i, layer_logits in enumerate(lens_logits):
            if self._get_top_token_id(layer_logits) == final_pred_id:
                return i
        return -1

    def _calculate_ground_truth_agreement(self, lens_logits, gt_token_id):
        if gt_token_id is None: return -1
        for i, layer_logits in enumerate(lens_logits):
            if self._get_top_token_id(layer_logits) == gt_token_id:
                return i
        return -1

    def _calculate_kl_divergence(self, lens_logits, final_logits):
        final_dist = F.log_softmax(final_logits, dim=-1)
        kl_divs = []
        for layer_logits in lens_logits:
            layer_dist = F.log_softmax(layer_logits, dim=-1)
            kl_div = F.kl_div(layer_dist.unsqueeze(0), final_dist.exp().unsqueeze(0), reduction='batchmean', log_target=False)
            kl_divs.append(kl_div.item())
        return kl_divs

    def _calculate_top_k_accuracy(self, lens_logits, gt_token_id, k=5):
        if gt_token_id is None: return [0] * len(lens_logits)
        accuracies = []
        for layer_logits in lens_logits:
            top_k_indices = torch.topk(layer_logits, k).indices
            accuracies.append(1 if gt_token_id in top_k_indices else 0)
        return accuracies

    # --- Individual Plotting Methods ---
    def plot_flip_layer(self):
        """
        Generates and saves a histogram of the 'flip layer'.
        The flip layer is the first layer where the lens's top prediction
        matches the final model's top prediction.
        """
        if self.results_df is None:
            print("Please run .run_and_save_all_metrics() first.")
            return

        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(12, 7))
        sns.histplot(data=self.results_df, x='flip_layer', hue='domain', multiple='stack', bins=self.num_layers)
        plt.title(f'Prediction Consistency (Flip Layer) - {self.lens_type.capitalize()} Lens')
        plt.xlabel('Layer Index')
        plt.ylabel('Number of Samples')
        save_path = os.path.join(self.output_dir, 'flip_layer_hist.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Flip layer plot saved to {save_path}")

    def plot_ground_truth_agreement(self):
        """
        Generates and saves a histogram of the ground truth agreement layer.
        This is the first layer where the lens's top prediction matches
        the ground truth answer.
        """
        if self.results_df is None:
            print("Please run .run_and_save_all_metrics() first.")
            return

        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(12, 7))
        sns.histplot(data=self.results_df, x='agreement_layer', hue='domain', multiple='stack', bins=self.num_layers)
        plt.title(f'Ground Truth Agreement Layer - {self.lens_type.capitalize()} Lens')
        plt.xlabel('Layer Index')
        plt.ylabel('Number of Samples')
        save_path = os.path.join(self.output_dir, 'agreement_layer_hist.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Agreement layer plot saved to {save_path}")

    def plot_kl_divergence(self):
        """
        Generates and saves a line plot of the KL divergence trajectory.
        This shows how the lens's predicted distribution at each layer
        converges towards the final model's predicted distribution.
        """
        if self.results_df is None:
            print("Please run .run_and_save_all_metrics() first.")
            return

        kl_cols = [f'kl_div_layer_{i}' for i in range(self.num_layers)]
        kl_df = self.results_df[['domain'] + kl_cols].melt(id_vars='domain', var_name='layer', value_name='kl_divergence')
        kl_df['layer'] = kl_df['layer'].str.extract('(\d+)').astype(int)
        
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(12, 7))
        sns.lineplot(data=kl_df, x='layer', y='kl_divergence', hue='domain', errorbar='sd')
        plt.title(f'KL Divergence Trajectory - {self.lens_type.capitalize()} Lens')
        plt.xlabel('Layer Index')
        plt.ylabel('Average KL Divergence to Final Prediction')
        save_path = os.path.join(self.output_dir, 'kl_divergence.png')
        plt.savefig(save_path)
        plt.close()
        print(f"KL divergence plot saved to {save_path}")

    def plot_top_k_accuracy(self):
        """
        Generates and saves a line plot of the layer-wise Top-K accuracy.
        This shows the percentage of samples where the ground truth answer
        is within the top K predictions of the lens at each layer.
        """
        if self.results_df is None:
            print("Please run .run_and_save_all_metrics() first.")
            return

        acc_cols = [f'top_k_acc_layer_{i}' for i in range(self.num_layers)]
        acc_df = self.results_df[['domain'] + acc_cols].melt(id_vars='domain', var_name='layer', value_name='accuracy')
        acc_df['layer'] = acc_df['layer'].str.extract('(\d+)').astype(int)

        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(12, 7))
        sns.lineplot(data=acc_df, x='layer', y='accuracy', hue='domain', errorbar='sd')
        plt.title(f'Layer-wise Top-5 Accuracy - {self.lens_type.capitalize()} Lens')
        plt.xlabel('Layer Index')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        save_path = os.path.join(self.output_dir, 'top_k_accuracy.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Top-K accuracy plot saved to {save_path}")


# --- NEW SELF-CONTAINED FUNCTION FOR QUALITATIVE INSPECTION ---

def inspect_domain_samples(lens_type, domain_path, top_k=5):
    """
    A self-contained function to apply a lens to all samples in a specific
    domain folder and print the qualitative results for each.

    This is meant for direct inspection of model behavior on a small set of samples
    and can be easily called from a notebook.

    Args:
        lens_type (str): 'language', 'vision', or 'logit'.
        domain_path (str): The full path to the domain folder to inspect
                           (e.g., 'ffn_states_output/com').
        top_k (int): The number of top predictions to show for each layer.
    """
    print(f"--- Starting Inspection for Domain: {domain_path} ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Setup: Determine keys, load model/lenses
    if lens_type in ['language', 'logit']:
        states_key = 'all_hidden_states'
    else: # vision
        states_key = 'vision_hidden_states'

    sample_dirs = sorted(glob.glob(os.path.join(domain_path, "sample_*")))
    if not sample_dirs:
        print(f"Error: No 'sample_*' directories found in {domain_path}")
        return

    with open(os.path.join(sample_dirs[0], "metadata.json"), 'r') as f:
        metadata = json.load(f)
    
    model_id = metadata['model_id']
    processor = AutoProcessor.from_pretrained(model_id)
    num_layers = metadata['num_lang_layers'] if lens_type in ['language', 'logit'] else metadata['num_vision_layers']
    
    data = np.load(os.path.join(sample_dirs[0], "states_and_logits.npz"))
    hidden_dim = data[states_key].shape[-1]
    vocab_size = data['final_logits'].shape[-1]

    lenses, lm_head, final_norm = None, None, None
    if lens_type in ['language', 'vision']:
        lens_file = 'tuned_lens.pt' if lens_type == 'language' else 'vision_lens.pt'
        # Assume lens file is two levels up from the domain path (e.g., ffn_states_output/com -> ffn_states_output/../lens.pt)
        lens_path = os.path.join(domain_path, '..', '..', lens_file)
        if not os.path.exists(lens_path):
            # Fallback for base directory
            lens_path_fallback = os.path.join(domain_path, '..', lens_file)
            if not os.path.exists(lens_path_fallback):
                 raise FileNotFoundError(f"Error: Lens file not found at {lens_path} or {lens_path_fallback}")
            lens_path = lens_path_fallback
        
        lenses = AnalysisRunner._load_lenses(None, lens_path, num_layers, hidden_dim, vocab_size)
        lenses.to(device)

    elif lens_type == 'logit':
        model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map='auto')
        final_norm = model.language_model.norm
        lm_head = model.get_output_embeddings()

    # Loop through each sample in the specified domain
    for sample_dir in sample_dirs:
        print(f"\n{'='*20} Analyzing Sample: {os.path.basename(sample_dir)} {'='*20}")
        with open(os.path.join(sample_dir, "metadata.json"), 'r') as f:
            meta = json.load(f)
        data = np.load(os.path.join(sample_dir, "states_and_logits.npz"))

        print(f"Question: {meta['prompt']}")
        print(f"Ground Truth: {meta['ground_truth']}")
        
        hidden_states = torch.from_numpy(data[states_key]).to(device)
        final_logits = torch.from_numpy(data['final_logits']).to(device)

        with torch.no_grad():
            if lenses: # For 'language' or 'vision' tuned lenses
                # The lenses are float32, so cast the hidden states to float32
                lens_logits = torch.stack([lens(hs.to(torch.float32)) for hs in hidden_states])
            else: # For 'logit' lens
                # The model's layers are float16, so cast the hidden states to float16
                norm_states = final_norm(hidden_states.to(torch.float16))
                lens_logits = lm_head(norm_states)

        # Print layer-by-layer predictions
        for i, layer_logits in enumerate(lens_logits):
            top_k_res = torch.topk(layer_logits, top_k)
            tokens = processor.tokenizer.convert_ids_to_tokens(top_k_res.indices)
            print(f"  Layer {i}: {tokens}")

        # Print final model prediction for comparison
        top_k_final = torch.topk(final_logits, top_k)
        tokens_final = processor.tokenizer.convert_ids_to_tokens(top_k_final.indices)
        print(f"  ---------------------------------")
        print(f"  Final Model Prediction: {tokens_final}")


if __name__ == '__main__':
    """
    Example of how to use the AnalysisRunner and the inspection function.
    """
    # --- QUANTITATIVE ANALYSIS EXAMPLE ---
    # 1. Choose the analysis type: 'language', 'vision', or 'logit'
    LENS_TYPE_TO_RUN = 'language'

    # 2. Initialize the runner for aggregate analysis
    analyzer = AnalysisRunner(lens_type=LENS_TYPE_TO_RUN)

    # 3. Run the main analysis loop to get summary statistics
    analyzer.run_and_save_all_metrics()

    # 4. Generate all plots from the calculated results
    if analyzer.results_df is not None:
        analyzer.plot_flip_layer()
        analyzer.plot_ground_truth_agreement()
        analyzer.plot_kl_divergence()
        analyzer.plot_top_k_accuracy()

    # --- QUALITATIVE INSPECTION EXAMPLE ---
    # To inspect the first 5 samples in the 'com' domain with the 'vision' lens:
    # inspect_domain_samples(lens_type='vision', domain_path='ffn_states_output/com', top_k=5)

