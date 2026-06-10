# coding=utf-8
import os
import sys
import argparse
import shutil
import torch

# Add local path to sys.path to load local custom config/model files
repo_path = os.path.dirname(os.path.abspath(__file__))
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from configuration_kite import KiteConfig
from modeling_kite import KiteForConditionalGeneration

def main():
    parser = argparse.ArgumentParser(description="Assemble Kite Multimodal Model with a pre-trained Qwen/Llama text backbone")
    parser.add_argument("--lm_model_id", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", 
                        help="Hugging Face model ID for the pre-trained language model backbone")
    parser.add_argument("--output_dir", type=str, default="./kite_qwen_base", 
                        help="Output directory to save the assembled model configuration and weights")
    args = parser.parse_args()

    print(f"1. Loading base Kite configuration from: config.json")
    kite_config = KiteConfig.from_pretrained(".", trust_remote_code=True)

    print(f"2. Loading pre-trained language model configuration from: {args.lm_model_id}")
    lm_config = AutoConfig.from_pretrained(args.lm_model_id)
    lm_tokenizer = AutoTokenizer.from_pretrained(args.lm_model_id)

    print("3. Merging configurations...")
    # Replace the text configuration with Qwen's config
    kite_config.text_config = lm_config
    
    # Align token ID mappings
    kite_config.bos_token_id = lm_tokenizer.bos_token_id
    kite_config.eos_token_id = lm_tokenizer.eos_token_id
    kite_config.pad_token_id = lm_tokenizer.pad_token_id if lm_tokenizer.pad_token_id is not None else lm_tokenizer.eos_token_id
    
    # Update vocabulary size in top-level config
    kite_config.vocab_size = lm_config.vocab_size

    # Ensure auto_map is set correctly for Hugging Face loader classes
    kite_config.auto_map = {
        "AutoConfig": "configuration_kite.KiteConfig",
        "AutoModel": "modeling_kite.KiteForConditionalGeneration",
        "AutoModelForCausalLM": "modeling_kite.KiteForConditionalGeneration"
    }

    print("4. Initializing model architecture (Kite VLM)...")
    # Initialize the model structure (vision tower and projector are randomly initialized)
    model = KiteForConditionalGeneration(kite_config)

    print(f"5. Loading pre-trained language backbone weights from: {args.lm_model_id}")
    lm_model = AutoModelForCausalLM.from_pretrained(
        args.lm_model_id, 
        torch_dtype=torch.float32, # CPU-safe loading
        low_cpu_mem_usage=True
    )

    print("6. Copying pre-trained language model weights into the VLM backbone...")
    # The architecture of model.language_model matches lm_model exactly because they share the same config
    model.language_model.load_state_dict(lm_model.state_dict(), strict=True)
    print("[SUCCESS] Language model weights successfully copied!")

    print(f"7. Saving assembled model and configuration to: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save model weights and configuration
    model.save_pretrained(args.output_dir)
    lm_tokenizer.save_pretrained(args.output_dir)

    # Copy files required by the processor and preprocessor
    config_files = ["preprocessor_config.json", "chat_template.jinja"]
    for file_name in config_files:
        src = os.path.join(".", file_name)
        dst = os.path.join(args.output_dir, file_name)
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"Copied: {file_name}")

    # Copy modeling/configuration python files so trust_remote_code=True works dynamically
    py_files = [
        "modeling_kite.py", "configuration_kite.py", 
        "modeling_deepseek.py", "configuration_deepseek.py", 
        "tokenization_kite.py", "kite_processor.py", 
        "kite_vision_processing.py", "media_utils.py"
    ]
    for file_name in py_files:
        src = os.path.join(".", file_name)
        dst = os.path.join(args.output_dir, file_name)
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"Copied python module: {file_name}")

    print("="*50)
    print("[FINISHED] Model successfully assembled!")
    print(f"You can now train this model by running:")
    print(f"python train_kite.py --model_path {args.output_dir} --mode projector_only --output_dir ./kite_qwen_trained")
    print("="*50)

if __name__ == "__main__":
    main()
