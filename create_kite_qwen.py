# coding=utf-8
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
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

    # Align tie_word_embeddings mapping
    kite_config.tie_word_embeddings = getattr(lm_config, "tie_word_embeddings", False)

    # Clear or copy quantization config based on the pre-trained language model config
    if getattr(lm_config, "quantization_config", None) is not None:
        kite_config.quantization_config = lm_config.quantization_config
    else:
        if hasattr(kite_config, "quantization_config"):
            delattr(kite_config, "quantization_config")

    # Align text hidden size in vision configuration to match language model embeddings size
    if hasattr(kite_config, "vision_config") and kite_config.vision_config is not None:
        kite_config.vision_config.text_hidden_size = lm_config.hidden_size

    # Ensure auto_map is set correctly for Hugging Face loader classes
    kite_config.auto_map = {
        "AutoConfig": "configuration_kite.KiteConfig",
        "AutoModel": "modeling_kite.KiteForConditionalGeneration",
        "AutoModelForCausalLM": "modeling_kite.KiteForConditionalGeneration"
    }

    # Check flash attention availability dynamically and fall back to eager if not available
    from transformers.utils import is_flash_attn_2_available
    if not is_flash_attn_2_available():
        print("Flash Attention 2 is not available. Setting vision_config and top-level attention implementation to eager...")
        if hasattr(kite_config, "vision_config") and kite_config.vision_config is not None:
            kite_config.vision_config._attn_implementation = "eager"
        if hasattr(kite_config, "text_config") and kite_config.text_config is not None:
            kite_config.text_config._attn_implementation = "eager"
        kite_config._attn_implementation = "eager"

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

    print("6.5. Adding special media tokens and resizing model embeddings...")
    special_tokens_dict = {
        "additional_special_tokens": [
            "<|im_end|>", "<|im_user|>", "<|im_assistant|>", 
            "<|start_header_id|>", "<|end_header_id|>", "[EOT]", 
            "<|im_system|>", "<|im_middle|>", "<|media_begin|>", 
            "<|media_content|>", "<|media_end|>", "<|media_pad|>"
        ]
    }
    num_added_toks = lm_tokenizer.add_special_tokens(special_tokens_dict)
    print(f"Added {num_added_toks} special tokens to the tokenizer.")
    
    # Resize embeddings
    model.resize_token_embeddings(len(lm_tokenizer))
    
    # Update config's placeholder token id
    media_pad_token_id = lm_tokenizer.convert_tokens_to_ids("<|media_pad|>")
    model.config.media_placeholder_token_id = media_pad_token_id
    if hasattr(model.config, "vision_config") and model.config.vision_config is not None:
        if isinstance(model.config.vision_config, dict):
            model.config.vision_config["media_placeholder_token_id"] = media_pad_token_id
            model.config.vision_config["text_hidden_size"] = lm_config.hidden_size
        else:
            model.config.vision_config.media_placeholder_token_id = media_pad_token_id
            model.config.vision_config.text_hidden_size = lm_config.hidden_size
            
    print(f"Set media_placeholder_token_id to: {media_pad_token_id}")

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
