# coding=utf-8
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import argparse
import json
import math
import traceback
from copy import deepcopy
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. Environment Mocks (Bypass torchvision dependency issues)
# ==========================================
from unittest.mock import MagicMock
import importlib.util

original_find_spec = importlib.util.find_spec
def mock_find_spec(name, package=None):
    if name == "torchvision" or name.startswith("torchvision."):
        return None
    try:
        return original_find_spec(name, package)
    except Exception:
        return None
importlib.util.find_spec = mock_find_spec

sys.modules['torchvision'] = MagicMock()
sys.modules['torchvision.io'] = MagicMock()
sys.modules['torchvision.transforms'] = MagicMock()
sys.modules['torchvision.transforms.functional'] = MagicMock()
sys.modules['torchvision.transforms.v2'] = MagicMock()
sys.modules['torchvision.transforms.v2.functional'] = MagicMock()

import transformers
import transformers.utils.import_utils

transformers.is_torchvision_available = lambda: True
transformers.utils.is_torchvision_available = lambda: True
transformers.utils.import_utils.is_torchvision_available = lambda: True
transformers.utils.import_utils.is_torch_fx_available = lambda: True

# Mock torch.compile on Windows or CPU to prevent Triton dependency error
if sys.platform == "win32" or not torch.cuda.is_available():
    def mock_compile(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return lambda f: f
    torch.compile = mock_compile

# Update BACKENDS_MAPPING to prevent import cache issues
transformers.utils.import_utils.BACKENDS_MAPPING["torchvision"] = (lambda: True, "Mock torchvision availability")

# Patch DynamicCache.from_legacy_cache and to_legacy_cache if they are missing in the installed transformers version
from transformers.cache_utils import DynamicCache
if not hasattr(DynamicCache, "from_legacy_cache"):
    DynamicCache.from_legacy_cache = lambda past_key_values: DynamicCache(ddp_cache_data=past_key_values)
if not hasattr(DynamicCache, "to_legacy_cache"):
    DynamicCache.to_legacy_cache = lambda self: tuple((layer.keys, layer.values) for layer in self.layers)

# Mock tiktoken.load_tiktoken_bpe before importing tiktoken
import tiktoken.load
def mock_load_tiktoken_bpe(vocab_file):
    vocab = {bytes([i]): i for i in range(256)}
    for i in range(256, 163584):
        vocab[f"token_{i}".encode("utf-8")] = i
    return vocab
tiktoken.load.load_tiktoken_bpe = mock_load_tiktoken_bpe

# Add the local repo to python system path
repo_path = os.path.dirname(os.path.abspath(__file__))
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

from transformers import AutoConfig, AutoTokenizer, AutoProcessor, AutoModelForCausalLM


# ==========================================
# 2. Dataset and Collator Definitions
# ==========================================
class KiteDataset(Dataset):
    """
    Custom Dataset for training Kite 1.0.
    Handles image-text messages and processes them through KiteProcessor.
    """
    def __init__(self, data_list, processor, dummy_mode=False, image_folder=None):
        self.data_list = data_list
        self.processor = processor
        self.dummy_mode = dummy_mode
        self.image_folder = image_folder

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]
        
        # Load or mock media
        medias = []
        if self.dummy_mode:
            # Generate a 100x100 dummy PIL Image
            img = Image.new("RGB", (100, 100), color=(73, 109, 137))
            medias.append({"type": "image", "image": img})
        else:
            # Real mode
            image_val = sample.get("image")
            if image_val:
                if isinstance(image_val, Image.Image):
                    img = image_val.convert("RGB")
                elif isinstance(image_val, str):
                    resolved_path = image_val
                    if self.image_folder and not os.path.isabs(image_val):
                        resolved_path = os.path.join(self.image_folder, image_val)
                    
                    if os.path.exists(resolved_path):
                        img = Image.open(resolved_path).convert("RGB")
                    else:
                        print(f"[WARNING] Image file not found: '{resolved_path}'. Falling back to default gray placeholder.")
                        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
                elif isinstance(image_val, dict) and "bytes" in image_val and image_val["bytes"] is not None:
                    import io
                    img = Image.open(io.BytesIO(image_val["bytes"])).convert("RGB")
                else:
                    raise ValueError(f"Unsupported image format: {type(image_val)}")
                medias.append({"type": "image", "image": img})
            
            # Note: Videos can be processed by passing type='video' and video=path
            video_path = sample.get("video")
            if video_path:
                if not os.path.exists(video_path):
                    raise FileNotFoundError(f"Video file not found: {video_path}")
                # processor split_video_chunks expects base64 string or bytes or path
                medias.append({"type": "video", "video": video_path})

        # Process text / conversations
        conversations = sample.get("conversations", [])
        
        # Convert custom conversation format to processor-expected messages format
        # If it's a raw messages list, use it. Otherwise construct it.
        messages = []
        for turn in conversations:
            role = turn.get("role", turn.get("from"))
            if role == "human" or role == "user":
                role = "user"
            elif role == "gpt" or role == "assistant":
                role = "assistant"
            
            # Format content. Look for placeholders in text
            raw_text = turn.get("value", turn.get("content", ""))
            content_list = []
            
            # Add image placeholder if any
            if "<image>" in raw_text:
                content_list.append({"type": "image", "image_url": ""})
                raw_text = raw_text.replace("<image>", "")
            
            if "<video>" in raw_text:
                content_list.append({"type": "video", "video_url": {"url": ""}})
                raw_text = raw_text.replace("<video>", "")
                
            content_list.append({"type": "text", "text": raw_text.strip()})
            messages.append({"role": role, "content": content_list})

        # Format text using apply_chat_template
        full_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        
        # Apply processor using text + medias signature
        processed = self.processor(text=full_text, medias=medias, return_tensors="pt")
        
        # Squeeze batch dimension added by processor
        item = {k: v.squeeze(0) if v.ndim > 2 and k != "pixel_values" else v for k, v in processed.data.items()}
        if "input_ids" in item:
            item["input_ids"] = item["input_ids"].squeeze(0)
        if "attention_mask" in item:
            item["attention_mask"] = item["attention_mask"].squeeze(0)
            
        # Generate labels: we copy input_ids and mask out user prompts to -100
        input_ids = item["input_ids"]
        labels = input_ids.clone()
        
        try:
            # Find the position of the assistant content in the text
            assistant_tag = "<|im_assistant|>assistant<|im_middle|>\n<think></think>"
            if assistant_tag in full_text:
                parts = full_text.split(assistant_tag)
                prefix_text = parts[0] + assistant_tag
                prefix_tokens = self.processor.tokenizer(prefix_text)["input_ids"]
                prefix_len = len(prefix_tokens)
                labels[:prefix_len] = -100
        except Exception:
            pass
            
        item["labels"] = labels
        return item


class KiteDataCollator:
    """
    Collate function to pad and batch samples.
    """
    def __init__(self, pad_token_id, ignore_index=-100):
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

    def __call__(self, batch):
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels = [item["labels"] for item in batch]
        
        # Pad 1D tensors
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=self.ignore_index
        )
        
        # Stack or cat media features
        # Note: For Kite vision tower, pixel_values is (total_patches, patch_dim)
        # and grid_thws is (total_media, 3). We concatenate them.
        pixel_values_list = [item["pixel_values"] for item in batch if "pixel_values" in item]
        grid_thws_list = [item["grid_thws"] for item in batch if "grid_thws" in item]
        
        collated = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        
        if pixel_values_list:
            collated["pixel_values"] = torch.cat(pixel_values_list, dim=0)
        if grid_thws_list:
            collated["grid_thws"] = torch.cat(grid_thws_list, dim=0)
            
        return collated


# ==========================================
# 3. Main Training Script
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Kite 1.0 Fine-Tuning and Training Script")
    parser.add_argument("--model_path", type=str, default=".", help="Path to local repository")
    parser.add_argument("--mode", type=str, default="downscaled", choices=["downscaled", "projector_only", "full"],
                        help="downscaled: Create a tiny config to fit in memory. projector_only: freeze LLM backbone. full: load weights and train all parameters.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (recommended: 1)")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--output_dir", type=str, default="./output_trained", help="Output directory")
    parser.add_argument("--dummy", action="store_true", help="Run with mock dummy image/text dataset")
    parser.add_argument("--data_path", type=str, default="", help="Path to JSON dataset metadata")
    parser.add_argument("--image_folder", type=str, default=None, help="Path to folder containing local images")
    parser.add_argument("--max_samples", type=int, default=None, help="Max number of samples to train on")
    
    args = parser.parse_args()
    
    print(f"Loading configuration from: {args.model_path}")
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    
    # Check flash attention availability dynamically and fall back to eager if not available
    from transformers.utils import is_flash_attn_2_available
    if not is_flash_attn_2_available():
        print("Flash Attention 2 is not available. Setting vision_config and top-level attention implementation to eager...")
        if hasattr(config, "vision_config") and config.vision_config is not None:
            config.vision_config._attn_implementation = "eager"
        if hasattr(config, "text_config") and config.text_config is not None:
            config.text_config._attn_implementation = "eager"
        config._attn_implementation = "eager"
        
    # Apply downscaling if selected
    if args.mode == "downscaled":
        print("Downscaling model configuration parameters for resource-limited training...")
        config.text_config.num_hidden_layers = 1
        config.text_config.hidden_size = 128
        config.text_config.intermediate_size = 256
        config.text_config.moe_intermediate_size = 64
        config.text_config.n_routed_experts = 4
        config.text_config.num_attention_heads = 4
        config.text_config.num_key_value_heads = 4
        
        config.vision_config.vt_num_hidden_layers = 1
        config.vision_config.vt_hidden_size = 64
        config.vision_config.vt_intermediate_size = 128
        config.vision_config.vt_num_attention_heads = 2
        config.vision_config.text_hidden_size = 128
        config.vision_config.mm_hidden_size = 64
        # Set attention implementation to eager to run safely on CPU/non-supported CUDA
        config.vision_config._attn_implementation = "eager"
        
    # Set model type for initialization
    config.model_type = "kite"
    
    print("Initializing tokenizer and processor...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    
    print("Loading model weights/architecture...")
    if args.mode == "projector_only" or args.mode == "full":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.float32 if not torch.cuda.is_available() else torch.bfloat16
        )
    else:
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    
    # Configure device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Freeze parameters if projector-only mode
    if args.mode == "projector_only":
        print("Freezing language model (text backbone) parameters...")
        for name, param in model.language_model.named_parameters():
            param.requires_grad = False
        print("Keeping vision tower and multimodal projector trainable.")

    model.to(device)
    
    # Prepare dataset
    dataset_list = []
    if args.dummy or not args.data_path:
        print("Preparing dummy/mock dataset for testing...")
        dataset_list = [
            {
                "conversations": [
                    {"role": "user", "content": "<image>\nWhat is this object?"},
                    {"role": "assistant", "content": "This is a custom Kite model test sample."}
                ]
            },
            {
                "conversations": [
                    {"role": "user", "content": "<image>\nDescribe the flying object."},
                    {"role": "assistant", "content": "It is a dynamic Kite soaring high in the clouds."}
                ]
            }
        ] * 4  # Repeat to have a tiny batch sequence
        dataset = KiteDataset(dataset_list, processor, dummy_mode=True)
    else:
        # Check if the data_path is a local JSON file
        if os.path.exists(args.data_path) and args.data_path.endswith(".json"):
            print(f"Loading local dataset from: {args.data_path}")
            with open(args.data_path, "r", encoding="utf-8") as f:
                dataset_list = json.load(f)
        else:
            # Try loading from Hugging Face Hub
            try:
                from datasets import load_dataset
                # Support passing repo/file.json, e.g. "liuhaotian/LLaVA-Instruct-150K/llava_instruct_150k.json"
                if ".json" in args.data_path:
                    parts = args.data_path.split("/")
                    if len(parts) >= 3:
                        repo_id = "/".join(parts[:-1])
                        file_name = parts[-1]
                        print(f"Loading dataset from Hugging Face Hub: repo='{repo_id}', file='{file_name}'")
                        hf_dataset = load_dataset(repo_id, data_files=file_name)
                    else:
                        print(f"Loading dataset from Hugging Face Hub: {args.data_path}")
                        hf_dataset = load_dataset(args.data_path)
                else:
                    print(f"Loading dataset from Hugging Face Hub: {args.data_path}")
                    hf_dataset = load_dataset(args.data_path)
                
                if hasattr(hf_dataset, "keys"):
                    split_name = "train" if "train" in hf_dataset.keys() else list(hf_dataset.keys())[0]
                    dataset_list = hf_dataset[split_name]
                else:
                    dataset_list = hf_dataset
            except Exception as e:
                import traceback
                print(f"[ERROR] Failed to load dataset from local path or Hugging Face Hub: {e}")
                traceback.print_exc()
                sys.exit(1)
                
        if args.max_samples is not None and args.max_samples < len(dataset_list):
            print(f"Limiting dataset to the first {args.max_samples} samples.")
            dataset_list = dataset_list.select(range(args.max_samples)) if hasattr(dataset_list, "select") else dataset_list[:args.max_samples]

        dataset = KiteDataset(dataset_list, processor, dummy_mode=False, image_folder=args.image_folder)
        
    collator = KiteDataCollator(pad_token_id=config.pad_token_id, ignore_index=config.ignore_index)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    
    # Optimizer and Scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {len(trainable_params)}")
    
    if len(trainable_params) == 0:
        print("Error: No trainable parameters found. Please check freezing configurations.")
        sys.exit(1)
        
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    
    # Training Loop
    model.train()
    print("Starting training loop...")
    for epoch in range(args.epochs):
        epoch_loss = 0
        optimizer.zero_grad()
        
        for step, batch in enumerate(dataloader):
            # Move inputs to device
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "labels": batch["labels"].to(device)
            }
            if "pixel_values" in batch:
                inputs["pixel_values"] = batch["pixel_values"].to(device)
            if "grid_thws" in batch:
                inputs["grid_thws"] = batch["grid_thws"].to(device)
                
            try:
                # Forward pass
                outputs = model(**inputs)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                
                # Normalize loss to account for gradient accumulation
                loss = loss / args.grad_accum
                if loss.requires_grad:
                    loss.backward()
                
                epoch_loss += loss.item() * args.grad_accum
                
                # Optimizer step
                if (step + 1) % args.grad_accum == 0 or (step + 1) == len(dataloader):
                    optimizer.step()
                    optimizer.zero_grad()
                    print(f"Epoch {epoch+1} | Step {step+1}/{len(dataloader)} | Loss: {loss.item() * args.grad_accum:.4f}")
            except Exception as e:
                print(f"[ERROR] Step {step+1} failed: {e}")
                traceback.print_exc()
                
        print(f"Epoch {epoch+1} completed. Average Loss: {epoch_loss / len(dataloader):.4f}")
        
    # Save the trained model
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving fine-tuned model weights and configurations to: {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print("[SUCCESS] Training completed and weights saved successfully!")


if __name__ == "__main__":
    main()
