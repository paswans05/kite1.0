# coding=utf-8
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import argparse
from PIL import Image
import torch

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

def main():
    parser = argparse.ArgumentParser(description="Kite 1.0 Testing and Inference Script")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model checkpoint directory")
    parser.add_argument("--image_path", type=str, default="", help="Path to image file (optional, will use dummy if empty)")
    parser.add_argument("--prompt", type=str, default="What is in this image?", help="Prompt question for the model")
    
    args = parser.parse_args()
    
    # Predefined sample text-only questions
    sample_text_questions = [
        "Explain what gravity is in simple terms.",
        "What is the capital of France?",
        "Write a short poem about a kite.",
        "How does a visual-language model work when no image is provided?",
        "Give me a recipe for chocolate chip cookies."
    ]
    
    if not args.image_path:
        if args.prompt == "What is in this image?":
            print(f"No image provided. Switching to default text prompt: '{sample_text_questions[0]}'")
            args.prompt = sample_text_questions[0]
        print("\nPredefined sample text-only questions you can try using --prompt:")
        for q in sample_text_questions:
            print(f"  - --prompt \"{q}\"")
        print()
    
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
        
    # Remove quantization config if loading unquantized weights (like downscaled/fine-tuned checkpoints)
    if hasattr(config, "quantization_config"):
        delattr(config, "quantization_config")
    if hasattr(config, "text_config") and config.text_config is not None:
        if hasattr(config.text_config, "quantization_config"):
            delattr(config.text_config, "quantization_config")
        
    print("Initializing tokenizer and processor...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    
    print("Loading model weights...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        config=config, 
        trust_remote_code=True,
        torch_dtype=torch.float32 if not torch.cuda.is_available() else torch.bfloat16
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Moving model to device: {device}")
    model.to(device)
    model.eval()
    
    # Load or generate image
    medias = []
    has_image = False
    
    if args.image_path:
        has_image = True
        img_to_load = None
        if args.image_path.startswith("http://") or args.image_path.startswith("https://"):
            print(f"Downloading image from URL: {args.image_path}")
            import urllib.request
            import tempfile
            try:
                temp_dir = tempfile.gettempdir()
                temp_path = os.path.join(temp_dir, "temp_test_image.jpg")
                req = urllib.request.Request(
                    args.image_path, 
                    headers={'User-Agent': 'Mozilla/5.0'}
                )
                with urllib.request.urlopen(req) as response:
                    with open(temp_path, "wb") as f:
                        f.write(response.read())
                img_to_load = temp_path
            except Exception as e:
                print(f"[WARNING] Failed to download online image: {e}. Falling back to dummy image.")
                img_to_load = None
        else:
            img_to_load = args.image_path
            
        if img_to_load:
            if not os.path.exists(img_to_load):
                raise FileNotFoundError(f"Image not found: {img_to_load}")
            print(f"Loading image from: {img_to_load}")
            img = Image.open(img_to_load).convert("RGB")
            medias.append({"type": "image", "image": img})
        else:
            print("Generating a dummy image for verification...")
            img = Image.new("RGB", (100, 100), color=(73, 109, 137))
            medias.append({"type": "image", "image": img})
    else:
        print("No image path provided. Running in text-only mode (without image).")
        
    # Format message with <image> placeholder only if we have an image
    content_list = []
    if has_image:
        content_list.append({"type": "image", "image_url": ""})
    content_list.append({"type": "text", "text": args.prompt})
    
    messages = [{"role": "user", "content": content_list}]
    
    # Format text using apply_chat_template
    full_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    print(f"Formatted prompt text:\n{full_text}")
    print("Running processor...")
    # Pass medias (which is empty [] for text-only mode) to avoid processor validation error
    inputs = processor(text=full_text, medias=medias, return_tensors="pt")
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    print("Generating model response...")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=100,
            use_cache=True
        )
        
    # Decode and print generated text
    # Slice out prompt tokens to print only the new assistant response
    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = output_ids[0][prompt_len:]
    
    response = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
    print("\n" + "="*40)
    print("MODEL RESPONSE:")
    print("="*40)
    print(response.strip())
    print("="*40)

if __name__ == "__main__":
    main()
