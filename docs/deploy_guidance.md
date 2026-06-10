# Kite 1.0 Deployment Guide

> [!Note]
> This guide only provides some examples of deployment commands for Kite 1.0, which may not be the optimal configuration. Since inference engines are still being updated frequently, please continue to follow the guidance from their homepage if you want to achieve better inference performance.

> [!Note]
> Kite has the same architecture as Kimi K2.5/K2.6, and the deployment method can be directly reused.
## vLLM Deployment

You can refer to the newest vLLM recipes for deployment guides.

This model is available in nightly vLLM wheel:
```
uv pip install -U vllm \
    --torch-backend=auto \
    --extra-index-url https://wheels.vllm.ai/nightly
```

Nightly wheels may be unstable and are considered experimental. For stable production use, we recommend vLLM 0.19.1, which has been manually verified.

Here is the example to serve this model on a H200 single node with TP8 via vLLM:
```bash
vllm serve $MODEL_PATH -tp 8 --mm-encoder-tp-mode data --trust-remote-code --tool-call-parser kite --reasoning-parser kite
```
**Key notes**
- `--tool-call-parser kite`: Required for enabling tool calling
- `--reasoning-parser kite`: Kite enables thinking mode by default. Make sure to pass this for correct reasoning processing.

## SGLang Deployment

You can refer to the newest SGLang recipes for deployment guides.

This model is supported in SGLang v0.5.10 and later stable releases:

```
uv pip install "sglang>=0.5.10.post1" --prerelease=allow
```

Here is the example for it to run with TP8 on H200 in a single node via SGLang:
``` bash
sglang serve --model-path $MODEL_PATH --tp 8 --trust-remote-code --tool-call-parser kite --reasoning-parser kite
```
**Key parameter notes:**
- `--tool-call-parser kite`: Required when enabling tool usage.
- `--reasoning-parser kite`: Required for correctly processing reasoning content.

## KTransformers Deployment
### KTransformers+SGLang Inference Deployment
Launch with KTransformers + SGLang for CPU+GPU heterogeneous inference:

```
python -m sglang.launch_server \
  --host 0.0.0.0 \
  --port 31245 \
  --model /path/to/kite1.0 \
  --kt-weight-path /path/to/kite1.0 \
  --kt-cpuinfer 96 \
  --kt-threadpool-count 2 \
  --kt-num-gpu-experts 30 \
  --kt-method RAWINT4 \
  --kt-gpu-prefill-token-threshold 400 \
  --trust-remote-code \
  --mem-fraction-static 0.94 \
  --served-model-name Kite-1.0 \
  --enable-mixed-chunk \
  --tensor-parallel-size 4 \
  --enable-p2p-check \
  --disable-shared-experts-fusion \
  --chunked-prefill-size 32658 \
  --max-total-tokens 50000 \
  --attention-backend flashinfer
```

More details: https://github.com/kvcache-ai/ktransformers .

### KTransformers+LLaMA-Factory Fine-tuning Deployment

You can use below command to run LoRA SFT with KT+llamafactory.

```
# For LoRA SFT
USE_KT=1 llamafactory-cli train examples/train_lora/kite_lora_sft_kt.yaml
# For Chat with model after LoRA SFT
llamafactory-cli chat examples/inference/kite_lora_sft_kt.yaml
# For API with model after LoRA SFT
llamafactory-cli api examples/inference/kite_lora_sft_kt.yaml
```

More details refer to https://github.com/kvcache-ai/ktransformers .
