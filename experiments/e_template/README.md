# E-template: what the chat template does to a conversation's cached prefix

**Question.** A client re-sends the same message blocks every turn; a self-hosted engine renders
them through the model's chat template. Where the rendering changes between turns, the token
prefix changes although the request did not. How often does that break the prefix (Prefixwise's
`unknown`, "not an eviction": the block where the read stopped is still on the GPU), and what does
it cost when it only re-renders the tail?

**Method.** Per variant, a session of six turns through the real Prefixwise proxy (in-process) with
the engine's KV events on; per turn, `cached_tokens` is compared with the previous prompt (what any
cache can serve) and with the previous prompt plus output (what an engine that caches decoded blocks
could serve). Variants: the answer appended as returned (`kept`), the `<think>` block stripped
client-side (`stripped`), thinking off (`nothink`), thinking toggled each turn (`toggle`, a parameter
change), and a tool-call loop (`tools`). `--cpu-template MODEL` renders the template alone and
diffs token ids, naming the clause behind each divergence; no server.

**Prediction (Qwen3, from the CPU rendering 2026-11-09).** The template drops earlier turns'
`<think>` blocks and re-renders their content at a shifted position, so the shared prefix with the
previous turn's prompt+output is exactly the previous prompt: reuse `prompt` on every turn after
the first, never `break`. The previous output is re-prefilled every turn (the lost tail), a cost
that is a clean `none` for Prefixwise and invisible in cache-hit shares. `toggle` is
`param_rendered` (chat_template_kwargs is a compared parameter since 0.10).

## Run on Colab (L4, Qwen3-4B)

```bash
pip install -e ".[engines]" "vllm>=0.30"
MODEL=Qwen/Qwen3-4B ./experiments/e1_open/vllm_server.sh &        # publishes KV events on 5557
python experiments/e_template/e_template.py --engine vllm --url http://127.0.0.1:8000 --model Qwen/Qwen3-4B \
    --kv-events tcp://127.0.0.1:5557
# SGLang (Python 3.12 venv, as in experiments/e_evict/README.md)
MODEL=Qwen/Qwen3-4B KV_EVENTS=1 ./experiments/e1_open/sglang_server.sh &
python experiments/e_template/e_template.py --engine sglang --url http://127.0.0.1:30000 --model Qwen/Qwen3-4B \
    --kv-events tcp://127.0.0.1:5557 --full-attention --block 1
# the template alone (any runtime, CPU)
pip install transformers && python experiments/e_template/e_template.py --cpu-template Qwen/Qwen3-4B
```

Mock (no GPU): `python experiments/e_template/e_template.py --mock`.
