"""A chat template's arguments are prompt-affecting parameters: a self-hosted engine renders the
prompt through the template, so `chat_template_kwargs` (enable_thinking on Qwen3) or a per-request
`chat_template` changes the rendered prefix although no block changed. Before, such a turn had
nothing to blame and read as `unknown` ("not an eviction") behind an engine with KV events."""

from prefixwise.classifier import MissCause, RequestSnapshot, classify
from prefixwise.explain import Turn, explain_pair
from prefixwise.fingerprint import Fingerprinter
from prefixwise.records import PARAM_KEYS, param_fingerprint

FP = Fingerprinter()
BLOCKS = FP.fingerprint_blocks([("system", "Be careful."), ("messages", {"role": "user", "content": "hi"})])


def snap(prompt, cached, **params):
    return RequestSnapshot(model="m", provider="openai", blocks=BLOCKS, prompt_tokens=prompt, cached_tokens=cached,
                           engine=True, params={k: param_fingerprint(v) for k, v in params.items()})


def test_chat_template_kwargs_and_chat_template_are_compared():
    assert "chat_template_kwargs" in PARAM_KEYS and "chat_template" in PARAM_KEYS
    prev = snap(4000, 0, chat_template_kwargs={"enable_thinking": True})
    cold = snap(4200, 0, chat_template_kwargs={"enable_thinking": False})
    c = classify(prev, cold)
    assert c.cause is MissCause.PARAM_RENDERED and "chat_template_kwargs" in c.evidence
    # the cache absorbed it (the tail moved, the prefix did not): a hit, the change kept as secondary
    hit = snap(4200, 4000, chat_template_kwargs={"enable_thinking": False})
    c = classify(prev, hit)
    assert c.cause is MissCause.NONE and MissCause.PARAM_RENDERED in c.secondary
    # a per-request template
    assert classify(snap(4000, 0), snap(4200, 0, chat_template="{{ messages }}")).cause is MissCause.PARAM_RENDERED


def test_explain_sees_the_same_parameters():
    body = {"model": "m", "messages": [{"role": "system", "content": "x " * 300}, {"role": "user", "content": "hi"}]}
    a = Turn(body={**body, "chat_template_kwargs": {"enable_thinking": True}},
             usage={"prompt_tokens": 4000, "prompt_tokens_details": {"cached_tokens": 0}})
    b = Turn(body={**body, "chat_template_kwargs": {"enable_thinking": False}},
             usage={"prompt_tokens": 4000, "prompt_tokens_details": {"cached_tokens": 0}})
    assert explain_pair(a, b, "vllm").cause == "param_rendered"
