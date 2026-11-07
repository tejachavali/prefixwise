"""Cases from the agent compatibility check (Muse, 2026-11-07), replayed."""

from prefixwise.proxy import PrefixwiseProxy


def scripted(payloads):
    p = PrefixwiseProxy(upstream_url="http://unused", api_key="k")
    sent, queue = [], list(payloads)

    def fake(provider, body, request_id, session_id, client_headers, api="chat"):
        sent.append(dict(body))
        return 200, queue.pop(0)

    p._forward = fake
    return p, sent


def test_mixing_chat_completions_and_responses_in_one_session_chains_only_same_api_ids():
    # The OpenAI SDK script ran Chat Completions, then the Responses API, same model:
    # chaining the chatcmpl- id into prompt_cache_options got a 400 from OpenAI.
    chat = {"id": "chatcmpl-EhUy", "object": "chat.completion",
            "usage": {"prompt_tokens": 27, "prompt_tokens_details": {"cached_tokens": 0}}}
    resp = lambda rid: {"id": rid, "object": "response",
                        "usage": {"input_tokens": 27, "input_tokens_details": {"cached_tokens": 0}}}
    p, sent = scripted([chat, resp("resp_1"), resp("resp_2")])
    h = {"x-prefixwise-session-id": "s"}
    msgs = [{"role": "user", "content": "hi"}]
    p.handle("openai", {"model": "gpt-5.6-luna", "messages": msgs}, h)
    p.handle("openai", {"model": "gpt-5.6-luna", "input": "hi"}, h, api="responses")
    assert "prompt_cache_options" not in sent[1]  # no foreign chatcmpl- id
    p.handle("openai", {"model": "gpt-5.6-luna", "input": "hi again"}, h, api="responses")
    assert sent[2]["prompt_cache_options"] == {"comparison_response_id": "resp_1"}  # same API: chained


def test_a_tiny_anthropic_prompt_is_below_the_minimum_not_a_misplaced_breakpoint():
    # The Anthropic SDK script: 33 then 214 input tokens, no cache_control. Nothing
    # could have been cached, so there is no waste and no breakpoint advice.
    usage = lambda n: {"id": f"msg_{n}", "type": "message", "content": [],
                       "usage": {"input_tokens": n, "cache_read_input_tokens": 0,
                                 "cache_creation_input_tokens": 0, "output_tokens": 5}}
    p, _ = scripted([usage(33), usage(214)])
    h = {"x-prefixwise-session-id": "s"}
    msgs = [{"role": "user", "content": "hi"}]
    p.handle("anthropic", {"model": "claude-haiku-5-5", "max_tokens": 10, "messages": msgs}, h)
    res = p.handle("anthropic", {"model": "claude-haiku-5-5", "max_tokens": 10,
                                 "messages": msgs + [{"role": "assistant", "content": "x"},
                                                     {"role": "user", "content": "more"}]}, h)[0]
    assert res["cause"] == "below_min_prefix" and res["avoidable_tokens"] == 0


def test_anthropic_minimums_by_model():
    from prefixwise.classifier import anthropic_min_prefix

    assert anthropic_min_prefix("claude-haiku-5-5") == 512
    assert anthropic_min_prefix("claude-sonnet-4-6") == 1024
    assert anthropic_min_prefix("claude-opus-4-5-20251101") == 4096
    assert anthropic_min_prefix("some-new-claude") == 512  # the floor: never over-claims
