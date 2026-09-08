"""The worker's half of tool calling: getting the schemas into the prompt.

The model emits tool calls only because its chat template rendered the tool
schemas into the prompt it was trained on. If `tools` does not reach
apply_chat_template, everything downstream is correct and the model still
never calls anything — so these tests are about that one argument.
"""

import torch

from tests.test_worker_dispatch import make_worker, StubModel


class ToolAwareTokenizer:
    """A modern template: accepts `tools` and records what it was given."""

    eos_token_id = 99

    def __init__(self):
        self.tools = None
        self.messages = None

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=True, tools=None
    ):
        self.messages = messages
        self.tools = tools
        return "rendered"

    def __call__(self, text, return_tensors=None):
        return {"input_ids": torch.tensor([[1, 2, 3]])}

    def decode(self, ids):
        return f"<{ids[0]}>"

    def convert_tokens_to_ids(self, token):
        return None


class LegacyTokenizer(ToolAwareTokenizer):
    """An older template whose signature predates tool calling."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.messages = messages
        return "rendered"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        },
    }
]


def test_tools_reach_the_chat_template():
    tok = ToolAwareTokenizer()
    worker = make_worker(StubModel(), tok)

    worker._tokenize_request(
        messages=[{"role": "user", "content": "read main.go"}],
        prompt="",
        tools=TOOLS,
    )

    assert tok.tools == TOOLS, "the schemas must be passed as `tools`, not inlined"


def test_tools_are_omitted_when_none_are_offered():
    """A request without tools must render byte-identically to before."""
    tok = ToolAwareTokenizer()
    worker = make_worker(StubModel(), tok)

    worker._tokenize_request(messages=[{"role": "user", "content": "hi"}], prompt="")

    assert tok.tools is None


def test_a_template_without_tool_support_still_generates():
    """Degrade to a tool-less prompt rather than failing the request."""
    tok = LegacyTokenizer()
    worker = make_worker(StubModel(), tok)

    token_ids = worker._tokenize_request(
        messages=[{"role": "user", "content": "read main.go"}],
        prompt="",
        tools=TOOLS,
    )

    assert token_ids, "the request must still produce tokens"
    assert tok.messages is not None, "the retry must have gone through the template"


def test_tool_loop_messages_survive_the_plain_fallback():
    """Without a chat template, a tool call must not vanish from the history."""
    worker = make_worker(StubModel(), ToolAwareTokenizer())

    rendered = worker._render_messages_plain(
        [
            {"role": "user", "content": "read main.go"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path": "main.go"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "package main"},
        ]
    )

    assert "read_file" in rendered, "the assistant turn would otherwise be blank"
    assert "main.go" in rendered
    assert "package main" in rendered


def test_a_message_with_null_content_does_not_crash_the_fallback():
    """OpenAI sends content: null on an assistant turn that only called tools."""
    worker = make_worker(StubModel(), ToolAwareTokenizer())

    rendered = worker._render_messages_plain([{"role": "assistant", "content": None}])

    assert "assistant:" in rendered
