package handlers

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/N0tT1m/hydra-v3/internal/api/types"
)

func editTool() []types.Tool {
	return []types.Tool{{
		Type: "function",
		Function: types.ToolFunction{
			Name: "edit_file",
			Parameters: json.RawMessage(`{
				"type": "object",
				"properties": {
					"path":      {"type": "string"},
					"line":      {"type": "integer"},
					"overwrite": {"type": "boolean"},
					"content":   {"type": "string"},
					"tags":      {"type": "array"}
				}
			}`),
		},
	}}
}

func onlyCall(t *testing.T, calls []types.ToolCall) types.ToolCall {
	t.Helper()
	if len(calls) != 1 {
		t.Fatalf("want 1 tool call, got %d", len(calls))
	}
	return calls[0]
}

func TestParseToolCalls_HermesJSONForm(t *testing.T) {
	out := "Let me look.\n<tool_call>\n{\"name\": \"edit_file\", \"arguments\": {\"path\": \"main.go\", \"line\": 42}}\n</tool_call>"

	content, calls := parseToolCalls(out, editTool())

	if content != "Let me look." {
		t.Errorf("prose outside the call must survive: %q", content)
	}
	call := onlyCall(t, calls)
	if call.Function.Name != "edit_file" {
		t.Errorf("name = %q", call.Function.Name)
	}
	var args map[string]any
	if err := json.Unmarshal([]byte(call.Function.Arguments), &args); err != nil {
		t.Fatalf("arguments must be valid JSON: %v", err)
	}
	if args["path"] != "main.go" {
		t.Errorf("path = %v", args["path"])
	}
}

// The Qwen3-Coder form carries no types, so the declared schema is the only
// thing that can put them back.
func TestParseToolCalls_CoderXMLFormCoercesUsingTheSchema(t *testing.T) {
	out := `<tool_call>
<function=edit_file>
<parameter=path>
main.go
</parameter>
<parameter=line>
42
</parameter>
<parameter=overwrite>
true
</parameter>
<parameter=tags>
["a","b"]
</parameter>
</function>
</tool_call>`

	_, calls := parseToolCalls(out, editTool())
	call := onlyCall(t, calls)

	var args map[string]any
	if err := json.Unmarshal([]byte(call.Function.Arguments), &args); err != nil {
		t.Fatalf("arguments must be valid JSON: %v\n%s", err, call.Function.Arguments)
	}
	if args["path"] != "main.go" {
		t.Errorf("path = %#v, want the string main.go", args["path"])
	}
	if n, ok := args["line"].(float64); !ok || n != 42 {
		t.Errorf("line = %#v, want the number 42 (schema says integer)", args["line"])
	}
	if args["overwrite"] != true {
		t.Errorf("overwrite = %#v, want the boolean true", args["overwrite"])
	}
	if tags, ok := args["tags"].([]any); !ok || len(tags) != 2 {
		t.Errorf("tags = %#v, want a 2-element array", args["tags"])
	}
}

// The reason this matters for a coding agent specifically: a `content`
// argument is usually source, and its indentation is not decoration.
func TestParseToolCalls_PreservesIndentationInsideAParameter(t *testing.T) {
	body := "def f():\n    return 1"
	out := "<tool_call>\n<function=edit_file>\n<parameter=content>\n" + body + "\n</parameter>\n</function>\n</tool_call>"

	_, calls := parseToolCalls(out, editTool())
	call := onlyCall(t, calls)

	var args map[string]string
	if err := json.Unmarshal([]byte(call.Function.Arguments), &args); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if args["content"] != body {
		t.Errorf("indentation must survive verbatim:\ngot  %q\nwant %q", args["content"], body)
	}
}

// An undeclared parameter must not be guessed at: a file literally named
// "null" is a string, not JSON null.
func TestParseToolCalls_UndeclaredParametersStayStrings(t *testing.T) {
	out := "<tool_call>\n<function=edit_file>\n<parameter=mystery>\nnull\n</parameter>\n</function>\n</tool_call>"

	_, calls := parseToolCalls(out, editTool())
	call := onlyCall(t, calls)

	if !strings.Contains(call.Function.Arguments, `"mystery":"null"`) {
		t.Errorf("want the text preserved as a string, got %s", call.Function.Arguments)
	}
}

func TestParseToolCalls_ParallelCalls(t *testing.T) {
	out := `<tool_call>
{"name": "edit_file", "arguments": {"path": "a.go"}}
</tool_call>
<tool_call>
{"name": "edit_file", "arguments": {"path": "b.go"}}
</tool_call>`

	_, calls := parseToolCalls(out, editTool())
	if len(calls) != 2 {
		t.Fatalf("want 2 calls, got %d", len(calls))
	}
	if calls[0].ID == calls[1].ID {
		t.Error("each call needs its own id; clients key tool results off it")
	}
}

// Truncation mid-call is the max_tokens case. Dispatching a call with missing
// arguments is worse than reporting none.
func TestParseToolCalls_TruncatedCallIsDropped(t *testing.T) {
	out := `Working.<tool_call>
{"name": "edit_file", "arguments": {"pa`

	content, calls := parseToolCalls(out, editTool())
	if len(calls) != 0 {
		t.Errorf("a half-written call must not be emitted, got %d", len(calls))
	}
	if content != "Working." {
		t.Errorf("content = %q", content)
	}
}

func TestParseToolCalls_NoCallsLeavesTextUntouched(t *testing.T) {
	out := "just prose, no calls here"
	content, calls := parseToolCalls(out, editTool())
	if content != out || calls != nil {
		t.Errorf("got (%q, %v)", content, calls)
	}
}

// SSE cannot retract a delta, so a fragment that might open a tool call is
// held until the next token proves it either way.
func TestToolCallStreamer_HoldsBackAPartialOpeningTag(t *testing.T) {
	var s toolCallStreamer

	if got := s.push("Reading the file <tool"); got != "Reading the file " {
		t.Errorf("the ambiguous tail must be withheld, got %q", got)
	}
	if got := s.push("_call>\n{\"name\": \"edit_file\", \"arguments\": {}}\n</tool_call>"); got != "" {
		t.Errorf("nothing may be emitted once a call has opened, got %q", got)
	}

	flushed, calls := s.finish(editTool())
	if flushed != "" {
		t.Errorf("flushed = %q", flushed)
	}
	if len(calls) != 1 {
		t.Fatalf("want the call to be parsed, got %d", len(calls))
	}
}

func TestToolCallStreamer_ReleasesAHeldFragmentThatWasNotATag(t *testing.T) {
	var s toolCallStreamer

	if got := s.push("compare a <to"); got != "compare a " {
		t.Errorf("the ambiguous tail must be withheld, got %q", got)
	}
	// "<to" was held; "b" disproves the tag, so both are released together.
	if got := s.push("b"); got != "<tob" {
		t.Errorf("a fragment that turned out to be prose must be released, got %q", got)
	}

	flushed, calls := s.finish(editTool())
	if len(calls) != 0 {
		t.Errorf("no calls expected, got %d", len(calls))
	}
	if flushed != "" {
		t.Errorf("flushed = %q", flushed)
	}
}

func TestToolCallStreamer_FlushesATrailingFragmentAtEndOfStream(t *testing.T) {
	var s toolCallStreamer

	if got := s.push("done <too"); got != "done " {
		t.Errorf("push = %q", got)
	}
	flushed, _ := s.finish(editTool())
	if flushed != "<too" {
		t.Errorf("the held fragment must be delivered at the end, got %q", flushed)
	}
}

func TestValidateToolChoice(t *testing.T) {
	cases := []struct {
		name      string
		raw       string
		useTools  bool
		wantError bool
	}{
		{"absent", "", true, false},
		{"auto", `"auto"`, true, false},
		{"none", `"none"`, false, false},
		{"required is not enforceable", `"required"`, false, true},
		{"named is not enforceable", `{"type":"function","function":{"name":"edit_file"}}`, false, true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			use, errMsg := validateToolChoice(json.RawMessage(tc.raw))
			if use != tc.useTools {
				t.Errorf("useTools = %v, want %v", use, tc.useTools)
			}
			if (errMsg != "") != tc.wantError {
				t.Errorf("error = %q, wantError = %v", errMsg, tc.wantError)
			}
		})
	}
}
