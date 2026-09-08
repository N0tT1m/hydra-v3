package handlers

import (
	"encoding/json"
	"regexp"
	"strconv"
	"strings"

	"github.com/N0tT1m/hydra-v3/internal/api/types"
	"github.com/google/uuid"
)

// Tool calls arrive as text. The model was never told about JSON-mode or a
// grammar; it was told about tools by its own chat template, and it answers in
// whatever format that template trained it to use. Two are in circulation
// across the models this loader supports, and a Qwen3 checkpoint and a
// Qwen3-Coder checkpoint disagree, so both are handled:
//
//	Hermes / Qwen3-Instruct — a JSON object:
//	    <tool_call>
//	    {"name": "read_file", "arguments": {"path": "main.go"}}
//	    </tool_call>
//
//	Qwen3-Coder — nested XML-ish tags, all values untyped text:
//	    <tool_call>
//	    <function=read_file>
//	    <parameter=path>
//	    main.go
//	    </parameter>
//	    </function>
//	    </tool_call>
//
// The second form loses type information on the wire — every value is text —
// so the declared JSON Schema is used to coerce values back. Without that,
// `{"line": "42"}` reaches the client where it promised `{"line": 42}`, and a
// strict tool handler rejects it.
const (
	toolCallOpen  = "<tool_call>"
	toolCallClose = "</tool_call>"
)

var (
	functionRe     = regexp.MustCompile(`(?s)<function=([^>\s]+)\s*>(.*?)</function>`)
	functionOpenRe = regexp.MustCompile(`(?s)<function=([^>\s]+)\s*>(.*)$`)
	parameterRe    = regexp.MustCompile(`(?s)<parameter=([^>\s]+)\s*>(.*?)</parameter>`)
)

// parseToolCalls splits a completed generation into the prose the model wrote
// and the calls it made. Text outside the tool-call blocks is returned as
// content — models routinely narrate ("I'll read the file first") before
// calling, and dropping that loses the reasoning trail.
func parseToolCalls(out string, tools []types.Tool) (string, []types.ToolCall) {
	if !strings.Contains(out, toolCallOpen) {
		return out, nil
	}

	var content strings.Builder
	var calls []types.ToolCall

	rest := out
	for {
		start := strings.Index(rest, toolCallOpen)
		if start < 0 {
			content.WriteString(rest)
			break
		}
		content.WriteString(rest[:start])
		body := rest[start+len(toolCallOpen):]

		end := strings.Index(body, toolCallClose)
		if end < 0 {
			// Truncated mid-call, almost always max_tokens. Half a call is
			// worse than no call: the client would dispatch a tool with
			// missing arguments. Drop it and let finish_reason say "length".
			break
		}

		if call, ok := parseToolCallBody(body[:end], tools); ok {
			calls = append(calls, call)
		}
		rest = body[end+len(toolCallClose):]
	}

	return strings.TrimSpace(content.String()), calls
}

func parseToolCallBody(body string, tools []types.Tool) (types.ToolCall, bool) {
	trimmed := strings.TrimSpace(body)
	if strings.HasPrefix(trimmed, "{") {
		if call, ok := parseJSONToolCall(trimmed); ok {
			return call, true
		}
	}
	return parseXMLToolCall(trimmed, tools)
}

func parseJSONToolCall(body string) (types.ToolCall, bool) {
	var raw struct {
		Name string `json:"name"`
		// Qwen's template says "arguments"; a few checkpoints emit
		// "parameters". Accepting both costs one field.
		Arguments  json.RawMessage `json:"arguments"`
		Parameters json.RawMessage `json:"parameters"`
	}
	if err := json.Unmarshal([]byte(body), &raw); err != nil || raw.Name == "" {
		return types.ToolCall{}, false
	}

	args := raw.Arguments
	if len(args) == 0 {
		args = raw.Parameters
	}
	if len(args) == 0 {
		args = json.RawMessage("{}")
	}
	// Some models double-encode: "arguments" is a JSON string that itself
	// contains the object. Unwrap one level so the client parses an object
	// rather than a quoted blob.
	if args[0] == '"' {
		var inner string
		if json.Unmarshal(args, &inner) == nil && json.Valid([]byte(inner)) {
			args = json.RawMessage(inner)
		}
	}

	return newToolCall(raw.Name, string(args)), true
}

func parseXMLToolCall(body string, tools []types.Tool) (types.ToolCall, bool) {
	name, inner, ok := matchFunction(body)
	if !ok {
		return types.ToolCall{}, false
	}

	schema := parameterTypes(tools, name)

	// Build the JSON by hand rather than through a map: a map would sort the
	// keys, and preserving the model's emission order keeps the arguments
	// readable next to the call the model actually wrote.
	var parts []string
	for _, m := range parameterRe.FindAllStringSubmatch(inner, -1) {
		key := m[1]
		encodedKey, err := json.Marshal(key)
		if err != nil {
			continue
		}
		parts = append(parts, string(encodedKey)+":"+coerceValue(trimValue(m[2]), schema[key]))
	}

	return newToolCall(name, "{"+strings.Join(parts, ",")+"}"), true
}

func matchFunction(body string) (name, inner string, ok bool) {
	if m := functionRe.FindStringSubmatch(body); m != nil {
		return m[1], m[2], true
	}
	// An unclosed <function=...> still carries usable parameters — the
	// closing tag is the most common thing a model drops.
	if m := functionOpenRe.FindStringSubmatch(body); m != nil {
		return m[1], m[2], true
	}
	return "", "", false
}

// trimValue strips the single newline the template puts either side of a
// value, and nothing more. Trimming all whitespace would be wrong: for a
// coding agent the value is frequently source code, where leading indentation
// on the first line is load-bearing.
func trimValue(v string) string {
	v = strings.TrimPrefix(v, "\r\n")
	v = strings.TrimPrefix(v, "\n")
	v = strings.TrimSuffix(v, "\n")
	v = strings.TrimSuffix(v, "\r")
	return v
}

// coerceValue turns the XML form's untyped text back into a typed JSON value
// using the schema the caller declared. When the schema says nothing, the
// value stays a string — guessing would silently turn a file called "null",
// or a commit message of "true", into the wrong JSON type.
func coerceValue(value, declared string) string {
	quoted := func() string {
		b, err := json.Marshal(value)
		if err != nil {
			return `""`
		}
		return string(b)
	}

	switch declared {
	case "integer":
		if _, err := strconv.ParseInt(strings.TrimSpace(value), 10, 64); err == nil {
			return strings.TrimSpace(value)
		}
	case "number":
		if _, err := strconv.ParseFloat(strings.TrimSpace(value), 64); err == nil {
			return strings.TrimSpace(value)
		}
	case "boolean":
		switch strings.ToLower(strings.TrimSpace(value)) {
		case "true":
			return "true"
		case "false":
			return "false"
		}
	case "array", "object":
		if trimmed := strings.TrimSpace(value); json.Valid([]byte(trimmed)) {
			return trimmed
		}
	}
	// Includes "string", the empty (undeclared) case, and every declared type
	// whose value failed to parse — a malformed number reaches the client as
	// the text the model wrote, which is debuggable, rather than as invalid
	// JSON, which is not.
	return quoted()
}

// parameterTypes extracts {property: type} from a tool's JSON Schema.
func parameterTypes(tools []types.Tool, fn string) map[string]string {
	for _, t := range tools {
		if t.Function.Name != fn || len(t.Function.Parameters) == 0 {
			continue
		}
		var schema struct {
			Properties map[string]struct {
				Type string `json:"type"`
			} `json:"properties"`
		}
		if err := json.Unmarshal(t.Function.Parameters, &schema); err != nil {
			return nil
		}
		out := make(map[string]string, len(schema.Properties))
		for name, prop := range schema.Properties {
			out[name] = prop.Type
		}
		return out
	}
	return nil
}

func newToolCall(name, arguments string) types.ToolCall {
	return types.ToolCall{
		ID:       "call_" + strings.ReplaceAll(uuid.New().String(), "-", "")[:24],
		Type:     "function",
		Function: types.ToolCallFunction{Name: name, Arguments: arguments},
	}
}

// splitAtToolCallPrefix separates text that is safe to stream from a trailing
// fragment that might be the beginning of "<tool_call>" split across token
// boundaries. Streaming that fragment and retracting it later is not possible
// over SSE, so it is held until the next token disambiguates it.
func splitAtToolCallPrefix(s string) (emit, hold string) {
	max := len(toolCallOpen) - 1
	if max > len(s) {
		max = len(s)
	}
	for n := max; n > 0; n-- {
		if strings.HasPrefix(toolCallOpen, s[len(s)-n:]) {
			return s[:len(s)-n], s[len(s)-n:]
		}
	}
	return s, ""
}

// toolCallStreamer decides, token by token, what is safe to send as a content
// delta while a tool call may still be forming.
//
// SSE is append-only: a content delta cannot be taken back. So once
// "<tool_call>" appears, content streaming stops for the rest of the turn —
// a second call may follow the first, and the text between them is template
// scaffolding, not prose meant for the user. The calls themselves go out in a
// single delta at the end, which is what OpenAI clients accumulate anyway.
type toolCallStreamer struct {
	full   strings.Builder
	held   string
	inCall bool
}

// push records generated text and returns the portion safe to emit now.
func (t *toolCallStreamer) push(text string) string {
	t.full.WriteString(text)
	if t.inCall {
		return ""
	}

	buf := t.held + text
	if i := strings.Index(buf, toolCallOpen); i >= 0 {
		t.inCall = true
		t.held = ""
		return buf[:i]
	}

	emit, hold := splitAtToolCallPrefix(buf)
	t.held = hold
	return emit
}

// finish flushes any text held back as a possible tag prefix that turned out
// not to be one, and returns the parsed calls.
func (t *toolCallStreamer) finish(tools []types.Tool) (string, []types.ToolCall) {
	_, calls := parseToolCalls(t.full.String(), tools)
	if t.inCall {
		return "", calls
	}
	held := t.held
	t.held = ""
	return held, calls
}

// indexPtr is a helper for the streaming delta's tool-call index field.
func indexPtr(i int) *int { return &i }

// validateToolChoice reports whether the request's tool_choice can be honoured.
//
// "auto" and "none" are decidable at the prompt level. "required" and a named
// function are not: enforcing them needs constrained decoding, which the
// worker's sampler does not implement. Accepting them and hoping the model
// complies would be the fabricated-response failure this codebase avoids
// elsewhere, so they are rejected with a reason.
func validateToolChoice(raw json.RawMessage) (useTools bool, err string) {
	if len(raw) == 0 {
		return true, ""
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		switch s {
		case "auto":
			return true, ""
		case "none":
			return false, ""
		case "required":
			return false, `tool_choice "required" needs constrained decoding, which this worker does not implement; use "auto"`
		default:
			return false, "unknown tool_choice: " + s
		}
	}
	return false, `a named tool_choice needs constrained decoding, which this worker does not implement; use "auto"`
}
