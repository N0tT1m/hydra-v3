package coordinator

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

// bytesPerParam maps a checkpoint's declared dtype to its width in bytes.
// Unknown dtypes fall back to 2, matching the fp16/bf16 default that every
// modern checkpoint ships.
func bytesPerParam(dtype string) float64 {
	switch strings.ToLower(strings.TrimSpace(dtype)) {
	case "float32", "f32":
		return 4
	case "float16", "f16", "bfloat16", "bf16":
		return 2
	case "float8_e4m3fn", "float8_e5m2", "f8", "int8":
		return 1
	case "int4", "uint4":
		return 0.5
	default:
		return 2
	}
}

// ModelShape is the subset of a model's config needed to predict what it
// costs to hold. It exists because layer distribution used to assume a fixed
// number of GB per layer for every model: a 27B layer is ~0.76GB against a
// 0.5GB default, so the allocator believed 38 layers needed 19GB when they
// needed ~29GB, filled the card, and left nothing for activations or KV
// cache.
type ModelShape struct {
	Layers            int
	HiddenSize        int
	IntermediateSize  int
	NumAttentionHeads int
	NumKeyValueHeads  int
	HeadDim           int
	VocabSize         int
	BytesPerParam     float64
	TieWordEmbeddings bool

	// LayerTypes is set for hybrid decoders (Qwen3.5 mixes linear_attention
	// and full_attention). Their layers are not uniform, so a per-layer
	// estimate derived from the dense formula is only approximate.
	LayerTypes []string
}

// kvHeadDim is the width of one key or value projection.
func (s ModelShape) kvHeadDim() int {
	head := s.HeadDim
	if head == 0 && s.NumAttentionHeads > 0 {
		head = s.HiddenSize / s.NumAttentionHeads
	}
	kvHeads := s.NumKeyValueHeads
	if kvHeads == 0 {
		kvHeads = s.NumAttentionHeads
	}
	return head * kvHeads
}

// ParamsPerLayer estimates the parameter count of one decoder layer, using
// the standard attention + gated-MLP shape shared by Llama, Mistral and Qwen.
func (s ModelShape) ParamsPerLayer() float64 {
	if s.HiddenSize == 0 {
		return 0
	}
	h := float64(s.HiddenSize)
	kv := float64(s.kvHeadDim())
	inter := float64(s.IntermediateSize)

	attn := h*h + 2*h*kv + h*h // q, k, v, o
	mlp := 3 * h * inter       // gate, up, down
	norms := 2 * h
	return attn + mlp + norms
}

// SafetyFactor scales the analytic estimate to cover what the dense formula
// does not model. Measured against real checkpoints: Qwen2.5-7B comes out at
// 14.2GiB against 14.2GiB actual, but Qwen3.5-27B at 44.1GiB against 50.1GiB
// actual, because its linear_attention layers carry convolution and gating
// projections the dense shape has no term for. Under-estimating is what
// filled a card to 94% and left 8MiB for activations, so hybrids are biased
// upward: over-estimating costs a layer of placement, under-estimating costs
// the whole run.
func (s ModelShape) SafetyFactor() float64 {
	if s.IsHybrid() {
		return 1.30
	}
	return 1.10
}

// BytesPerLayer is what one layer costs to hold, in bytes, including the
// safety factor.
func (s ModelShape) BytesPerLayer() float64 {
	return s.ParamsPerLayer() * s.BytesPerParam * s.SafetyFactor()
}

// GBPerLayer is BytesPerLayer in GiB, the unit distribution works in.
func (s ModelShape) GBPerLayer() float64 {
	return s.BytesPerLayer() / (1024 * 1024 * 1024)
}

// EmbeddingGB is the embedding table's cost. Only the node holding the
// embedding pays it, and for a large vocabulary it is far from negligible:
// Qwen3.5-27B's is ~2.4GiB, five times a 0.5GB "per layer" allowance.
func (s ModelShape) EmbeddingGB() float64 {
	return float64(s.VocabSize) * float64(s.HiddenSize) * s.BytesPerParam / (1024 * 1024 * 1024)
}

// LMHeadGB is the output projection's cost, paid by the last node. Models
// with tied embeddings reuse the embedding weight and pay nothing extra.
func (s ModelShape) LMHeadGB() float64 {
	if s.TieWordEmbeddings {
		return 0
	}
	return s.EmbeddingGB()
}

// KVCacheGBPerToken is the per-token cost of the KV cache across *all*
// layers, both keys and values.
func (s ModelShape) KVCacheGBPerToken() float64 {
	kv := float64(s.kvHeadDim())
	return 2 * kv * float64(s.Layers) * s.BytesPerParam / (1024 * 1024 * 1024)
}

// Usable reports whether the shape carries enough information to size a
// layer. Without it, callers must fall back to the configured constant.
func (s ModelShape) Usable() bool {
	return s.Layers > 0 && s.HiddenSize > 0 && s.IntermediateSize > 0
}

// IsHybrid reports whether the decoder mixes token-mixer types, in which case
// the dense per-layer formula is an approximation.
func (s ModelShape) IsHybrid() bool {
	if len(s.LayerTypes) == 0 {
		return false
	}
	first := s.LayerTypes[0]
	for _, t := range s.LayerTypes[1:] {
		if t != first {
			return true
		}
	}
	return false
}

// rawModelConfig mirrors the fields we read out of a HuggingFace config.json.
// Multimodal checkpoints (Qwen3.5, VLMs generally) nest the decoder's own
// hyperparameters under text_config and leave only wrapper fields at the top
// level, so every field has to be resolved through that indirection.
type rawModelConfig struct {
	NumHiddenLayers   int      `json:"num_hidden_layers"`
	NumLayers         int      `json:"num_layers"`
	NLayer            int      `json:"n_layer"`
	HiddenSize        int      `json:"hidden_size"`
	IntermediateSize  int      `json:"intermediate_size"`
	NumAttentionHeads int      `json:"num_attention_heads"`
	NumKeyValueHeads  int      `json:"num_key_value_heads"`
	HeadDim           int      `json:"head_dim"`
	VocabSize         int      `json:"vocab_size"`
	TorchDtype        string   `json:"torch_dtype"`
	Dtype             string   `json:"dtype"`
	TieWordEmbeddings *bool    `json:"tie_word_embeddings"`
	LayerTypes        []string `json:"layer_types"`

	TextConfig *rawModelConfig `json:"text_config"`
}

// layerCount returns the layer count under whichever name this architecture
// uses.
func (r *rawModelConfig) layerCount() int {
	for _, n := range []int{r.NumHiddenLayers, r.NumLayers, r.NLayer} {
		if n > 0 {
			return n
		}
	}
	return 0
}

// parseModelShape reads a config.json body into a ModelShape, descending into
// text_config when the decoder's parameters live there.
func parseModelShape(body io.Reader) (ModelShape, error) {
	var raw rawModelConfig
	if err := json.NewDecoder(body).Decode(&raw); err != nil {
		return ModelShape{}, fmt.Errorf("failed to parse config: %w", err)
	}

	// The decoder's own parameters win; the wrapper's are a fallback so a
	// flat config keeps working unchanged.
	inner := &raw
	if raw.TextConfig != nil && raw.TextConfig.layerCount() > 0 {
		inner = raw.TextConfig
	}

	pick := func(a, b int) int {
		if a > 0 {
			return a
		}
		return b
	}

	dtype := inner.TorchDtype
	if dtype == "" {
		dtype = inner.Dtype
	}
	if dtype == "" {
		dtype = raw.TorchDtype
	}
	if dtype == "" {
		dtype = raw.Dtype
	}

	// tie_word_embeddings is commonly declared on the wrapper, not the text
	// config, so prefer whichever actually set it.
	tied := false
	if inner.TieWordEmbeddings != nil {
		tied = *inner.TieWordEmbeddings
	} else if raw.TieWordEmbeddings != nil {
		tied = *raw.TieWordEmbeddings
	}

	layerTypes := inner.LayerTypes
	if len(layerTypes) == 0 {
		layerTypes = raw.LayerTypes
	}

	shape := ModelShape{
		Layers:            pick(inner.layerCount(), raw.layerCount()),
		HiddenSize:        pick(inner.HiddenSize, raw.HiddenSize),
		IntermediateSize:  pick(inner.IntermediateSize, raw.IntermediateSize),
		NumAttentionHeads: pick(inner.NumAttentionHeads, raw.NumAttentionHeads),
		NumKeyValueHeads:  pick(inner.NumKeyValueHeads, raw.NumKeyValueHeads),
		HeadDim:           pick(inner.HeadDim, raw.HeadDim),
		VocabSize:         pick(inner.VocabSize, raw.VocabSize),
		BytesPerParam:     bytesPerParam(dtype),
		TieWordEmbeddings: tied,
		LayerTypes:        layerTypes,
	}

	if shape.Layers == 0 {
		return shape, fmt.Errorf("no layer count found in config")
	}
	return shape, nil
}
