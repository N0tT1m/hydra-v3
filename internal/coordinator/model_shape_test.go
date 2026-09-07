package coordinator

import (
	"math"
	"strings"
	"testing"
)

// Real config.json excerpts. Values are copied from the published checkpoints
// so the estimates below are checked against models that actually exist.

const qwen25_7B = `{
  "model_type": "qwen2",
  "num_hidden_layers": 28,
  "hidden_size": 3584,
  "intermediate_size": 18944,
  "num_attention_heads": 28,
  "num_key_value_heads": 4,
  "vocab_size": 152064,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16"
}`

// Qwen3.5-27B: multimodal wrapper, decoder nested under text_config, hybrid
// layer types. The layer count is NOT at the top level.
const qwen35_27B = `{
  "model_type": "qwen3_5",
  "architectures": ["Qwen3_5ForConditionalGeneration"],
  "tie_word_embeddings": false,
  "vision_config": {"model_type": "qwen3_5", "hidden_size": 1152},
  "text_config": {
    "model_type": "qwen3_5_text",
    "num_hidden_layers": 64,
    "hidden_size": 5120,
    "intermediate_size": 17408,
    "num_attention_heads": 40,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 248320,
    "dtype": "bfloat16",
    "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
  }
}`

func TestShapeReadsAFlatConfig(t *testing.T) {
	shape, err := parseModelShape(strings.NewReader(qwen25_7B))
	if err != nil {
		t.Fatal(err)
	}
	if shape.Layers != 28 || shape.HiddenSize != 3584 {
		t.Fatalf("got layers=%d hidden=%d", shape.Layers, shape.HiddenSize)
	}
	if shape.IsHybrid() {
		t.Error("a uniform decoder must not be treated as hybrid")
	}
}

func TestShapeDescendsIntoTextConfig(t *testing.T) {
	shape, err := parseModelShape(strings.NewReader(qwen35_27B))
	if err != nil {
		t.Fatal(err)
	}
	// Every one of these lives under text_config, not at the top level.
	if shape.Layers != 64 {
		t.Errorf("layers = %d, want 64 (nested under text_config)", shape.Layers)
	}
	if shape.HiddenSize != 5120 {
		t.Errorf("hidden = %d, want 5120", shape.HiddenSize)
	}
	if shape.VocabSize != 248320 {
		t.Errorf("vocab = %d, want 248320", shape.VocabSize)
	}
	if !shape.IsHybrid() {
		t.Error("mixed layer_types must be detected as hybrid")
	}
	// tie_word_embeddings is declared on the wrapper, not the text config.
	if shape.TieWordEmbeddings {
		t.Error("tie_word_embeddings=false on the wrapper must be honoured")
	}
}

// The whole point: per-layer cost must track the model, not a constant.
func TestPerLayerCostVariesByModel(t *testing.T) {
	small, _ := parseModelShape(strings.NewReader(qwen25_7B))
	large, _ := parseModelShape(strings.NewReader(qwen35_27B))

	if small.GBPerLayer() >= large.GBPerLayer() {
		t.Fatalf("a 27B layer (%.3f GiB) must cost more than a 7B layer (%.3f GiB)",
			large.GBPerLayer(), small.GBPerLayer())
	}
	// The old hardcoded default. A 27B layer exceeding it is precisely the
	// condition that overfilled a card and left 8MiB for the KV cache.
	const oldDefault = 0.5
	if large.GBPerLayer() <= oldDefault {
		t.Errorf("27B per-layer %.3f GiB should exceed the old %.1f GiB constant",
			large.GBPerLayer(), oldDefault)
	}
}

func TestTiedEmbeddingsCostNothingExtraForTheLMHead(t *testing.T) {
	tied, err := parseModelShape(strings.NewReader(`{
	  "num_hidden_layers": 4, "hidden_size": 64, "intermediate_size": 128,
	  "num_attention_heads": 4, "num_key_value_heads": 4, "vocab_size": 1000,
	  "tie_word_embeddings": true, "torch_dtype": "float16"}`))
	if err != nil {
		t.Fatal(err)
	}
	if tied.LMHeadGB() != 0 {
		t.Errorf("tied lm_head should cost 0, got %f", tied.LMHeadGB())
	}
	if tied.EmbeddingGB() <= 0 {
		t.Error("the embedding itself still costs memory")
	}
}

func TestDtypeWidthChangesTheEstimate(t *testing.T) {
	base := `{"num_hidden_layers":4,"hidden_size":64,"intermediate_size":128,
	  "num_attention_heads":4,"num_key_value_heads":4,"vocab_size":1000,"torch_dtype":%q}`
	f16, _ := parseModelShape(strings.NewReader(strings_Replace(base, "float16")))
	f32, _ := parseModelShape(strings.NewReader(strings_Replace(base, "float32")))

	if math.Abs(f32.GBPerLayer()-2*f16.GBPerLayer()) > 1e-9 {
		t.Errorf("fp32 should cost exactly twice fp16: %f vs %f",
			f32.GBPerLayer(), f16.GBPerLayer())
	}
}

func strings_Replace(format, dtype string) string {
	return strings.Replace(format, "%q", `"`+dtype+`"`, 1)
}

func TestUnusableShapeFallsBackToTheConstant(t *testing.T) {
	cfg := DistributionConfig{MemoryPerLayerGB: 0.5}
	if got := cfg.EffectiveMemoryPerLayerGB(); got != 0.5 {
		t.Errorf("with no shape the configured constant must be used, got %f", got)
	}

	shape, _ := parseModelShape(strings.NewReader(qwen35_27B))
	cfg.Shape = shape
	if got := cfg.EffectiveMemoryPerLayerGB(); got == 0.5 {
		t.Error("with a usable shape the constant must not be used")
	}
}

func TestHybridModelsAreSizedMoreConservatively(t *testing.T) {
	dense, _ := parseModelShape(strings.NewReader(qwen25_7B))
	hybrid, _ := parseModelShape(strings.NewReader(qwen35_27B))

	if hybrid.SafetyFactor() <= dense.SafetyFactor() {
		t.Errorf("hybrid decoders must carry more margin: %f vs %f",
			hybrid.SafetyFactor(), dense.SafetyFactor())
	}
}

// The regression this whole change exists for: Qwen3.5-27B across a 31.4GiB
// CUDA node and a 22.3GiB MPS node. With a flat 0.5GB/layer the allocator put
// 38 layers on the 5090, which loaded to 94% and then died with
//
//	CUDA out of memory. Tried to allocate 12.00 MiB.
//	GPU 0 has 31.39 GiB capacity, of which 8.19 MiB is free.
//
// on the first prompt of any length.
func TestRealClusterNoLongerOverfillsTheLargestNode(t *testing.T) {
	shape, err := parseModelShape(strings.NewReader(qwen35_27B))
	if err != nil {
		t.Fatal(err)
	}
	vram := map[string]float64{"goose-5090": 31.39, "macbook-mps": 22.3}

	old := DistributionConfig{
		TotalLayers: 64, MinLayersPerNode: 1,
		MemoryPerLayerGB: 0.5, ReservedVRAMGB: 2.0,
	}
	oldDist, err := DistributeLayersProportional(vram, old)
	if err != nil {
		t.Fatalf("baseline distribution failed: %v", err)
	}

	now := DistributionConfig{
		TotalLayers: 64, MinLayersPerNode: 1,
		MemoryPerLayerGB: 0.5, ReservedVRAMGB: 2.0,
		Shape:            shape,
		EmbeddingGB:      shape.EmbeddingGB(),
		LMHeadGB:         shape.LMHeadGB(),
		KVCacheReserveGB: shape.KVCacheGBPerToken() * 4096,
	}
	newDist, err := DistributeLayersProportional(vram, now)
	if err != nil {
		// Refusing outright is also an acceptable outcome: it beats loading
		// and then OOMing. Just say so loudly.
		t.Logf("model refused with real sizing: %v", err)
		return
	}

	byNode := func(d []LayerAssignment, id string) int {
		for _, a := range d {
			if a.NodeID == id {
				return a.LayerEnd - a.LayerStart
			}
		}
		return 0
	}

	oldGPU, newGPU := byNode(oldDist, "goose-5090"), byNode(newDist, "goose-5090")
	t.Logf("5090 layers: old(flat 0.5GB)=%d  new=%d  (model-derived %.3f GiB/layer)",
		oldGPU, newGPU, shape.GBPerLayer())

	// What the 5090 would actually have to hold under the new plan:
	// its layers, plus the embedding, since it sorts first.
	held := float64(newGPU)*shape.GBPerLayer() + shape.EmbeddingGB()
	budget := 31.39 - now.ReservedVRAMGB - now.KVCacheReserveGB
	t.Logf("5090 would hold %.1f GiB against a %.1f GiB budget", held, budget)

	if held > budget {
		t.Errorf("still over-subscribed: %.1f GiB assigned into %.1f GiB", held, budget)
	}
	if newGPU >= oldGPU {
		t.Errorf("expected fewer layers on the 5090 than the flat-constant plan (%d), got %d",
			oldGPU, newGPU)
	}
}
