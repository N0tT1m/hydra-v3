package coordinator

import (
	"strings"
	"testing"

	"github.com/N0tT1m/hydra-v3/internal/config"
)

// validator returns a Coordinator with only the config slot populated —
// enough for validateRegister(), which doesn't touch the broker or registry.
func validator(token string, maxVRAM float64) *Coordinator {
	return &Coordinator{
		config: &config.Config{
			Cluster: config.ClusterConfig{
				RegisterToken: token,
				MaxVRAMGB:     maxVRAM,
			},
		},
	}
}

func base() *RegisterRequest {
	return &RegisterRequest{
		NodeID:       "worker-1",
		Host:         "127.0.0.1",
		PipelinePort: 6000,
		VRAMGB:       16,
		Capabilities: []string{"cuda"},
	}
}

func TestValidateRegister_AcceptsValid(t *testing.T) {
	c := validator("", 512)
	if err := c.validateRegister(base()); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestValidateRegister_RejectsEmptyNodeID(t *testing.T) {
	c := validator("", 512)
	req := base()
	req.NodeID = ""
	err := c.validateRegister(req)
	if err == nil || !strings.Contains(err.Error(), "node_id") {
		t.Errorf("expected node_id error, got %v", err)
	}
}

func TestValidateRegister_RejectsMissingToken(t *testing.T) {
	c := validator("s3cret", 512)
	if err := c.validateRegister(base()); err == nil {
		t.Error("missing token should be rejected when coordinator has one configured")
	}
}

func TestValidateRegister_RejectsWrongToken(t *testing.T) {
	c := validator("s3cret", 512)
	req := base()
	req.Token = "wrong"
	if err := c.validateRegister(req); err == nil {
		t.Error("wrong token should be rejected")
	}
}

func TestValidateRegister_AcceptsCorrectToken(t *testing.T) {
	c := validator("s3cret", 512)
	req := base()
	req.Token = "s3cret"
	if err := c.validateRegister(req); err != nil {
		t.Errorf("correct token should be accepted: %v", err)
	}
}

func TestValidateRegister_RejectsNegativeVRAM(t *testing.T) {
	c := validator("", 512)
	req := base()
	req.VRAMGB = 0
	if err := c.validateRegister(req); err == nil {
		t.Error("zero VRAM should be rejected")
	}
	req.VRAMGB = -5
	if err := c.validateRegister(req); err == nil {
		t.Error("negative VRAM should be rejected")
	}
}

func TestValidateRegister_RejectsImplausibleVRAM(t *testing.T) {
	// A hostile worker claiming 1TB grabs all layers in proportional
	// distribution. The cap is load-bearing.
	c := validator("", 512)
	req := base()
	req.VRAMGB = 1024
	err := c.validateRegister(req)
	if err == nil || !strings.Contains(err.Error(), "max_vram_gb") {
		t.Errorf("implausible VRAM should be rejected with cap error, got %v", err)
	}
}

func TestValidateRegister_CapZeroDisablesCheck(t *testing.T) {
	c := validator("", 0) // cap=0 means no limit
	req := base()
	req.VRAMGB = 999999
	if err := c.validateRegister(req); err != nil {
		t.Errorf("cap=0 should disable VRAM upper bound, got %v", err)
	}
}

func TestValidateRegister_RejectsBadPort(t *testing.T) {
	c := validator("", 512)
	for _, port := range []int{0, -1, 65536, 100000} {
		req := base()
		req.PipelinePort = port
		if err := c.validateRegister(req); err == nil {
			t.Errorf("port %d should be rejected", port)
		}
	}
}
