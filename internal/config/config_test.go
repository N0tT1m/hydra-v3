package config

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestLoad_MissingFileUsesDefaults(t *testing.T) {
	// Point Load at a path that doesn't exist. Viper's "file not found" is
	// swallowed and defaults are used.
	path := filepath.Join(t.TempDir(), "nonexistent.toml")
	cfg, err := Load(path)
	if err != nil {
		t.Fatalf("Load should accept missing config: %v", err)
	}
	if cfg.Server.HTTPAddr != "0.0.0.0:8080" {
		t.Errorf("default HTTPAddr = %q, want 0.0.0.0:8080", cfg.Server.HTTPAddr)
	}
	if cfg.Cluster.HeartbeatInterval != 500*time.Millisecond {
		t.Errorf("default heartbeat = %v, want 500ms", cfg.Cluster.HeartbeatInterval)
	}
	if cfg.Cluster.MaxVRAMGB != 512 {
		t.Errorf("default max_vram_gb = %v, want 512", cfg.Cluster.MaxVRAMGB)
	}
	if cfg.Auth.Enabled {
		t.Error("auth should default to disabled")
	}
}

func TestLoad_ParsesToml(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.toml")
	content := `
[server]
http_addr = "127.0.0.1:9999"

[cluster]
register_token = "file-secret"
max_vram_gb = 128.0
heartbeat_interval = "250ms"

[auth]
enabled = true
api_keys = ["key1", "key2"]
`
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}

	cfg, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	if cfg.Server.HTTPAddr != "127.0.0.1:9999" {
		t.Errorf("HTTPAddr = %q, want 127.0.0.1:9999", cfg.Server.HTTPAddr)
	}
	if cfg.Cluster.RegisterToken != "file-secret" {
		t.Errorf("RegisterToken = %q, want file-secret", cfg.Cluster.RegisterToken)
	}
	if cfg.Cluster.MaxVRAMGB != 128 {
		t.Errorf("MaxVRAMGB = %v, want 128", cfg.Cluster.MaxVRAMGB)
	}
	if cfg.Cluster.HeartbeatInterval != 250*time.Millisecond {
		t.Errorf("HeartbeatInterval = %v, want 250ms", cfg.Cluster.HeartbeatInterval)
	}
	if !cfg.Auth.Enabled {
		t.Error("Auth.Enabled should be true")
	}
	if len(cfg.Auth.APIKeys) != 2 {
		t.Errorf("APIKeys len = %d, want 2", len(cfg.Auth.APIKeys))
	}
}

func TestLoad_EnvOverridesFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.toml")
	content := `
[cluster]
register_token = "from-file"
`
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("HYDRA_CLUSTER_REGISTER_TOKEN", "from-env")

	cfg, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Cluster.RegisterToken != "from-env" {
		t.Errorf("env should override file: got %q, want from-env", cfg.Cluster.RegisterToken)
	}
}

func TestLoad_LogDefaults(t *testing.T) {
	// den-den-mushi wiring: default to JSON-on-stdout so the fleet forwarder
	// captures logs with no per-app change.
	cfg, err := Load(filepath.Join(t.TempDir(), "nonexistent.toml"))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Log.Format != "json" {
		t.Errorf("default log.format = %q, want json", cfg.Log.Format)
	}
	if cfg.Log.App != "hydra" {
		t.Errorf("default log.app = %q, want hydra", cfg.Log.App)
	}
}

func TestLoad_LogEnvOverride(t *testing.T) {
	t.Setenv("HYDRA_LOG_FORMAT", "console")
	t.Setenv("HYDRA_LOG_APP", "hydra-dev")
	cfg, err := Load(filepath.Join(t.TempDir(), "nonexistent.toml"))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Log.Format != "console" {
		t.Errorf("log.format = %q, want console (env override)", cfg.Log.Format)
	}
	if cfg.Log.App != "hydra-dev" {
		t.Errorf("log.app = %q, want hydra-dev (env override)", cfg.Log.App)
	}
}

func TestLoad_MalformedTomlError(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.toml")
	if err := os.WriteFile(path, []byte("not a valid ][[ toml"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(path); err == nil {
		t.Error("Load should error on malformed TOML")
	}
}

func TestLoad_ReportsAValueOfTheWrongType(t *testing.T) {
	// The TOML parses fine, so the failure surfaces at decode time: a
	// heartbeat interval that is not a duration cannot be mapped onto
	// time.Duration, and Load must say so rather than hand back a config with
	// a silently zeroed field.
	path := filepath.Join(t.TempDir(), "config.toml")
	content := `
[cluster]
heartbeat_interval = "not-a-duration"
`
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	if _, err := Load(path); err == nil {
		t.Fatal("expected an error for a value that cannot be decoded")
	}
}
