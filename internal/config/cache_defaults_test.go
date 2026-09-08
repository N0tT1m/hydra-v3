package config

import (
	"testing"
	"time"
)

// Prefix caching is on by default, so an existing deployment with no [cache]
// section gets it without editing config.
func TestLoad_PrefixCacheDefaults(t *testing.T) {
	cfg, err := Load("")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Cache.Sessions != 4 {
		t.Errorf("cache.prefix_sessions = %d, want 4", cfg.Cache.Sessions)
	}
	if cfg.Cache.TTL != 30*time.Minute {
		t.Errorf("cache.prefix_ttl = %v, want 30m", cfg.Cache.TTL)
	}
}
