package config

import (
	"os"
	"strings"
	"time"

	"github.com/spf13/viper"
)

// Config holds the complete configuration for the coordinator
type Config struct {
	Server      ServerConfig      `mapstructure:"server"`
	Cluster     ClusterConfig     `mapstructure:"cluster"`
	Auth        AuthConfig        `mapstructure:"auth"`
	ZMQ         ZMQConfig         `mapstructure:"zmq"`
	Model       ModelConfig       `mapstructure:"model"`
	Cache       CacheConfig       `mapstructure:"cache"`
	LocalWorker LocalWorkerConfig `mapstructure:"local_worker"`
	Log         LogConfig         `mapstructure:"log"`
}

// CacheConfig controls prefix caching: keeping a finished request's KV cache
// alive so the next turn of the same conversation prefills only its new
// tokens instead of the whole history.
//
// The cost is VRAM. Every retained session holds a full KV cache for as long
// as it lives, so Sessions is effectively a multiplier on the KV budget that
// cluster.kv_reserve_tokens holds back.
type CacheConfig struct {
	// Sessions is how many conversations may keep a cache at once. 0 disables
	// prefix caching entirely, clearing every cache on completion.
	Sessions int `mapstructure:"prefix_sessions"`
	// TTL is how long an idle conversation keeps its cache before it is
	// evicted and its VRAM returned.
	TTL time.Duration `mapstructure:"prefix_ttl"`
}

// LogConfig controls log emission. The default ("json") produces
// den-den-mushi-shaped lines on stdout ({ts, level, app, host, msg, ...}) that
// the fleet's log forwarder (Vector docker_logs source) ships to the hub with
// no per-app change. Set format="console" for human-readable local dev output.
type LogConfig struct {
	// Format is "json" (den-den-mushi, default) or "console" (dev).
	// HYDRA_LOG_FORMAT overrides this.
	Format string `mapstructure:"format"`
	// App is the den-den-mushi `app` tag stamped on every line.
	// HYDRA_LOG_APP overrides this.
	App string `mapstructure:"app"`
}

// LocalWorkerConfig holds local worker configuration
type LocalWorkerConfig struct {
	Enabled bool   `mapstructure:"enabled"`
	NodeID  string `mapstructure:"node_id"`
	Device  string `mapstructure:"device"`
	Dtype   string `mapstructure:"dtype"`
}

// ServerConfig holds HTTP server configuration
type ServerConfig struct {
	HTTPAddr    string `mapstructure:"http_addr"`
	MetricsAddr string `mapstructure:"metrics_addr"`
}

// ClusterConfig holds cluster management configuration
type ClusterConfig struct {
	NodeID             string        `mapstructure:"node_id"`
	HeartbeatInterval  time.Duration `mapstructure:"heartbeat_interval"`
	UnhealthyThreshold int           `mapstructure:"unhealthy_threshold"`
	ReservedVRAMGB     float64       `mapstructure:"reserved_vram_gb"`
	// MemoryPerLayerGB is only a fallback. When the model's config can be
	// read, per-layer cost is derived from the model itself; this constant
	// cannot be right for every architecture at once.
	MemoryPerLayerGB float64 `mapstructure:"memory_per_layer_gb"`
	// KVReserveTokens is the context length that memory is held back for on
	// every node. Set it to 0 to reserve nothing -- which is how a model can
	// load into all available memory and then OOM on the first real prompt.
	KVReserveTokens int `mapstructure:"kv_reserve_tokens"`

	// MaxVRAMGB is the upper bound on per-worker VRAM claims during
	// registration. Rejects implausible values (including hostile workers
	// claiming enormous VRAM to grab the whole layer distribution).
	// A value of 0 disables the check.
	MaxVRAMGB float64 `mapstructure:"max_vram_gb"`

	// RegisterToken is a shared secret that workers must present in the
	// `token` field of their register message. Empty disables the check.
	// The HYDRA_CLUSTER_REGISTER_TOKEN env var overrides this.
	RegisterToken string `mapstructure:"register_token"`
}

// AuthConfig holds authentication configuration
type AuthConfig struct {
	Enabled    bool          `mapstructure:"enabled"`
	APIKeys    []string      `mapstructure:"api_keys"`
	RateLimit  int           `mapstructure:"rate_limit"`
	RateWindow time.Duration `mapstructure:"rate_window"`
}

// ZMQConfig holds ZeroMQ configuration
type ZMQConfig struct {
	RouterAddr    string `mapstructure:"router_addr"`
	MetricsAddr   string `mapstructure:"metrics_addr"`
	BroadcastAddr string `mapstructure:"broadcast_addr"`
	HighWaterMark int    `mapstructure:"high_water_mark"`
}

// ModelConfig holds model management configuration
type ModelConfig struct {
	CacheDir   string `mapstructure:"cache_dir"`
	HFToken    string `mapstructure:"hf_token"`
	MaxCacheGB int    `mapstructure:"max_cache_gb"`
}

// Load loads configuration from a file
func Load(path string) (*Config, error) {
	v := viper.New()

	// Set defaults
	v.SetDefault("server.http_addr", "0.0.0.0:8080")
	v.SetDefault("server.metrics_addr", "0.0.0.0:9090")

	v.SetDefault("cluster.node_id", "coordinator")
	v.SetDefault("cluster.heartbeat_interval", "500ms")
	v.SetDefault("cluster.unhealthy_threshold", 3)
	v.SetDefault("cluster.reserved_vram_gb", 2.0)
	v.SetDefault("cluster.memory_per_layer_gb", 0.5)
	v.SetDefault("cluster.kv_reserve_tokens", 4096)
	// 512 GB: generous vs. current top GPUs (~192 GB H200) but still low
	// enough that a hostile worker can't dominate distribution with a lie.
	v.SetDefault("cluster.max_vram_gb", 512.0)
	v.SetDefault("cluster.register_token", "")

	v.SetDefault("auth.enabled", false)
	v.SetDefault("auth.rate_limit", 100)
	v.SetDefault("auth.rate_window", "1m")

	v.SetDefault("zmq.router_addr", "tcp://*:5555")
	v.SetDefault("zmq.metrics_addr", "tcp://*:5556")
	v.SetDefault("zmq.broadcast_addr", "tcp://*:5557")
	v.SetDefault("zmq.high_water_mark", 1000)

	// Four concurrent conversations is a deliberately conservative default:
	// on a cluster sized so one KV cache fits, four do not.
	v.SetDefault("cache.prefix_sessions", 4)
	v.SetDefault("cache.prefix_ttl", "30m")

	v.SetDefault("model.cache_dir", "~/.cache/hydra/models")
	v.SetDefault("model.max_cache_gb", 100)

	v.SetDefault("local_worker.enabled", false)
	v.SetDefault("local_worker.node_id", "local-worker")
	v.SetDefault("local_worker.device", "auto")
	v.SetDefault("local_worker.dtype", "bfloat16")

	// Logging: default to den-den-mushi JSON on stdout so a Dockerized deploy
	// is captured by the hub forwarder with zero extra wiring.
	v.SetDefault("log.format", "json")
	v.SetDefault("log.app", "hydra")

	// Config file is optional — callers often pass a conventional path even
	// when they want pure defaults. Missing is fine; malformed is not.
	if _, statErr := os.Stat(path); statErr == nil {
		v.SetConfigFile(path)
		v.SetConfigType("toml")
		if err := v.ReadInConfig(); err != nil {
			return nil, err
		}
	}

	// Environment variables. HYDRA_CLUSTER_REGISTER_TOKEN -> cluster.register_token.
	v.SetEnvPrefix("HYDRA")
	v.SetEnvKeyReplacer(strings.NewReplacer(".", "_"))
	v.AutomaticEnv()

	var cfg Config
	if err := v.Unmarshal(&cfg); err != nil {
		return nil, err
	}

	return &cfg, nil
}
