package config

import (
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
	LocalWorker LocalWorkerConfig `mapstructure:"local_worker"`
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
	MemoryPerLayerGB   float64       `mapstructure:"memory_per_layer_gb"`
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
	RouterAddr   string `mapstructure:"router_addr"`
	MetricsAddr  string `mapstructure:"metrics_addr"`
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

	v.SetDefault("auth.enabled", false)
	v.SetDefault("auth.rate_limit", 100)
	v.SetDefault("auth.rate_window", "1m")

	v.SetDefault("zmq.router_addr", "tcp://*:5555")
	v.SetDefault("zmq.metrics_addr", "tcp://*:5556")
	v.SetDefault("zmq.broadcast_addr", "tcp://*:5557")
	v.SetDefault("zmq.high_water_mark", 1000)

	v.SetDefault("model.cache_dir", "~/.cache/hydra/models")
	v.SetDefault("model.max_cache_gb", 100)

	v.SetDefault("local_worker.enabled", false)
	v.SetDefault("local_worker.node_id", "local-worker")
	v.SetDefault("local_worker.device", "auto")
	v.SetDefault("local_worker.dtype", "bfloat16")

	// Read config file
	v.SetConfigFile(path)
	v.SetConfigType("toml")

	if err := v.ReadInConfig(); err != nil {
		// Config file is optional, use defaults
		if _, ok := err.(viper.ConfigFileNotFoundError); !ok {
			return nil, err
		}
	}

	// Environment variables
	v.SetEnvPrefix("HYDRA")
	v.AutomaticEnv()

	var cfg Config
	if err := v.Unmarshal(&cfg); err != nil {
		return nil, err
	}

	return &cfg, nil
}
