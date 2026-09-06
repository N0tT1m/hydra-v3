package api

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/N0tT1m/hydra-v3/internal/cluster"
	"github.com/N0tT1m/hydra-v3/internal/config"
	"github.com/N0tT1m/hydra-v3/internal/coordinator"
	"github.com/N0tT1m/hydra-v3/internal/testutil"
	"github.com/gin-gonic/gin"
)

// newTestServer builds a server over an in-memory transport.
func newTestServer(t *testing.T, mutate func(*config.Config)) (*Server, *coordinator.Coordinator) {
	t.Helper()

	cfg := &config.Config{
		Server: config.ServerConfig{HTTPAddr: "127.0.0.1:0"},
		Cluster: config.ClusterConfig{
			HeartbeatInterval:  10 * time.Millisecond,
			UnhealthyThreshold: 3,
			ReservedVRAMGB:     1,
			MemoryPerLayerGB:   0.5,
		},
	}
	if mutate != nil {
		mutate(cfg)
	}

	coord := coordinator.New(cfg, testutil.NewFakeBroker())
	return NewServer(coord, cfg), coord
}

// request runs one request through the server's router.
func request(t *testing.T, s *Server, method, path string, body string, headers map[string]string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(method, path, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	w := httptest.NewRecorder()
	s.engine.ServeHTTP(w, req)
	return w
}

func TestHealth_AlwaysOK(t *testing.T) {
	s, _ := newTestServer(t, nil)

	w := request(t, s, http.MethodGet, "/health", "", nil)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	var body map[string]string
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if body["status"] != "ok" {
		t.Errorf("body = %+v, want status ok", body)
	}
}

// /health answers even with no workers — it reports process liveness, while
// /ready reports whether the cluster can actually serve.
func TestReady_ReflectsHealthyWorkers(t *testing.T) {
	s, coord := newTestServer(t, nil)

	w := request(t, s, http.MethodGet, "/ready", "", nil)
	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("status with no workers = %d, want 503", w.Code)
	}

	coord.GetRegistry().Register(&cluster.Node{ID: "worker-1", Host: "127.0.0.1", VRAMGB: 16})

	w = request(t, s, http.MethodGet, "/ready", "", nil)
	if w.Code != http.StatusOK {
		t.Fatalf("status with a healthy worker = %d, want 200", w.Code)
	}
	var body map[string]interface{}
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if body["healthy_nodes"].(float64) != 1 {
		t.Errorf("healthy_nodes = %v, want 1", body["healthy_nodes"])
	}
}

// Every documented route must be registered. Asserting against gin's route
// table (rather than issuing requests) keeps a handler's own 404 from looking
// like a missing route.
func TestRoutesAreRegistered(t *testing.T) {
	s, _ := newTestServer(t, nil)

	registered := make(map[string]bool)
	for _, route := range s.engine.Routes() {
		registered[route.Method+" "+route.Path] = true
	}

	want := []string{
		"GET /health",
		"GET /ready",
		"GET /metrics",
		"POST /v1/chat/completions",
		"POST /v1/completions",
		"GET /v1/models",
		"POST /v1/vision/caption",
		"POST /v1/vision/validate",
		"POST /v1/vision/verify",
		"POST /v1/images/generate",
		"POST /api/models/load",
		"POST /api/models/unload",
		"POST /api/models/hot-swap",
		"GET /api/cluster/status",
		"POST /api/cluster/rebalance",
	}

	for _, route := range want {
		if !registered[route] {
			t.Errorf("%s is not registered", route)
		}
	}
}

func TestUnknownRouteIs404(t *testing.T) {
	s, _ := newTestServer(t, nil)
	if w := request(t, s, http.MethodGet, "/nope", "", nil); w.Code != http.StatusNotFound {
		t.Errorf("status = %d, want 404", w.Code)
	}
}

func TestAuthDisabled_ProtectedRoutesAreOpen(t *testing.T) {
	s, _ := newTestServer(t, nil)

	w := request(t, s, http.MethodGet, "/v1/models", "", nil)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200 when auth is disabled", w.Code)
	}
}

func TestAuthEnabled_ProtectedRoutesRequireAKey(t *testing.T) {
	s, _ := newTestServer(t, func(cfg *config.Config) {
		cfg.Auth = config.AuthConfig{
			Enabled:    true,
			APIKeys:    []string{"secret"},
			RateLimit:  100,
			RateWindow: time.Minute,
		}
	})

	if w := request(t, s, http.MethodGet, "/v1/models", "", nil); w.Code != http.StatusUnauthorized {
		t.Errorf("unauthenticated status = %d, want 401", w.Code)
	}

	headers := map[string]string{"Authorization": "Bearer secret"}
	if w := request(t, s, http.MethodGet, "/v1/models", "", headers); w.Code != http.StatusOK {
		t.Errorf("authenticated status = %d, want 200", w.Code)
	}
}

// Health and metrics stay open when auth is on, so orchestrators and scrapers
// don't need credentials.
func TestAuthEnabled_HealthAndMetricsStayOpen(t *testing.T) {
	s, _ := newTestServer(t, func(cfg *config.Config) {
		cfg.Auth = config.AuthConfig{
			Enabled: true, APIKeys: []string{"secret"},
			RateLimit: 100, RateWindow: time.Minute,
		}
	})

	for _, path := range []string{"/health", "/ready", "/metrics"} {
		w := request(t, s, http.MethodGet, path, "", nil)
		if w.Code == http.StatusUnauthorized {
			t.Errorf("%s requires auth but should not", path)
		}
	}
}

func TestAuthEnabled_RateLimitApplies(t *testing.T) {
	s, _ := newTestServer(t, func(cfg *config.Config) {
		cfg.Auth = config.AuthConfig{
			Enabled: true, APIKeys: []string{"secret"},
			RateLimit: 2, RateWindow: time.Minute,
		}
	})

	headers := map[string]string{"Authorization": "Bearer secret"}
	var last int
	for i := 0; i < 4; i++ {
		last = request(t, s, http.MethodGet, "/v1/models", "", headers).Code
	}
	if last != http.StatusTooManyRequests {
		t.Errorf("status after exceeding the limit = %d, want 429", last)
	}
}

func TestCORSHeadersOnEveryResponse(t *testing.T) {
	s, _ := newTestServer(t, nil)

	w := request(t, s, http.MethodGet, "/health", "", nil)

	if got := w.Header().Get("Access-Control-Allow-Origin"); got != "*" {
		t.Errorf("Allow-Origin = %q, want *", got)
	}
}

// A panicking handler must produce a 500, not take the process down.
func TestRecoveryMiddlewareIsInstalled(t *testing.T) {
	s, _ := newTestServer(t, nil)
	s.engine.GET("/boom", func(c *gin.Context) { panic("boom") })

	w := request(t, s, http.MethodGet, "/boom", "", nil)

	if w.Code != http.StatusInternalServerError {
		t.Errorf("status = %d, want 500 from the recovery middleware", w.Code)
	}
}

// --- lifecycle --------------------------------------------------------------

func TestRunAndShutdown(t *testing.T) {
	s, _ := newTestServer(t, nil)

	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := listener.Addr().String()
	listener.Close()

	errCh := make(chan error, 1)
	go func() { errCh <- s.Run(addr) }()

	// Wait for the listener to come up.
	client := &http.Client{Timeout: time.Second}
	url := fmt.Sprintf("http://%s/health", addr)
	var resp *http.Response
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		resp, err = client.Get(url)
		if err == nil {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if err != nil {
		t.Fatalf("server never became reachable: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("GET /health = %d, want 200", resp.StatusCode)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := s.Shutdown(ctx); err != nil {
		t.Fatalf("Shutdown: %v", err)
	}

	// A clean shutdown is reported as success, not http.ErrServerClosed.
	select {
	case err := <-errCh:
		if err != nil {
			t.Errorf("Run returned %v after a clean shutdown, want nil", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Run did not return after Shutdown")
	}
}

func TestShutdownBeforeRunIsANoop(t *testing.T) {
	s, _ := newTestServer(t, nil)
	if err := s.Shutdown(context.Background()); err != nil {
		t.Errorf("Shutdown on a server that never ran = %v, want nil", err)
	}
}

func TestRun_ReportsListenError(t *testing.T) {
	s, _ := newTestServer(t, nil)

	// Occupy the port so ListenAndServe fails.
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()

	if err := s.Run(listener.Addr().String()); err == nil {
		t.Fatal("Run on a busy port should return an error")
	}
}
