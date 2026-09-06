package middleware

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strconv"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
)

func init() {
	gin.SetMode(gin.TestMode)
}

// authRouter builds a router whose only protected route records whether the
// handler was reached.
func authRouter(keys []string, reached *bool) *gin.Engine {
	r := gin.New()
	r.Use(Auth(keys))
	r.GET("/protected", func(c *gin.Context) {
		*reached = true
		c.JSON(http.StatusOK, gin.H{"ok": true})
	})
	return r
}

func TestAuth_AcceptsValidBearerToken(t *testing.T) {
	reached := false
	r := authRouter([]string{"key-a", "key-b"}, &reached)

	req := httptest.NewRequest(http.MethodGet, "/protected", nil)
	req.Header.Set("Authorization", "Bearer key-b")
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	if !reached {
		t.Error("handler was not reached with a valid key")
	}
}

func TestAuth_AcceptsCaseInsensitiveScheme(t *testing.T) {
	reached := false
	r := authRouter([]string{"key-a"}, &reached)

	req := httptest.NewRequest(http.MethodGet, "/protected", nil)
	req.Header.Set("Authorization", "bearer key-a")
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	if w.Code != http.StatusOK || !reached {
		t.Errorf("lowercase scheme rejected: status=%d reached=%v", w.Code, reached)
	}
}

func TestAuth_Rejects(t *testing.T) {
	cases := []struct {
		name   string
		header string
	}{
		{"missing header", ""},
		{"no scheme", "key-a"},
		{"wrong scheme", "Basic key-a"},
		{"unknown key", "Bearer nope"},
		{"empty token", "Bearer "},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			reached := false
			r := authRouter([]string{"key-a"}, &reached)

			req := httptest.NewRequest(http.MethodGet, "/protected", nil)
			if tc.header != "" {
				req.Header.Set("Authorization", tc.header)
			}
			w := httptest.NewRecorder()
			r.ServeHTTP(w, req)

			if w.Code != http.StatusUnauthorized {
				t.Errorf("status = %d, want 401", w.Code)
			}
			if reached {
				t.Error("handler must not run for a rejected request")
			}

			var body map[string]map[string]string
			if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
				t.Fatalf("response is not the error envelope: %v", err)
			}
			if body["error"]["message"] == "" {
				t.Error("rejection should explain itself")
			}
		})
	}
}

func TestAuth_StoresKeyForDownstreamMiddleware(t *testing.T) {
	var seen string
	r := gin.New()
	r.Use(Auth([]string{"key-a"}))
	r.GET("/protected", func(c *gin.Context) {
		seen = c.GetString("api_key")
		c.Status(http.StatusOK)
	})

	req := httptest.NewRequest(http.MethodGet, "/protected", nil)
	req.Header.Set("Authorization", "Bearer key-a")
	r.ServeHTTP(httptest.NewRecorder(), req)

	if seen != "key-a" {
		t.Errorf("api_key in context = %q, want key-a", seen)
	}
}

func TestAuth_NoConfiguredKeysRejectsEverything(t *testing.T) {
	reached := false
	r := authRouter(nil, &reached)

	req := httptest.NewRequest(http.MethodGet, "/protected", nil)
	req.Header.Set("Authorization", "Bearer anything")
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	if w.Code != http.StatusUnauthorized || reached {
		t.Error("with no keys configured, every key must be rejected")
	}
}

// --- rate limiting ---------------------------------------------------------

func rateLimitRouter(limit int, window time.Duration) *gin.Engine {
	r := gin.New()
	r.Use(RateLimit(limit, window))
	r.GET("/limited", func(c *gin.Context) { c.Status(http.StatusOK) })
	return r
}

func TestRateLimit_AllowsUpToTheBurstThenRejects(t *testing.T) {
	r := rateLimitRouter(3, time.Minute)

	for i := 0; i < 3; i++ {
		w := httptest.NewRecorder()
		r.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/limited", nil))
		if w.Code != http.StatusOK {
			t.Fatalf("request %d: status = %d, want 200", i+1, w.Code)
		}
	}

	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/limited", nil))
	if w.Code != http.StatusTooManyRequests {
		t.Fatalf("status = %d, want 429 once the burst is spent", w.Code)
	}
	if w.Header().Get("Retry-After") == "" {
		t.Error("a 429 should tell the client when to retry")
	}
}

// The remaining-count header must be a number. Formatting it with
// string(rune(n)) — the previous implementation — emitted a control character.
func TestRateLimit_RemainingHeaderIsANumber(t *testing.T) {
	r := rateLimitRouter(5, time.Minute)

	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/limited", nil))

	got := w.Header().Get("X-RateLimit-Remaining")
	n, err := strconv.Atoi(got)
	if err != nil {
		t.Fatalf("X-RateLimit-Remaining = %q, which is not a number: %v", got, err)
	}
	if n < 0 || n > 5 {
		t.Errorf("remaining = %d, want between 0 and 5", n)
	}
}

func TestRateLimit_IsPerAPIKey(t *testing.T) {
	r := gin.New()
	r.Use(Auth([]string{"key-a", "key-b"}))
	r.Use(RateLimit(1, time.Minute))
	r.GET("/limited", func(c *gin.Context) { c.Status(http.StatusOK) })

	call := func(key string) int {
		req := httptest.NewRequest(http.MethodGet, "/limited", nil)
		req.Header.Set("Authorization", "Bearer "+key)
		w := httptest.NewRecorder()
		r.ServeHTTP(w, req)
		return w.Code
	}

	if code := call("key-a"); code != http.StatusOK {
		t.Fatalf("first call for key-a = %d, want 200", code)
	}
	if code := call("key-a"); code != http.StatusTooManyRequests {
		t.Fatalf("second call for key-a = %d, want 429", code)
	}
	// A different key has its own bucket.
	if code := call("key-b"); code != http.StatusOK {
		t.Errorf("first call for key-b = %d, want 200 — buckets must be per key", code)
	}
}

func TestRateLimit_RefillsOverTime(t *testing.T) {
	// 10 requests per 100ms => one token roughly every 10ms.
	r := rateLimitRouter(1, 100*time.Millisecond)

	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/limited", nil))
	if w.Code != http.StatusOK {
		t.Fatalf("first request = %d, want 200", w.Code)
	}

	w = httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/limited", nil))
	if w.Code != http.StatusTooManyRequests {
		t.Fatalf("second immediate request = %d, want 429", w.Code)
	}

	time.Sleep(150 * time.Millisecond)
	w = httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/limited", nil))
	if w.Code != http.StatusOK {
		t.Errorf("request after the window = %d, want 200", w.Code)
	}
}
