package middleware

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

// captureLogs redirects the global logger to a buffer for the test's duration.
func captureLogs(t *testing.T) *bytes.Buffer {
	t.Helper()
	buf := &bytes.Buffer{}
	original := log.Logger
	log.Logger = zerolog.New(buf)
	t.Cleanup(func() { log.Logger = original })
	return buf
}

func TestLogger_RecordsRequestDetails(t *testing.T) {
	buf := captureLogs(t)

	r := gin.New()
	r.Use(Logger())
	r.GET("/thing", func(c *gin.Context) { c.String(http.StatusOK, "hello") })

	r.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest(http.MethodGet, "/thing?a=1", nil))

	var entry map[string]interface{}
	if err := json.Unmarshal(buf.Bytes(), &entry); err != nil {
		t.Fatalf("log line is not JSON: %v (%s)", err, buf.String())
	}
	if entry["method"] != "GET" {
		t.Errorf("method = %v, want GET", entry["method"])
	}
	if entry["path"] != "/thing?a=1" {
		t.Errorf("path = %v, want the query string appended", entry["path"])
	}
	if entry["status"].(float64) != 200 {
		t.Errorf("status = %v, want 200", entry["status"])
	}
	if entry["size"].(float64) != float64(len("hello")) {
		t.Errorf("size = %v, want 5", entry["size"])
	}
}

func TestLogger_LevelTracksStatusCode(t *testing.T) {
	cases := []struct {
		status int
		level  string
	}{
		{http.StatusOK, "info"},
		{http.StatusNotFound, "warn"},
		{http.StatusInternalServerError, "error"},
	}

	for _, tc := range cases {
		buf := captureLogs(t)

		r := gin.New()
		r.Use(Logger())
		r.GET("/thing", func(c *gin.Context) { c.Status(tc.status) })
		r.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest(http.MethodGet, "/thing", nil))

		var entry map[string]interface{}
		if err := json.Unmarshal(buf.Bytes(), &entry); err != nil {
			t.Fatalf("log line is not JSON: %v", err)
		}
		if entry["level"] != tc.level {
			t.Errorf("status %d logged at %v, want %s", tc.status, entry["level"], tc.level)
		}
	}
}

func TestCORS_SetsHeaders(t *testing.T) {
	r := gin.New()
	r.Use(CORS())
	r.GET("/thing", func(c *gin.Context) { c.Status(http.StatusOK) })

	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/thing", nil))

	if got := w.Header().Get("Access-Control-Allow-Origin"); got != "*" {
		t.Errorf("Allow-Origin = %q, want *", got)
	}
	if got := w.Header().Get("Access-Control-Allow-Headers"); got == "" {
		t.Error("Allow-Headers should be set")
	}
}

func TestCORS_PreflightShortCircuits(t *testing.T) {
	reached := false
	r := gin.New()
	r.Use(CORS())
	r.OPTIONS("/thing", func(c *gin.Context) {
		reached = true
		c.Status(http.StatusOK)
	})

	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest(http.MethodOptions, "/thing", nil))

	if w.Code != http.StatusNoContent {
		t.Errorf("preflight status = %d, want 204", w.Code)
	}
	if reached {
		t.Error("preflight should be answered by the middleware, not the handler")
	}
}
