package main

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/hydra-v3/internal/config"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

func TestDeriveModelID(t *testing.T) {
	cases := []struct {
		in   string
		want string
	}{
		{"meta-llama/Llama-2-7b-hf", "llama-2-7b-hf"},
		{"Qwen/Qwen2.5-7B-Instruct", "qwen2.5-7b-instruct"},
		{"tinyllama", "tinyllama"},
		{"org/sub/Model-X", "model-x"},
		{"org/model/", "model"},
		{"", ""},
	}

	for _, tc := range cases {
		if got := deriveModelID(tc.in); got != tc.want {
			t.Errorf("deriveModelID(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

func TestWorkerCoordinatorAddr(t *testing.T) {
	cases := []struct {
		in   string
		want string
	}{
		{"tcp://*:5555", "tcp://localhost:5555"},
		// The rewrite must not be pinned to the default port.
		{"tcp://*:15555", "tcp://localhost:15555"},
		{"tcp://0.0.0.0:5555", "tcp://localhost:5555"},
		{"tcp://[::]:5555", "tcp://localhost:5555"},
		// A concrete host is already dialable and must be left alone.
		{"tcp://192.168.1.10:5555", "tcp://192.168.1.10:5555"},
		{"tcp://localhost:5555", "tcp://localhost:5555"},
	}

	for _, tc := range cases {
		if got := workerCoordinatorAddr(tc.in); got != tc.want {
			t.Errorf("workerCoordinatorAddr(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

func TestResolveWorkerSettings(t *testing.T) {
	flags := workerSettings{NodeID: "cli-node", Device: "auto", Dtype: "bfloat16"}

	t.Run("cli flag wins over config", func(t *testing.T) {
		cfg := config.LocalWorkerConfig{Enabled: true, NodeID: "cfg-node", Device: "cuda:0", Dtype: "int8"}
		got := resolveWorkerSettings(true, cfg, flags)
		if got != flags {
			t.Errorf("settings = %+v, want the CLI values %+v", got, flags)
		}
	})

	t.Run("config fills in when only config enabled it", func(t *testing.T) {
		cfg := config.LocalWorkerConfig{Enabled: true, NodeID: "cfg-node", Device: "cuda:0", Dtype: "int8"}
		got := resolveWorkerSettings(false, cfg, flags)
		want := workerSettings{NodeID: "cfg-node", Device: "cuda:0", Dtype: "int8"}
		if got != want {
			t.Errorf("settings = %+v, want %+v", got, want)
		}
	})

	t.Run("empty config fields keep the flag defaults", func(t *testing.T) {
		cfg := config.LocalWorkerConfig{Enabled: true}
		got := resolveWorkerSettings(false, cfg, flags)
		if got != flags {
			t.Errorf("settings = %+v, want the flag defaults %+v", got, flags)
		}
	})

	t.Run("worker disabled entirely", func(t *testing.T) {
		got := resolveWorkerSettings(false, config.LocalWorkerConfig{}, flags)
		if got != flags {
			t.Errorf("settings = %+v, want the flag defaults", got)
		}
	})
}

func TestApplyLogging_JSONUsesDenDenMushiKeys(t *testing.T) {
	original := log.Logger
	defer func() {
		log.Logger = original
		zerolog.TimestampFieldName = "time"
		zerolog.MessageFieldName = "message"
	}()

	applyLogging(config.LogConfig{Format: "json", App: "test-app"})

	buf := &bytes.Buffer{}
	log.Logger = log.Output(buf)
	log.Info().Str("extra", "value").Msg("hello")

	var entry map[string]interface{}
	if err := json.Unmarshal(buf.Bytes(), &entry); err != nil {
		t.Fatalf("log line is not JSON: %v (%s)", err, buf.String())
	}
	for _, key := range []string{"ts", "msg", "level", "app", "host"} {
		if _, ok := entry[key]; !ok {
			t.Errorf("log entry is missing the %q field: %+v", key, entry)
		}
	}
	if entry["msg"] != "hello" {
		t.Errorf("msg = %v, want hello", entry["msg"])
	}
	if entry["app"] != "test-app" {
		t.Errorf("app = %v, want test-app", entry["app"])
	}
}

func TestApplyLogging_DefaultsAppName(t *testing.T) {
	original := log.Logger
	defer func() { log.Logger = original }()

	applyLogging(config.LogConfig{Format: "json"})

	buf := &bytes.Buffer{}
	log.Logger = log.Output(buf)
	log.Info().Msg("x")

	var entry map[string]interface{}
	if err := json.Unmarshal(buf.Bytes(), &entry); err != nil {
		t.Fatal(err)
	}
	if entry["app"] != "robin-hydra" {
		t.Errorf("app = %v, want the robin-hydra default", entry["app"])
	}
}

func TestApplyLogging_ConsoleFormatIsNotJSON(t *testing.T) {
	original := log.Logger
	defer func() { log.Logger = original }()

	// Console mode writes to stderr; we only assert it doesn't panic and
	// that the logger still works.
	applyLogging(config.LogConfig{Format: "console", App: "test-app"})
	log.Info().Msg("console line")
}

func TestSleepCtx(t *testing.T) {
	if !sleepCtx(context.Background(), time.Millisecond) {
		t.Error("sleepCtx should report true when the sleep completes")
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if sleepCtx(ctx, time.Hour) {
		t.Error("sleepCtx should report false when the context is already cancelled")
	}
}

// fakeRegistry stands in for the cluster registry in the startup waits.
type fakeRegistry struct {
	nodePresent  atomic.Bool
	healthyCount atomic.Int32
}

func (f *fakeRegistry) HasNode(string) bool   { return f.nodePresent.Load() }
func (f *fakeRegistry) HealthyNodeCount() int { return int(f.healthyCount.Load()) }

func TestWaitForNode_ReturnsWhenNodeAppears(t *testing.T) {
	reg := &fakeRegistry{}
	reg.nodePresent.Store(true)

	if !waitForNode(context.Background(), reg, "worker-1", time.Second) {
		t.Error("waitForNode should return true for a node that is already present")
	}
}

func TestWaitForNode_TimesOut(t *testing.T) {
	reg := &fakeRegistry{}
	start := time.Now()

	if waitForNode(context.Background(), reg, "worker-1", 10*time.Millisecond) {
		t.Error("waitForNode should return false when the node never registers")
	}
	if elapsed := time.Since(start); elapsed > 5*time.Second {
		t.Errorf("waitForNode took %v, far longer than its deadline", elapsed)
	}
}

func TestWaitForNode_StopsOnContextCancellation(t *testing.T) {
	reg := &fakeRegistry{}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if waitForNode(ctx, reg, "worker-1", time.Hour) {
		t.Error("a cancelled context should end the wait as a failure")
	}
}

func TestWaitForHealthyWorker(t *testing.T) {
	reg := &fakeRegistry{}
	reg.healthyCount.Store(1)
	if !waitForHealthyWorker(context.Background(), reg, time.Second) {
		t.Error("should return true when a healthy worker is already present")
	}

	empty := &fakeRegistry{}
	if waitForHealthyWorker(context.Background(), empty, 10*time.Millisecond) {
		t.Error("should return false when no worker ever becomes healthy")
	}
}

func TestFindWorkerDir_FindsWorkerRelativeToCwd(t *testing.T) {
	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, "worker"), 0o755); err != nil {
		t.Fatal(err)
	}

	original, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(dir); err != nil {
		t.Fatal(err)
	}
	defer os.Chdir(original)

	got := findWorkerDir()
	if got == "" {
		t.Fatal("findWorkerDir found nothing next to the working directory")
	}
	if !strings.HasSuffix(got, "worker") {
		t.Errorf("findWorkerDir = %q, want a path ending in worker", got)
	}
}

// findWorkerDir walks three fallbacks in order: next to the executable, next
// to the working directory, then a short list of relative candidates. The
// executable branch cannot be steered from a test (os.Executable reports the
// test binary), so these cover the rest.

func TestFindWorkerDir_ReturnsEmptyWhenNothingIsFound(t *testing.T) {
	dir := t.TempDir()

	original, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(dir); err != nil {
		t.Fatal(err)
	}
	defer os.Chdir(original)

	if got := findWorkerDir(); got != "" {
		t.Errorf("expected no worker dir under an empty tree, got %q", got)
	}
}

func TestFindWorkerDir_FindsWorkerOneLevelUp(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "worker"), 0o755); err != nil {
		t.Fatal(err)
	}
	nested := filepath.Join(root, "build")
	if err := os.MkdirAll(nested, 0o755); err != nil {
		t.Fatal(err)
	}

	original, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(nested); err != nil {
		t.Fatal(err)
	}
	defer os.Chdir(original)

	got := findWorkerDir()
	if got == "" {
		t.Fatal("findWorkerDir missed ../worker")
	}
	// macOS temp dirs are symlinked through /private, so compare resolved paths.
	wantResolved, _ := filepath.EvalSymlinks(filepath.Join(root, "worker"))
	gotResolved, _ := filepath.EvalSymlinks(got)
	if gotResolved != wantResolved {
		t.Errorf("findWorkerDir() = %q, want %q", gotResolved, wantResolved)
	}
}

func TestStartLocalWorker_ReturnsNilWhenTheWorkerDirIsMissing(t *testing.T) {
	dir := t.TempDir()

	original, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(dir); err != nil {
		t.Fatal(err)
	}
	defer os.Chdir(original)

	if cmd := startLocalWorker("n1", "cpu", "float16", "tcp://127.0.0.1:5555"); cmd != nil {
		t.Fatal("a missing worker directory should not produce a process")
	}
}

// fakePython writes an executable stub at the venv path startLocalWorker
// looks for, so the spawned process is a shell script we control rather than
// a real interpreter. It records its argv to argsFile.
func fakePython(t *testing.T, workerDir, argsFile string) {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("venv layout and shell stub differ on Windows")
	}
	binDir := filepath.Join(workerDir, "venv", "bin")
	if err := os.MkdirAll(binDir, 0o755); err != nil {
		t.Fatal(err)
	}
	script := "#!/bin/sh\nprintf '%s\\n' \"$@\" > " + argsFile + "\n"
	if err := os.WriteFile(filepath.Join(binDir, "python"), []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
}

func TestStartLocalWorker_SpawnsThePythonWorkerWithItsSettings(t *testing.T) {
	root := t.TempDir()
	workerDir := filepath.Join(root, "worker")
	if err := os.MkdirAll(workerDir, 0o755); err != nil {
		t.Fatal(err)
	}
	argsFile := filepath.Join(root, "argv")
	fakePython(t, workerDir, argsFile)

	original, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(root); err != nil {
		t.Fatal(err)
	}
	defer os.Chdir(original)

	cmd := startLocalWorker("node-7", "mps", "int8", "tcp://0.0.0.0:5555")
	if cmd == nil {
		t.Fatal("startLocalWorker returned no process")
	}
	t.Cleanup(func() {
		if cmd.Process != nil {
			cmd.Process.Kill()
		}
	})
	if err := cmd.Wait(); err != nil {
		t.Fatalf("worker stub exited badly: %v", err)
	}

	raw, err := os.ReadFile(argsFile)
	if err != nil {
		t.Fatalf("stub did not record its arguments: %v", err)
	}
	argv := strings.Fields(string(raw))

	for _, want := range []string{"hydra_worker", "start", "node-7", "mps", "int8"} {
		if !slices.Contains(argv, want) {
			t.Errorf("argv %v is missing %q", argv, want)
		}
	}
}

func TestStartLocalWorker_RewritesAWildcardCoordinatorAddress(t *testing.T) {
	root := t.TempDir()
	workerDir := filepath.Join(root, "worker")
	if err := os.MkdirAll(workerDir, 0o755); err != nil {
		t.Fatal(err)
	}
	argsFile := filepath.Join(root, "argv")
	fakePython(t, workerDir, argsFile)

	original, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(root); err != nil {
		t.Fatal(err)
	}
	defer os.Chdir(original)

	// The coordinator binds 0.0.0.0, but a worker cannot dial that — it has to
	// be handed a routable address.
	cmd := startLocalWorker("node-1", "cpu", "float16", "tcp://0.0.0.0:5555")
	if cmd == nil {
		t.Fatal("startLocalWorker returned no process")
	}
	t.Cleanup(func() {
		if cmd.Process != nil {
			cmd.Process.Kill()
		}
	})
	if err := cmd.Wait(); err != nil {
		t.Fatalf("worker stub exited badly: %v", err)
	}

	raw, _ := os.ReadFile(argsFile)
	argv := strings.Fields(string(raw))
	want := workerCoordinatorAddr("tcp://0.0.0.0:5555")
	if !slices.Contains(argv, want) {
		t.Errorf("argv %v does not carry the rewritten address %q", argv, want)
	}
}

func TestStartLocalWorker_RunsInTheWorkerDirectory(t *testing.T) {
	root := t.TempDir()
	workerDir := filepath.Join(root, "worker")
	if err := os.MkdirAll(workerDir, 0o755); err != nil {
		t.Fatal(err)
	}
	fakePython(t, workerDir, filepath.Join(root, "argv"))

	original, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(root); err != nil {
		t.Fatal(err)
	}
	defer os.Chdir(original)

	cmd := startLocalWorker("node-1", "cpu", "float16", "tcp://127.0.0.1:5555")
	if cmd == nil {
		t.Fatal("startLocalWorker returned no process")
	}
	t.Cleanup(func() {
		if cmd.Process != nil {
			cmd.Process.Kill()
		}
	})
	cmd.Wait()

	if filepath.Base(cmd.Dir) != "worker" {
		t.Errorf("cmd.Dir = %q, want the worker directory", cmd.Dir)
	}
}

func TestStartLocalWorker_FallsBackToSystemPythonWithoutAVenv(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("venv layout differs on Windows")
	}
	root := t.TempDir()
	// A worker directory with no venv inside it.
	if err := os.MkdirAll(filepath.Join(root, "worker"), 0o755); err != nil {
		t.Fatal(err)
	}

	original, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(root); err != nil {
		t.Fatal(err)
	}
	defer os.Chdir(original)

	cmd := startLocalWorker("node-1", "cpu", "float16", "tcp://127.0.0.1:5555")
	if cmd == nil {
		t.Skip("no system python3 on PATH to fall back to")
	}
	t.Cleanup(func() {
		if cmd.Process != nil {
			cmd.Process.Kill()
		}
	})
	cmd.Wait()

	if filepath.Base(cmd.Path) != "python3" {
		t.Errorf("cmd.Path = %q, want the system python3", cmd.Path)
	}
}
