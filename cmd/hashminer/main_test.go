package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadDotEnv(t *testing.T) {
	path := filepath.Join(t.TempDir(), ".env")
	content := []byte("HASHMINER_TEST_RPC=https://example.invalid\nHASHMINER_TEST_KEEP=file\n")
	if err := os.WriteFile(path, content, 0o600); err != nil {
		t.Fatal(err)
	}

	t.Setenv("HASHMINER_TEST_RPC", "")
	t.Setenv("HASHMINER_TEST_KEEP", "env")

	if err := loadDotEnv(path); err != nil {
		t.Fatal(err)
	}
	if got := os.Getenv("HASHMINER_TEST_RPC"); got != "https://example.invalid" {
		t.Fatalf("HASHMINER_TEST_RPC = %q", got)
	}
	if got := os.Getenv("HASHMINER_TEST_KEEP"); got != "env" {
		t.Fatalf("HASHMINER_TEST_KEEP = %q", got)
	}
}
