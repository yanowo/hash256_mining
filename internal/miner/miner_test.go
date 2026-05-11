package miner

import (
	"bytes"
	"math/big"
	"testing"

	"github.com/ethereum/go-ethereum/accounts/abi"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/crypto"
)

func TestHashNonceMatchesSolidityABIEncode(t *testing.T) {
	challenge := common.HexToHash("0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef")
	nonce := uint64(42)

	bytes32Ty, err := abi.NewType("bytes32", "", nil)
	if err != nil {
		t.Fatal(err)
	}
	uintTy, err := abi.NewType("uint256", "", nil)
	if err != nil {
		t.Fatal(err)
	}
	args := abi.Arguments{
		{Type: bytes32Ty},
		{Type: uintTy},
	}

	encoded, err := args.Pack(challenge, new(big.Int).SetUint64(nonce))
	if err != nil {
		t.Fatal(err)
	}
	want := crypto.Keccak256Hash(encoded)
	got := HashNonce(challenge, nonce)

	if got != want {
		t.Fatalf("hash mismatch: got %s want %s", got.Hex(), want.Hex())
	}
}

func TestTargetBytes(t *testing.T) {
	target := new(big.Int).SetUint64(0x1020)
	got, err := TargetBytes(target)
	if err != nil {
		t.Fatal(err)
	}
	want := make([]byte, 32)
	want[30] = 0x10
	want[31] = 0x20
	if !bytes.Equal(got[:], want) {
		t.Fatalf("target mismatch: got %x want %x", got, want)
	}
}
