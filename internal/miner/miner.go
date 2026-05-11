package miner

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math/big"
	"runtime"
	"sync/atomic"
	"time"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/crypto"
)

var ErrStopped = errors.New("miner stopped")

type Job struct {
	Challenge      common.Hash
	Difficulty     *big.Int
	StartNonce     uint64
	Threads        int
	ReportInterval time.Duration
}

type Progress struct {
	Hashes   uint64
	Hashrate float64
	Elapsed  time.Duration
}

type Result struct {
	Backend string
	Nonce   uint64
	Hash    common.Hash
	Hashes  uint64
	Elapsed time.Duration
}

func Search(ctx context.Context, job Job, onProgress func(Progress)) (Result, error) {
	if job.Difficulty == nil || job.Difficulty.Sign() <= 0 {
		return Result{}, fmt.Errorf("difficulty must be positive")
	}
	target, err := TargetBytes(job.Difficulty)
	if err != nil {
		return Result{}, err
	}
	threads := job.Threads
	if threads <= 0 {
		threads = runtime.NumCPU()
	}
	interval := job.ReportInterval
	if interval <= 0 {
		interval = 5 * time.Second
	}

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	started := time.Now()
	found := make(chan Result, 1)
	var hashes atomic.Uint64

	for worker := 0; worker < threads; worker++ {
		go func(worker int) {
			var buf [64]byte
			copy(buf[:32], job.Challenge[:])

			nonce := job.StartNonce + uint64(worker)
			step := uint64(threads)
			var local uint64

			for {
				select {
				case <-ctx.Done():
					if local > 0 {
						hashes.Add(local)
					}
					return
				default:
				}

				for i := 0; i < 4096; i++ {
					binary.BigEndian.PutUint64(buf[56:64], nonce)
					hash := crypto.Keccak256Hash(buf[:])
					local++

					if bytes.Compare(hash[:], target[:]) < 0 {
						total := hashes.Add(local)
						select {
						case found <- Result{
							Nonce:   nonce,
							Hash:    hash,
							Hashes:  total,
							Elapsed: time.Since(started),
						}:
							cancel()
						default:
						}
						return
					}

					next := nonce + step
					if next < nonce {
						if local > 0 {
							hashes.Add(local)
						}
						return
					}
					nonce = next
				}

				if local > 0 {
					hashes.Add(local)
					local = 0
				}
			}
		}(worker)
	}

	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case result := <-found:
			return result, nil
		case <-ticker.C:
			if onProgress != nil {
				elapsed := time.Since(started)
				count := hashes.Load()
				onProgress(Progress{
					Hashes:   count,
					Hashrate: float64(count) / elapsed.Seconds(),
					Elapsed:  elapsed,
				})
			}
		case <-ctx.Done():
			return Result{}, ErrStopped
		}
	}
}

func HashNonce(challenge common.Hash, nonce uint64) common.Hash {
	var buf [64]byte
	copy(buf[:32], challenge[:])
	binary.BigEndian.PutUint64(buf[56:64], nonce)
	return crypto.Keccak256Hash(buf[:])
}

func TargetBytes(target *big.Int) ([32]byte, error) {
	var out [32]byte
	if target == nil || target.Sign() <= 0 {
		return out, fmt.Errorf("difficulty must be positive")
	}
	if target.BitLen() > 256 {
		return out, fmt.Errorf("difficulty exceeds uint256")
	}
	bytes := target.Bytes()
	copy(out[32-len(bytes):], bytes)
	return out, nil
}

func TargetHex(target *big.Int) (string, error) {
	bytes, err := TargetBytes(target)
	if err != nil {
		return "", err
	}
	return "0x" + common.Bytes2Hex(bytes[:]), nil
}

func ExpectedHashes(target *big.Int) *big.Float {
	two256 := new(big.Int).Lsh(big.NewInt(1), 256)
	out := new(big.Float).SetPrec(256).SetInt(two256)
	return out.Quo(out, new(big.Float).SetPrec(256).SetInt(target))
}
