//go:build windows || ((linux || darwin) && cgo)

package cudaminer

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"math/big"
	"os"
	"runtime"
	"time"
	"unsafe"

	"hash256mining/internal/cuda"
	"hash256mining/internal/miner"

	"github.com/ethereum/go-ethereum/common"
)

type Job struct {
	Challenge      common.Hash
	Difficulty     *big.Int
	StartNonce     uint64
	DeviceIndex    int
	Blocks         uint32
	Threads        uint32
	Iterations     uint32
	ReportInterval time.Duration
}

type Progress = miner.Progress
type Result = miner.Result
type DeviceInfo = cuda.DeviceInfo

func ListDevices() ([]DeviceInfo, error) {
	return cuda.ListDevices()
}

func Search(ctx context.Context, job Job, onProgress func(Progress)) (Result, error) {
	if job.Difficulty == nil || job.Difficulty.Sign() <= 0 {
		return Result{}, fmt.Errorf("difficulty must be positive")
	}
	deviceIndex := job.DeviceIndex
	if deviceIndex < 0 {
		deviceIndex = 0
	}
	blocks := job.Blocks
	if blocks == 0 {
		blocks = 65536
	}
	threads := job.Threads
	if threads == 0 {
		threads = 128
	}
	iterations := job.Iterations
	if iterations == 0 {
		iterations = 16
	}
	if blocks == 0 || threads == 0 || iterations == 0 {
		return Result{}, fmt.Errorf("CUDA blocks, threads, and iterations must be positive")
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	runner, err := newRunner(deviceIndex)
	if err != nil {
		return Result{}, err
	}
	defer runner.close()

	if uint32(runner.info.MaxThreadsPerBlock) > 0 && threads > uint32(runner.info.MaxThreadsPerBlock) {
		return Result{}, fmt.Errorf("CUDA threads per block %d exceeds device max %d", threads, runner.info.MaxThreadsPerBlock)
	}
	if err := runner.selfTest(); err != nil {
		return Result{}, fmt.Errorf("CUDA self-test failed: %w", err)
	}

	return runner.search(ctx, job, blocks, threads, iterations, onProgress)
}

type runner struct {
	info         DeviceInfo
	ctx          cuda.Context
	module       cuda.Module
	function     cuda.Function
	challengeBuf cuda.DevicePtr
	targetBuf    cuda.DevicePtr
	resultBuf    cuda.DevicePtr
}

func newRunner(deviceIndex int) (*runner, error) {
	device, info, err := cuda.PickDevice(deviceIndex)
	if err != nil {
		return nil, err
	}
	ctx, err := cuda.CreateContext(device)
	if err != nil {
		return nil, err
	}
	r := &runner{info: info, ctx: ctx}
	defer func() {
		if err != nil {
			r.close()
		}
	}()

	arch := os.Getenv("HASHMINER_CUDA_ARCH")
	if arch == "" {
		arch = cudaArch(info.ComputeCapability)
	}
	ptx, err := cuda.CompilePTX(cudaKernel, "hash256_mine.cu", []string{
		"--gpu-architecture=" + arch,
		"--std=c++11",
	})
	if err != nil {
		return nil, err
	}
	r.module, err = cuda.LoadModuleData(ptx)
	if err != nil {
		return nil, err
	}
	r.function, err = cuda.ModuleFunction(r.module, "hash256_mine")
	if err != nil {
		return nil, err
	}
	r.challengeBuf, err = cuda.MemAlloc(8 * 4)
	if err != nil {
		return nil, err
	}
	r.targetBuf, err = cuda.MemAlloc(8 * 4)
	if err != nil {
		return nil, err
	}
	r.resultBuf, err = cuda.MemAlloc(12 * 4)
	if err != nil {
		return nil, err
	}
	return r, nil
}

func (r *runner) close() {
	if r.resultBuf != 0 {
		cuda.MemFree(r.resultBuf)
	}
	if r.targetBuf != 0 {
		cuda.MemFree(r.targetBuf)
	}
	if r.challengeBuf != 0 {
		cuda.MemFree(r.challengeBuf)
	}
	if r.module != 0 {
		cuda.UnloadModule(r.module)
	}
	if r.ctx != 0 {
		cuda.DestroyContext(r.ctx)
	}
}

func (r *runner) selfTest() error {
	var challenge common.Hash
	target := new(big.Int).Sub(new(big.Int).Lsh(big.NewInt(1), 256), big.NewInt(1))
	result, err := r.runBatch(challenge, target, 0, 1, 1, 1)
	if err != nil {
		return err
	}
	if result[0] == 0 {
		return fmt.Errorf("kernel did not find nonce 0 with max target")
	}
	gotNonce := uint64(result[2])<<32 | uint64(result[1])
	if gotNonce != 0 {
		return fmt.Errorf("kernel self-test nonce = %d, want 0", gotNonce)
	}
	gotHash := hashFromResult(result)
	wantHash := miner.HashNonce(challenge, 0)
	if !bytes.Equal(gotHash[:], wantHash[:]) {
		return fmt.Errorf("kernel hash = %s, want %s", gotHash.Hex(), wantHash.Hex())
	}
	return nil
}

func (r *runner) search(ctx context.Context, job Job, blocks uint32, threads uint32, iterations uint32, onProgress func(Progress)) (Result, error) {
	started := time.Now()
	lastReport := started
	base := job.StartNonce
	var hashes uint64
	batchSize := uint64(blocks) * uint64(threads) * uint64(iterations)
	if err := r.writeStaticInputs(job.Challenge, job.Difficulty); err != nil {
		return Result{}, err
	}

	for {
		select {
		case <-ctx.Done():
			return Result{}, miner.ErrStopped
		default:
		}

		result, err := r.runPreparedBatch(base, blocks, threads, iterations)
		if err != nil {
			return Result{}, err
		}
		hashes += batchSize
		if result[0] != 0 {
			nonce := uint64(result[2])<<32 | uint64(result[1])
			return Result{
				Backend: "cuda",
				Nonce:   nonce,
				Hash:    hashFromResult(result),
				Hashes:  hashes,
				Elapsed: time.Since(started),
			}, nil
		}

		next := base + batchSize
		if next < base {
			return Result{}, miner.ErrStopped
		}
		base = next

		now := time.Now()
		if onProgress != nil && now.Sub(lastReport) >= reportInterval(job.ReportInterval) {
			elapsed := now.Sub(started)
			onProgress(Progress{
				Hashes:   hashes,
				Hashrate: float64(hashes) / elapsed.Seconds(),
				Elapsed:  elapsed,
			})
			lastReport = now
		}
	}
}

func (r *runner) runBatch(challenge common.Hash, target *big.Int, base uint64, blocks uint32, threads uint32, iterations uint32) ([12]uint32, error) {
	if err := r.writeStaticInputs(challenge, target); err != nil {
		return [12]uint32{}, err
	}
	return r.runPreparedBatch(base, blocks, threads, iterations)
}

func (r *runner) writeStaticInputs(challenge common.Hash, target *big.Int) error {
	challengeWords := challengeLEWords(challenge)
	targetWords, err := targetBEWords(target)
	if err != nil {
		return err
	}

	if err := cuda.MemcpyHtoD(r.challengeBuf, unsafe.Pointer(&challengeWords[0]), 8*4); err != nil {
		return err
	}
	if err := cuda.MemcpyHtoD(r.targetBuf, unsafe.Pointer(&targetWords[0]), 8*4); err != nil {
		return err
	}
	return nil
}

func (r *runner) runPreparedBatch(base uint64, blocks uint32, threads uint32, iterations uint32) ([12]uint32, error) {
	result := [12]uint32{}

	if err := cuda.MemcpyHtoD(r.resultBuf, unsafe.Pointer(&result[0]), 12*4); err != nil {
		return result, err
	}

	challengePtr := uintptr(r.challengeBuf)
	targetPtr := uintptr(r.targetBuf)
	resultPtr := uintptr(r.resultBuf)
	if err := cuda.LaunchKernel(r.function, blocks, threads, []unsafe.Pointer{
		unsafe.Pointer(&challengePtr),
		unsafe.Pointer(&targetPtr),
		unsafe.Pointer(&base),
		unsafe.Pointer(&iterations),
		unsafe.Pointer(&resultPtr),
	}); err != nil {
		return result, err
	}
	if err := cuda.MemcpyDtoH(unsafe.Pointer(&result[0]), r.resultBuf, 12*4); err != nil {
		return result, err
	}
	return result, nil
}

func challengeLEWords(challenge common.Hash) [8]uint32 {
	var out [8]uint32
	for i := 0; i < 8; i++ {
		out[i] = binary.LittleEndian.Uint32(challenge[i*4 : i*4+4])
	}
	return out
}

func targetBEWords(target *big.Int) ([8]uint32, error) {
	targetBytes, err := miner.TargetBytes(target)
	if err != nil {
		return [8]uint32{}, err
	}
	var out [8]uint32
	for i := 0; i < 8; i++ {
		out[i] = binary.BigEndian.Uint32(targetBytes[i*4 : i*4+4])
	}
	return out, nil
}

func hashFromResult(result [12]uint32) common.Hash {
	var out common.Hash
	for i := 0; i < 8; i++ {
		binary.BigEndian.PutUint32(out[i*4:i*4+4], result[4+i])
	}
	return out
}

func reportInterval(v time.Duration) time.Duration {
	if v <= 0 {
		return 5 * time.Second
	}
	return v
}

func cudaArch(computeCapability string) string {
	var major, minor int
	if _, err := fmt.Sscanf(computeCapability, "%d.%d", &major, &minor); err != nil || major <= 0 {
		return "sm_75"
	}
	return fmt.Sprintf("sm_%d%d", major, minor)
}
