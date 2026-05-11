package gpuminer

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math/big"
	"time"
	"unsafe"

	"hash256mining/internal/miner"
	"hash256mining/internal/opencl"

	"github.com/ethereum/go-ethereum/common"
)

type Job struct {
	Challenge      common.Hash
	Difficulty     *big.Int
	StartNonce     uint64
	DeviceIndex    int
	Workgroups     uint32
	LocalSize      uint32
	Iterations     uint32
	ReportInterval time.Duration
}

type Progress = miner.Progress
type Result = miner.Result

type DeviceInfo = opencl.DeviceInfo

func ListDevices() ([]DeviceInfo, error) {
	return opencl.ListGPUDevices()
}

func Search(ctx context.Context, job Job, onProgress func(Progress)) (Result, error) {
	if job.Difficulty == nil || job.Difficulty.Sign() <= 0 {
		return Result{}, fmt.Errorf("difficulty must be positive")
	}
	deviceIndex := job.DeviceIndex
	if deviceIndex < 0 {
		deviceIndex = 0
	}
	localSize := job.LocalSize
	if localSize == 0 {
		localSize = 64
	}
	workgroups := job.Workgroups
	if workgroups == 0 {
		workgroups = 16384
	}
	iterations := job.Iterations
	if iterations == 0 {
		iterations = 16
	}
	if workgroups == 0 || localSize == 0 || iterations == 0 {
		return Result{}, fmt.Errorf("GPU workgroups, local size, and iterations must be positive")
	}

	runner, err := newRunner(deviceIndex)
	if err != nil {
		return Result{}, err
	}
	defer runner.close()

	if err := runner.selfTest(); err != nil {
		return Result{}, fmt.Errorf("GPU self-test failed: %w", err)
	}

	return runner.search(ctx, job, workgroups, localSize, iterations, onProgress)
}

type runner struct {
	info         DeviceInfo
	ctx          opencl.Context
	queue        opencl.CommandQueue
	program      opencl.Program
	kernel       opencl.Kernel
	challengeBuf opencl.Mem
	targetBuf    opencl.Mem
	resultBuf    opencl.Mem
}

func newRunner(deviceIndex int) (*runner, error) {
	_, device, info, err := opencl.PickGPU(deviceIndex)
	if err != nil {
		return nil, err
	}
	ctx, err := opencl.CreateContext(device)
	if err != nil {
		return nil, err
	}
	r := &runner{info: info, ctx: ctx}
	defer func() {
		if err != nil {
			r.close()
		}
	}()

	r.queue, err = opencl.CreateCommandQueue(ctx, device)
	if err != nil {
		return nil, err
	}
	r.program, err = opencl.CreateProgramWithSource(ctx, openCLKernel)
	if err != nil {
		return nil, err
	}
	if err = opencl.BuildProgram(r.program, device, ""); err != nil {
		return nil, err
	}
	r.kernel, err = opencl.CreateKernel(r.program, "hash256_mine")
	if err != nil {
		return nil, err
	}
	r.challengeBuf, err = opencl.CreateBuffer(ctx, opencl.MemReadOnly, 8*4)
	if err != nil {
		return nil, err
	}
	r.targetBuf, err = opencl.CreateBuffer(ctx, opencl.MemReadOnly, 8*4)
	if err != nil {
		return nil, err
	}
	r.resultBuf, err = opencl.CreateBuffer(ctx, opencl.MemReadWrite, 12*4)
	if err != nil {
		return nil, err
	}
	if err = opencl.SetKernelMemArg(r.kernel, 0, r.challengeBuf); err != nil {
		return nil, err
	}
	if err = opencl.SetKernelMemArg(r.kernel, 1, r.targetBuf); err != nil {
		return nil, err
	}
	if err = opencl.SetKernelMemArg(r.kernel, 4, r.resultBuf); err != nil {
		return nil, err
	}
	return r, nil
}

func (r *runner) close() {
	if r.resultBuf != 0 {
		opencl.ReleaseMem(r.resultBuf)
	}
	if r.targetBuf != 0 {
		opencl.ReleaseMem(r.targetBuf)
	}
	if r.challengeBuf != 0 {
		opencl.ReleaseMem(r.challengeBuf)
	}
	if r.kernel != 0 {
		opencl.ReleaseKernel(r.kernel)
	}
	if r.program != 0 {
		opencl.ReleaseProgram(r.program)
	}
	if r.queue != 0 {
		opencl.ReleaseCommandQueue(r.queue)
	}
	if r.ctx != 0 {
		opencl.ReleaseContext(r.ctx)
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

func (r *runner) search(ctx context.Context, job Job, workgroups uint32, localSize uint32, iterations uint32, onProgress func(Progress)) (Result, error) {
	started := time.Now()
	lastReport := started
	base := job.StartNonce
	var hashes uint64
	batchSize := uint64(workgroups) * uint64(localSize) * uint64(iterations)
	if err := r.writeStaticInputs(job.Challenge, job.Difficulty); err != nil {
		return Result{}, err
	}

	for {
		select {
		case <-ctx.Done():
			return Result{}, miner.ErrStopped
		default:
		}

		result, err := r.runPreparedBatch(base, workgroups, localSize, iterations)
		if err != nil {
			return Result{}, err
		}
		hashes += batchSize
		if result[0] != 0 {
			nonce := uint64(result[2])<<32 | uint64(result[1])
			return Result{
				Backend: "gpu",
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

func (r *runner) runBatch(challenge common.Hash, target *big.Int, base uint64, workgroups uint32, localSize uint32, iterations uint32) ([12]uint32, error) {
	if err := r.writeStaticInputs(challenge, target); err != nil {
		return [12]uint32{}, err
	}
	return r.runPreparedBatch(base, workgroups, localSize, iterations)
}

func (r *runner) writeStaticInputs(challenge common.Hash, target *big.Int) error {
	challengeWords := challengeLEWords(challenge)
	targetWords, err := targetBEWords(target)
	if err != nil {
		return err
	}

	if err := opencl.EnqueueWrite(r.queue, r.challengeBuf, unsafe.Pointer(&challengeWords[0]), 8*4); err != nil {
		return err
	}
	if err := opencl.EnqueueWrite(r.queue, r.targetBuf, unsafe.Pointer(&targetWords[0]), 8*4); err != nil {
		return err
	}
	return nil
}

func (r *runner) runPreparedBatch(base uint64, workgroups uint32, localSize uint32, iterations uint32) ([12]uint32, error) {
	result := [12]uint32{}

	if err := opencl.EnqueueWrite(r.queue, r.resultBuf, unsafe.Pointer(&result[0]), 12*4); err != nil {
		return result, err
	}
	if err := opencl.SetKernelScalarArg(r.kernel, 2, base); err != nil {
		return result, err
	}
	if err := opencl.SetKernelScalarArg(r.kernel, 3, iterations); err != nil {
		return result, err
	}
	global := uintptr(workgroups) * uintptr(localSize)
	local := uintptr(localSize)
	if err := opencl.EnqueueKernel(r.queue, r.kernel, global, local); err != nil {
		return result, err
	}
	if err := opencl.Finish(r.queue); err != nil {
		return result, err
	}
	if err := opencl.EnqueueRead(r.queue, r.resultBuf, unsafe.Pointer(&result[0]), 12*4); err != nil {
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

var ErrStopped = errors.New("GPU miner stopped")
