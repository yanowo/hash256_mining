//go:build (!windows && !linux && !darwin) || ((linux || darwin) && !cgo)

package cudaminer

import (
	"context"
	"fmt"
	"math/big"
	"time"

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

type DeviceInfo struct {
	DeviceIndex        int
	Name               string
	ComputeCapability  string
	MaxThreadsPerBlock int
	Multiprocessors    int
}

func ListDevices() ([]DeviceInfo, error) {
	return nil, fmt.Errorf("CUDA backend requires Windows, or Linux/macOS with CGO_ENABLED=1 and CUDA driver/toolkit libraries")
}

func Search(context.Context, Job, func(Progress)) (Result, error) {
	return Result{}, fmt.Errorf("CUDA backend requires Windows, or Linux/macOS with CGO_ENABLED=1 and CUDA driver/toolkit libraries")
}
