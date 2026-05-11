//go:build !windows

package gpuminer

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
	Workgroups     uint32
	LocalSize      uint32
	Iterations     uint32
	ReportInterval time.Duration
}

type Progress = miner.Progress
type Result = miner.Result

type DeviceInfo struct {
	PlatformIndex    int
	DeviceIndex      int
	PlatformName     string
	PlatformVendor   string
	Name             string
	Vendor           string
	MaxWorkGroupSize uintptr
}

func ListDevices() ([]DeviceInfo, error) {
	return nil, fmt.Errorf("GPU backend currently supports Windows OpenCL only")
}

func Search(context.Context, Job, func(Progress)) (Result, error) {
	return Result{}, fmt.Errorf("GPU backend currently supports Windows OpenCL only")
}
