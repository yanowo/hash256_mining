//go:build !windows

package cuda

import "fmt"

type Device int32
type Context uintptr
type Module uintptr
type Function uintptr
type DevicePtr uintptr
type Program uintptr

type DeviceInfo struct {
	DeviceIndex        int
	Name               string
	ComputeCapability  string
	MaxThreadsPerBlock int
	Multiprocessors    int
}

func ListDevices() ([]DeviceInfo, error) {
	return nil, fmt.Errorf("CUDA backend currently supports Windows only")
}
