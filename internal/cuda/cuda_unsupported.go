//go:build (!windows && !linux && !darwin) || ((linux || darwin) && !cgo)

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
	return nil, fmt.Errorf("CUDA backend requires Windows, or Linux/macOS with CGO_ENABLED=1 and CUDA driver/toolkit libraries")
}
