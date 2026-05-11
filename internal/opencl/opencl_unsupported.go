//go:build (!windows && !linux && !darwin) || ((linux || darwin) && !cgo)

package opencl

import "fmt"

type DeviceInfo struct {
	PlatformIndex    int
	DeviceIndex      int
	PlatformName     string
	PlatformVendor   string
	Name             string
	Vendor           string
	MaxWorkGroupSize uintptr
}

func ListGPUDevices() ([]DeviceInfo, error) {
	return nil, fmt.Errorf("OpenCL GPU backend requires Windows, or Linux/macOS with CGO_ENABLED=1 and an OpenCL runtime")
}
