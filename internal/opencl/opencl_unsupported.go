//go:build !windows

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
	return nil, fmt.Errorf("OpenCL GPU backend currently supports Windows only")
}
