package opencl

import (
	"fmt"
	"strings"
	"syscall"
	"unsafe"
)

type Platform uintptr
type Device uintptr
type Context uintptr
type CommandQueue uintptr
type Mem uintptr
type Program uintptr
type Kernel uintptr

const (
	Success = 0

	DeviceTypeGPU = 1 << 2

	DeviceType             = 0x1000
	DeviceMaxWorkGroupSize = 0x1004
	DeviceName             = 0x102B
	DeviceVendor           = 0x102C

	PlatformName   = 0x0902
	PlatformVendor = 0x0903

	MemReadWrite = 1 << 0
	MemWriteOnly = 1 << 1
	MemReadOnly  = 1 << 2

	ProgramBuildLog = 0x1183
)

var cl = syscall.NewLazyDLL("OpenCL.dll")

var (
	clGetPlatformIDs       = cl.NewProc("clGetPlatformIDs")
	clGetPlatformInfo      = cl.NewProc("clGetPlatformInfo")
	clGetDeviceIDs         = cl.NewProc("clGetDeviceIDs")
	clGetDeviceInfo        = cl.NewProc("clGetDeviceInfo")
	clCreateContext        = cl.NewProc("clCreateContext")
	clCreateCommandQueue   = cl.NewProc("clCreateCommandQueue")
	clCreateProgramSource  = cl.NewProc("clCreateProgramWithSource")
	clBuildProgram         = cl.NewProc("clBuildProgram")
	clGetProgramBuildInfo  = cl.NewProc("clGetProgramBuildInfo")
	clCreateKernel         = cl.NewProc("clCreateKernel")
	clCreateBuffer         = cl.NewProc("clCreateBuffer")
	clSetKernelArg         = cl.NewProc("clSetKernelArg")
	clEnqueueWriteBuffer   = cl.NewProc("clEnqueueWriteBuffer")
	clEnqueueReadBuffer    = cl.NewProc("clEnqueueReadBuffer")
	clEnqueueNDRangeKernel = cl.NewProc("clEnqueueNDRangeKernel")
	clFinish               = cl.NewProc("clFinish")
	clReleaseMemObject     = cl.NewProc("clReleaseMemObject")
	clReleaseKernel        = cl.NewProc("clReleaseKernel")
	clReleaseProgram       = cl.NewProc("clReleaseProgram")
	clReleaseCommandQueue  = cl.NewProc("clReleaseCommandQueue")
	clReleaseContext       = cl.NewProc("clReleaseContext")
)

type DeviceInfo struct {
	PlatformIndex    int
	DeviceIndex      int
	PlatformName     string
	PlatformVendor   string
	Name             string
	Vendor           string
	MaxWorkGroupSize uintptr
}

func Platforms() ([]Platform, error) {
	var count uint32
	if err := check(clGetPlatformIDs.Call(0, 0, uintptr(unsafe.Pointer(&count)))); err != nil {
		return nil, err
	}
	if count == 0 {
		return nil, fmt.Errorf("no OpenCL platforms found")
	}
	platforms := make([]Platform, count)
	if err := check(clGetPlatformIDs.Call(uintptr(count), uintptr(unsafe.Pointer(&platforms[0])), 0)); err != nil {
		return nil, err
	}
	return platforms, nil
}

func Devices(platform Platform, deviceType uint64) ([]Device, error) {
	var count uint32
	if err := check(clGetDeviceIDs.Call(uintptr(platform), uintptr(deviceType), 0, 0, uintptr(unsafe.Pointer(&count)))); err != nil {
		return nil, err
	}
	if count == 0 {
		return nil, nil
	}
	devices := make([]Device, count)
	if err := check(clGetDeviceIDs.Call(uintptr(platform), uintptr(deviceType), uintptr(count), uintptr(unsafe.Pointer(&devices[0])), 0)); err != nil {
		return nil, err
	}
	return devices, nil
}

func ListGPUDevices() ([]DeviceInfo, error) {
	platforms, err := Platforms()
	if err != nil {
		return nil, err
	}
	var out []DeviceInfo
	for pi, platform := range platforms {
		devices, err := Devices(platform, DeviceTypeGPU)
		if err != nil {
			continue
		}
		for di, device := range devices {
			info := DeviceInfo{
				PlatformIndex:    pi,
				DeviceIndex:      di,
				PlatformName:     platformString(platform, PlatformName),
				PlatformVendor:   platformString(platform, PlatformVendor),
				Name:             deviceString(device, DeviceName),
				Vendor:           deviceString(device, DeviceVendor),
				MaxWorkGroupSize: deviceSize(device, DeviceMaxWorkGroupSize),
			}
			out = append(out, info)
		}
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no OpenCL GPU devices found")
	}
	return out, nil
}

func PickGPU(globalIndex int) (Platform, Device, DeviceInfo, error) {
	platforms, err := Platforms()
	if err != nil {
		return 0, 0, DeviceInfo{}, err
	}
	seen := 0
	for pi, platform := range platforms {
		devices, err := Devices(platform, DeviceTypeGPU)
		if err != nil {
			continue
		}
		for di, device := range devices {
			if seen == globalIndex {
				return platform, device, DeviceInfo{
					PlatformIndex:    pi,
					DeviceIndex:      di,
					PlatformName:     platformString(platform, PlatformName),
					PlatformVendor:   platformString(platform, PlatformVendor),
					Name:             deviceString(device, DeviceName),
					Vendor:           deviceString(device, DeviceVendor),
					MaxWorkGroupSize: deviceSize(device, DeviceMaxWorkGroupSize),
				}, nil
			}
			seen++
		}
	}
	return 0, 0, DeviceInfo{}, fmt.Errorf("OpenCL GPU device index %d not found", globalIndex)
}

func CreateContext(device Device) (Context, error) {
	var code int32
	ctx, _, _ := clCreateContext.Call(
		0,
		1,
		uintptr(unsafe.Pointer(&device)),
		0,
		0,
		uintptr(unsafe.Pointer(&code)),
	)
	if code != Success {
		return 0, statusError("clCreateContext", code)
	}
	return Context(ctx), nil
}

func CreateCommandQueue(ctx Context, device Device) (CommandQueue, error) {
	var code int32
	queue, _, _ := clCreateCommandQueue.Call(
		uintptr(ctx),
		uintptr(device),
		0,
		uintptr(unsafe.Pointer(&code)),
	)
	if code != Success {
		return 0, statusError("clCreateCommandQueue", code)
	}
	return CommandQueue(queue), nil
}

func CreateProgramWithSource(ctx Context, source string) (Program, error) {
	src, err := syscall.BytePtrFromString(source)
	if err != nil {
		return 0, err
	}
	srcPtr := uintptr(unsafe.Pointer(src))
	length := uintptr(len(source))
	var code int32
	program, _, _ := clCreateProgramSource.Call(
		uintptr(ctx),
		1,
		uintptr(unsafe.Pointer(&srcPtr)),
		uintptr(unsafe.Pointer(&length)),
		uintptr(unsafe.Pointer(&code)),
	)
	if code != Success {
		return 0, statusError("clCreateProgramWithSource", code)
	}
	return Program(program), nil
}

func BuildProgram(program Program, device Device, options string) error {
	opt, err := syscall.BytePtrFromString(options)
	if err != nil {
		return err
	}
	if err := check(clBuildProgram.Call(uintptr(program), 1, uintptr(unsafe.Pointer(&device)), uintptr(unsafe.Pointer(opt)), 0, 0)); err != nil {
		log := BuildLog(program, device)
		if strings.TrimSpace(log) != "" {
			return fmt.Errorf("%w\n%s", err, log)
		}
		return err
	}
	return nil
}

func BuildLog(program Program, device Device) string {
	var size uintptr
	_, _, _ = clGetProgramBuildInfo.Call(uintptr(program), uintptr(device), ProgramBuildLog, 0, 0, uintptr(unsafe.Pointer(&size)))
	if size == 0 {
		return ""
	}
	buf := make([]byte, size)
	_, _, _ = clGetProgramBuildInfo.Call(uintptr(program), uintptr(device), ProgramBuildLog, size, uintptr(unsafe.Pointer(&buf[0])), 0)
	return strings.TrimRight(string(buf), "\x00")
}

func CreateKernel(program Program, name string) (Kernel, error) {
	cname, err := syscall.BytePtrFromString(name)
	if err != nil {
		return 0, err
	}
	var code int32
	kernel, _, _ := clCreateKernel.Call(uintptr(program), uintptr(unsafe.Pointer(cname)), uintptr(unsafe.Pointer(&code)))
	if code != Success {
		return 0, statusError("clCreateKernel", code)
	}
	return Kernel(kernel), nil
}

func CreateBuffer(ctx Context, flags uint64, size uintptr) (Mem, error) {
	var code int32
	mem, _, _ := clCreateBuffer.Call(uintptr(ctx), uintptr(flags), size, 0, uintptr(unsafe.Pointer(&code)))
	if code != Success {
		return 0, statusError("clCreateBuffer", code)
	}
	return Mem(mem), nil
}

func SetKernelMemArg(kernel Kernel, index uint32, mem Mem) error {
	return check(clSetKernelArg.Call(uintptr(kernel), uintptr(index), unsafe.Sizeof(mem), uintptr(unsafe.Pointer(&mem))))
}

func SetKernelScalarArg[T ~uint32 | ~uint64](kernel Kernel, index uint32, value T) error {
	return check(clSetKernelArg.Call(uintptr(kernel), uintptr(index), unsafe.Sizeof(value), uintptr(unsafe.Pointer(&value))))
}

func EnqueueWrite(queue CommandQueue, mem Mem, data unsafe.Pointer, size uintptr) error {
	return check(clEnqueueWriteBuffer.Call(uintptr(queue), uintptr(mem), 1, 0, size, uintptr(data), 0, 0, 0))
}

func EnqueueRead(queue CommandQueue, mem Mem, data unsafe.Pointer, size uintptr) error {
	return check(clEnqueueReadBuffer.Call(uintptr(queue), uintptr(mem), 1, 0, size, uintptr(data), 0, 0, 0))
}

func EnqueueKernel(queue CommandQueue, kernel Kernel, globalWorkSize uintptr, localWorkSize uintptr) error {
	return check(clEnqueueNDRangeKernel.Call(
		uintptr(queue),
		uintptr(kernel),
		1,
		0,
		uintptr(unsafe.Pointer(&globalWorkSize)),
		uintptr(unsafe.Pointer(&localWorkSize)),
		0,
		0,
		0,
	))
}

func Finish(queue CommandQueue) error {
	return check(clFinish.Call(uintptr(queue)))
}

func ReleaseMem(mem Mem)                 { _, _, _ = clReleaseMemObject.Call(uintptr(mem)) }
func ReleaseKernel(kernel Kernel)        { _, _, _ = clReleaseKernel.Call(uintptr(kernel)) }
func ReleaseProgram(program Program)     { _, _, _ = clReleaseProgram.Call(uintptr(program)) }
func ReleaseCommandQueue(q CommandQueue) { _, _, _ = clReleaseCommandQueue.Call(uintptr(q)) }
func ReleaseContext(ctx Context)         { _, _, _ = clReleaseContext.Call(uintptr(ctx)) }

func platformString(platform Platform, param uint32) string {
	var size uintptr
	_, _, _ = clGetPlatformInfo.Call(uintptr(platform), uintptr(param), 0, 0, uintptr(unsafe.Pointer(&size)))
	if size == 0 {
		return ""
	}
	buf := make([]byte, size)
	_, _, _ = clGetPlatformInfo.Call(uintptr(platform), uintptr(param), size, uintptr(unsafe.Pointer(&buf[0])), 0)
	return strings.TrimRight(string(buf), "\x00")
}

func deviceString(device Device, param uint32) string {
	var size uintptr
	_, _, _ = clGetDeviceInfo.Call(uintptr(device), uintptr(param), 0, 0, uintptr(unsafe.Pointer(&size)))
	if size == 0 {
		return ""
	}
	buf := make([]byte, size)
	_, _, _ = clGetDeviceInfo.Call(uintptr(device), uintptr(param), size, uintptr(unsafe.Pointer(&buf[0])), 0)
	return strings.TrimRight(string(buf), "\x00")
}

func deviceSize(device Device, param uint32) uintptr {
	var out uintptr
	_, _, _ = clGetDeviceInfo.Call(uintptr(device), uintptr(param), unsafe.Sizeof(out), uintptr(unsafe.Pointer(&out)), 0)
	return out
}

func check(ret uintptr, _ uintptr, _ error) error {
	code := int32(ret)
	if code == Success {
		return nil
	}
	return statusError("OpenCL", code)
}

func statusError(op string, code int32) error {
	return fmt.Errorf("%s failed with OpenCL status %d", op, code)
}
