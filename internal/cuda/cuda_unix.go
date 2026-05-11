//go:build (linux || darwin) && cgo

package cuda

/*
#cgo linux LDFLAGS: -ldl
#include <dlfcn.h>
#include <stdint.h>
#include <stdlib.h>

static uintptr_t call0(void *fn) { return ((uintptr_t (*)())fn)(); }
static uintptr_t call1(void *fn, uintptr_t a0) { return ((uintptr_t (*)(uintptr_t))fn)(a0); }
static uintptr_t call2(void *fn, uintptr_t a0, uintptr_t a1) { return ((uintptr_t (*)(uintptr_t, uintptr_t))fn)(a0, a1); }
static uintptr_t call3(void *fn, uintptr_t a0, uintptr_t a1, uintptr_t a2) { return ((uintptr_t (*)(uintptr_t, uintptr_t, uintptr_t))fn)(a0, a1, a2); }
static uintptr_t call4(void *fn, uintptr_t a0, uintptr_t a1, uintptr_t a2, uintptr_t a3) { return ((uintptr_t (*)(uintptr_t, uintptr_t, uintptr_t, uintptr_t))fn)(a0, a1, a2, a3); }
static uintptr_t call5(void *fn, uintptr_t a0, uintptr_t a1, uintptr_t a2, uintptr_t a3, uintptr_t a4) { return ((uintptr_t (*)(uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t))fn)(a0, a1, a2, a3, a4); }
static uintptr_t call6(void *fn, uintptr_t a0, uintptr_t a1, uintptr_t a2, uintptr_t a3, uintptr_t a4, uintptr_t a5) { return ((uintptr_t (*)(uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t))fn)(a0, a1, a2, a3, a4, a5); }
static uintptr_t call7(void *fn, uintptr_t a0, uintptr_t a1, uintptr_t a2, uintptr_t a3, uintptr_t a4, uintptr_t a5, uintptr_t a6) { return ((uintptr_t (*)(uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t))fn)(a0, a1, a2, a3, a4, a5, a6); }
static uintptr_t call8(void *fn, uintptr_t a0, uintptr_t a1, uintptr_t a2, uintptr_t a3, uintptr_t a4, uintptr_t a5, uintptr_t a6, uintptr_t a7) { return ((uintptr_t (*)(uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t))fn)(a0, a1, a2, a3, a4, a5, a6, a7); }
static uintptr_t call9(void *fn, uintptr_t a0, uintptr_t a1, uintptr_t a2, uintptr_t a3, uintptr_t a4, uintptr_t a5, uintptr_t a6, uintptr_t a7, uintptr_t a8) { return ((uintptr_t (*)(uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t))fn)(a0, a1, a2, a3, a4, a5, a6, a7, a8); }
static uintptr_t call10(void *fn, uintptr_t a0, uintptr_t a1, uintptr_t a2, uintptr_t a3, uintptr_t a4, uintptr_t a5, uintptr_t a6, uintptr_t a7, uintptr_t a8, uintptr_t a9) { return ((uintptr_t (*)(uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t))fn)(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9); }
static uintptr_t call11(void *fn, uintptr_t a0, uintptr_t a1, uintptr_t a2, uintptr_t a3, uintptr_t a4, uintptr_t a5, uintptr_t a6, uintptr_t a7, uintptr_t a8, uintptr_t a9, uintptr_t a10) { return ((uintptr_t (*)(uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t))fn)(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10); }
*/
import "C"

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"syscall"
	"unsafe"
)

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

const (
	success = 0

	deviceAttributeMaxThreadsPerBlock  = 1
	deviceAttributeMultiprocessorCount = 16
)

var driver = newLazyLibrary(cudaDriverCandidates())

var (
	cuInit                    = driver.NewProc("cuInit")
	cuDeviceGetCount          = driver.NewProc("cuDeviceGetCount")
	cuDeviceGet               = driver.NewProc("cuDeviceGet")
	cuDeviceGetName           = driver.NewProc("cuDeviceGetName")
	cuDeviceComputeCapability = driver.NewProc("cuDeviceComputeCapability")
	cuDeviceGetAttribute      = driver.NewProc("cuDeviceGetAttribute")
	cuCtxCreate               = driver.NewProc("cuCtxCreate_v2")
	cuCtxDestroy              = driver.NewProc("cuCtxDestroy_v2")
	cuCtxSynchronize          = driver.NewProc("cuCtxSynchronize")
	cuModuleLoadData          = driver.NewProc("cuModuleLoadData")
	cuModuleUnload            = driver.NewProc("cuModuleUnload")
	cuModuleGetFunction       = driver.NewProc("cuModuleGetFunction")
	cuMemAlloc                = driver.NewProc("cuMemAlloc_v2")
	cuMemFree                 = driver.NewProc("cuMemFree_v2")
	cuMemcpyHtoD              = driver.NewProc("cuMemcpyHtoD_v2")
	cuMemcpyDtoH              = driver.NewProc("cuMemcpyDtoH_v2")
	cuLaunchKernel            = driver.NewProc("cuLaunchKernel")
)

func Init() error {
	return driverCall("cuInit", cuInit, 0)
}

func ListDevices() ([]DeviceInfo, error) {
	if err := Init(); err != nil {
		return nil, err
	}
	count, err := DeviceCount()
	if err != nil {
		return nil, err
	}
	if count == 0 {
		return nil, fmt.Errorf("no CUDA devices found")
	}
	out := make([]DeviceInfo, 0, count)
	for i := 0; i < count; i++ {
		device, err := DeviceGet(i)
		if err != nil {
			return nil, err
		}
		info, err := Info(i, device)
		if err != nil {
			return nil, err
		}
		out = append(out, info)
	}
	return out, nil
}

func PickDevice(index int) (Device, DeviceInfo, error) {
	if index < 0 {
		index = 0
	}
	if err := Init(); err != nil {
		return 0, DeviceInfo{}, err
	}
	count, err := DeviceCount()
	if err != nil {
		return 0, DeviceInfo{}, err
	}
	if index >= count {
		return 0, DeviceInfo{}, fmt.Errorf("CUDA device index %d not found", index)
	}
	device, err := DeviceGet(index)
	if err != nil {
		return 0, DeviceInfo{}, err
	}
	info, err := Info(index, device)
	if err != nil {
		return 0, DeviceInfo{}, err
	}
	return device, info, nil
}

func DeviceCount() (int, error) {
	var count int32
	if err := driverCall("cuDeviceGetCount", cuDeviceGetCount, uintptr(unsafe.Pointer(&count))); err != nil {
		return 0, err
	}
	return int(count), nil
}

func DeviceGet(index int) (Device, error) {
	var device Device
	if err := driverCall("cuDeviceGet", cuDeviceGet, uintptr(unsafe.Pointer(&device)), uintptr(index)); err != nil {
		return 0, err
	}
	return device, nil
}

func Info(index int, device Device) (DeviceInfo, error) {
	major, minor, err := ComputeCapability(device)
	if err != nil {
		return DeviceInfo{}, err
	}
	maxThreads, err := DeviceAttribute(device, deviceAttributeMaxThreadsPerBlock)
	if err != nil {
		return DeviceInfo{}, err
	}
	mps, err := DeviceAttribute(device, deviceAttributeMultiprocessorCount)
	if err != nil {
		return DeviceInfo{}, err
	}
	return DeviceInfo{
		DeviceIndex:        index,
		Name:               DeviceName(device),
		ComputeCapability:  fmt.Sprintf("%d.%d", major, minor),
		MaxThreadsPerBlock: maxThreads,
		Multiprocessors:    mps,
	}, nil
}

func DeviceName(device Device) string {
	buf := make([]byte, 256)
	if err := driverCall("cuDeviceGetName", cuDeviceGetName,
		uintptr(unsafe.Pointer(&buf[0])),
		uintptr(len(buf)),
		uintptr(device),
	); err != nil {
		return ""
	}
	return strings.TrimRight(string(buf), "\x00")
}

func ComputeCapability(device Device) (int, int, error) {
	var major, minor int32
	if err := driverCall("cuDeviceComputeCapability", cuDeviceComputeCapability,
		uintptr(unsafe.Pointer(&major)),
		uintptr(unsafe.Pointer(&minor)),
		uintptr(device),
	); err != nil {
		return 0, 0, err
	}
	return int(major), int(minor), nil
}

func DeviceAttribute(device Device, attr int32) (int, error) {
	var value int32
	if err := driverCall("cuDeviceGetAttribute", cuDeviceGetAttribute,
		uintptr(unsafe.Pointer(&value)),
		uintptr(attr),
		uintptr(device),
	); err != nil {
		return 0, err
	}
	return int(value), nil
}

func CreateContext(device Device) (Context, error) {
	var ctx Context
	if err := driverCall("cuCtxCreate", cuCtxCreate,
		uintptr(unsafe.Pointer(&ctx)),
		0,
		uintptr(device),
	); err != nil {
		return 0, err
	}
	return ctx, nil
}

func DestroyContext(ctx Context) {
	if ctx != 0 {
		_, _, _ = cuCtxDestroy.Call(uintptr(ctx))
	}
}

func LoadModuleData(ptx []byte) (Module, error) {
	if len(ptx) == 0 || ptx[len(ptx)-1] != 0 {
		ptx = append(ptx, 0)
	}
	var module Module
	if err := driverCall("cuModuleLoadData", cuModuleLoadData,
		uintptr(unsafe.Pointer(&module)),
		uintptr(unsafe.Pointer(&ptx[0])),
	); err != nil {
		return 0, err
	}
	return module, nil
}

func UnloadModule(module Module) {
	if module != 0 {
		_, _, _ = cuModuleUnload.Call(uintptr(module))
	}
}

func ModuleFunction(module Module, name string) (Function, error) {
	cname, err := syscall.BytePtrFromString(name)
	if err != nil {
		return 0, err
	}
	var fn Function
	if err := driverCall("cuModuleGetFunction", cuModuleGetFunction,
		uintptr(unsafe.Pointer(&fn)),
		uintptr(module),
		uintptr(unsafe.Pointer(cname)),
	); err != nil {
		return 0, err
	}
	return fn, nil
}

func MemAlloc(size uintptr) (DevicePtr, error) {
	var ptr DevicePtr
	if err := driverCall("cuMemAlloc", cuMemAlloc, uintptr(unsafe.Pointer(&ptr)), size); err != nil {
		return 0, err
	}
	return ptr, nil
}

func MemFree(ptr DevicePtr) {
	if ptr != 0 {
		_, _, _ = cuMemFree.Call(uintptr(ptr))
	}
}

func MemcpyHtoD(dst DevicePtr, src unsafe.Pointer, size uintptr) error {
	return driverCall("cuMemcpyHtoD", cuMemcpyHtoD, uintptr(dst), uintptr(src), size)
}

func MemcpyDtoH(dst unsafe.Pointer, src DevicePtr, size uintptr) error {
	return driverCall("cuMemcpyDtoH", cuMemcpyDtoH, uintptr(dst), uintptr(src), size)
}

func LaunchKernel(fn Function, gridX uint32, blockX uint32, params []unsafe.Pointer) error {
	paramPtrs := make([]uintptr, len(params))
	for i := range params {
		paramPtrs[i] = uintptr(params[i])
	}
	var paramPtr uintptr
	if len(paramPtrs) > 0 {
		paramPtr = uintptr(unsafe.Pointer(&paramPtrs[0]))
	}
	if err := driverCall("cuLaunchKernel", cuLaunchKernel,
		uintptr(fn),
		uintptr(gridX), 1, 1,
		uintptr(blockX), 1, 1,
		0,
		0,
		paramPtr,
		0,
	); err != nil {
		return err
	}
	if err := driverCall("cuCtxSynchronize", cuCtxSynchronize); err != nil {
		return err
	}
	return nil
}

func driverCall(op string, proc *lazyProc, args ...uintptr) error {
	r1, r2, lastErr := proc.Call(args...)
	return check(op, r1, r2, lastErr)
}

func check(op string, ret uintptr, _ uintptr, _ error) error {
	code := int32(ret)
	if code == success {
		return nil
	}
	return fmt.Errorf("%s failed with CUDA status %d", op, code)
}

type lazyLibrary struct {
	candidates []string
	once       sync.Once
	lib        *loadedLibrary
	err        error
}

type loadedLibrary struct {
	name   string
	handle unsafe.Pointer
}

type lazyProc struct {
	lib  *lazyLibrary
	name string
	once sync.Once
	addr unsafe.Pointer
	err  error
}

func newLazyLibrary(candidates []string) *lazyLibrary {
	return &lazyLibrary{candidates: candidates}
}

func (l *lazyLibrary) NewProc(name string) *lazyProc {
	return &lazyProc{lib: l, name: name}
}

func (l *lazyLibrary) load() error {
	l.once.Do(func() {
		var lastErr error
		for _, candidate := range l.candidates {
			lib, err := openLoadedLibrary(candidate)
			if err == nil {
				l.lib = lib
				return
			}
			lastErr = err
		}
		if lastErr != nil {
			l.err = fmt.Errorf("load CUDA driver: %w", lastErr)
			return
		}
		l.err = fmt.Errorf("load CUDA driver: library not found; set HASHMINER_CUDA_DRIVER")
	})
	return l.err
}

func (p *lazyProc) load() error {
	p.once.Do(func() {
		if err := p.lib.load(); err != nil {
			p.err = err
			return
		}
		resolved, err := p.lib.lib.proc(p.name)
		if err != nil {
			p.err = err
			return
		}
		p.addr = resolved.addr
	})
	return p.err
}

func (l *loadedLibrary) proc(name string) (*lazyProc, error) {
	cname := C.CString(name)
	addr := C.dlsym(l.handle, cname)
	C.free(unsafe.Pointer(cname))
	if addr == nil {
		return nil, fmt.Errorf("load symbol %s from %s: %w", name, l.name, dlError())
	}
	return &lazyProc{name: name, addr: addr}, nil
}

func openLoadedLibrary(candidate string) (*loadedLibrary, error) {
	candidate = strings.TrimSpace(candidate)
	if candidate == "" {
		return nil, fmt.Errorf("empty library name")
	}
	cname := C.CString(candidate)
	handle := C.dlopen(cname, C.RTLD_LAZY|C.RTLD_LOCAL)
	C.free(unsafe.Pointer(cname))
	if handle == nil {
		return nil, dlError()
	}
	return &loadedLibrary{name: candidate, handle: handle}, nil
}

func (l *loadedLibrary) close() {
	if l != nil && l.handle != nil {
		C.dlclose(l.handle)
		l.handle = nil
	}
}

func (p *lazyProc) Call(args ...uintptr) (uintptr, uintptr, error) {
	if p.addr == nil {
		if err := p.load(); err != nil {
			return 0, 0, err
		}
	}
	a := [11]uintptr{}
	copy(a[:], args)
	var r uintptr
	switch len(args) {
	case 0:
		r = uintptr(C.call0(p.addr))
	case 1:
		r = uintptr(C.call1(p.addr, C.uintptr_t(a[0])))
	case 2:
		r = uintptr(C.call2(p.addr, C.uintptr_t(a[0]), C.uintptr_t(a[1])))
	case 3:
		r = uintptr(C.call3(p.addr, C.uintptr_t(a[0]), C.uintptr_t(a[1]), C.uintptr_t(a[2])))
	case 4:
		r = uintptr(C.call4(p.addr, C.uintptr_t(a[0]), C.uintptr_t(a[1]), C.uintptr_t(a[2]), C.uintptr_t(a[3])))
	case 5:
		r = uintptr(C.call5(p.addr, C.uintptr_t(a[0]), C.uintptr_t(a[1]), C.uintptr_t(a[2]), C.uintptr_t(a[3]), C.uintptr_t(a[4])))
	case 6:
		r = uintptr(C.call6(p.addr, C.uintptr_t(a[0]), C.uintptr_t(a[1]), C.uintptr_t(a[2]), C.uintptr_t(a[3]), C.uintptr_t(a[4]), C.uintptr_t(a[5])))
	case 7:
		r = uintptr(C.call7(p.addr, C.uintptr_t(a[0]), C.uintptr_t(a[1]), C.uintptr_t(a[2]), C.uintptr_t(a[3]), C.uintptr_t(a[4]), C.uintptr_t(a[5]), C.uintptr_t(a[6])))
	case 8:
		r = uintptr(C.call8(p.addr, C.uintptr_t(a[0]), C.uintptr_t(a[1]), C.uintptr_t(a[2]), C.uintptr_t(a[3]), C.uintptr_t(a[4]), C.uintptr_t(a[5]), C.uintptr_t(a[6]), C.uintptr_t(a[7])))
	case 9:
		r = uintptr(C.call9(p.addr, C.uintptr_t(a[0]), C.uintptr_t(a[1]), C.uintptr_t(a[2]), C.uintptr_t(a[3]), C.uintptr_t(a[4]), C.uintptr_t(a[5]), C.uintptr_t(a[6]), C.uintptr_t(a[7]), C.uintptr_t(a[8])))
	case 10:
		r = uintptr(C.call10(p.addr, C.uintptr_t(a[0]), C.uintptr_t(a[1]), C.uintptr_t(a[2]), C.uintptr_t(a[3]), C.uintptr_t(a[4]), C.uintptr_t(a[5]), C.uintptr_t(a[6]), C.uintptr_t(a[7]), C.uintptr_t(a[8]), C.uintptr_t(a[9])))
	case 11:
		r = uintptr(C.call11(p.addr, C.uintptr_t(a[0]), C.uintptr_t(a[1]), C.uintptr_t(a[2]), C.uintptr_t(a[3]), C.uintptr_t(a[4]), C.uintptr_t(a[5]), C.uintptr_t(a[6]), C.uintptr_t(a[7]), C.uintptr_t(a[8]), C.uintptr_t(a[9]), C.uintptr_t(a[10])))
	default:
		return 0, 0, fmt.Errorf("CUDA call %s has unsupported arity %d", p.name, len(args))
	}
	return r, 0, nil
}

func dlError() error {
	raw := C.dlerror()
	if raw == nil {
		return fmt.Errorf("unknown dlopen error")
	}
	return fmt.Errorf("%s", C.GoString(raw))
}

type nvrtcLibrary struct {
	lib               *loadedLibrary
	createProgram     *lazyProc
	compileProgram    *lazyProc
	getProgramLogSize *lazyProc
	getProgramLog     *lazyProc
	getPTXSize        *lazyProc
	getPTX            *lazyProc
	getCUBINSize      *lazyProc
	getCUBIN          *lazyProc
	destroyProgram    *lazyProc
	getErrorString    *lazyProc
}

func CompilePTX(source string, name string, options []string) ([]byte, error) {
	nvrtc, err := loadNVRTC()
	if err != nil {
		return nil, err
	}
	defer nvrtc.lib.close()

	src := append([]byte(source), 0)
	cname := append([]byte(name), 0)

	var program Program
	if err := nvrtc.call("nvrtcCreateProgram", nvrtc.createProgram,
		uintptr(unsafe.Pointer(&program)),
		uintptr(unsafe.Pointer(&src[0])),
		uintptr(unsafe.Pointer(&cname[0])),
		0,
		0,
		0,
	); err != nil {
		return nil, err
	}
	defer func() {
		_, _, _ = nvrtc.destroyProgram.Call(uintptr(unsafe.Pointer(&program)))
	}()

	optionBytes := make([][]byte, 0, len(options))
	optionPtrs := make([]uintptr, 0, len(options))
	for _, option := range options {
		b := append([]byte(option), 0)
		optionBytes = append(optionBytes, b)
		optionPtrs = append(optionPtrs, uintptr(unsafe.Pointer(&optionBytes[len(optionBytes)-1][0])))
	}

	var optionPtr uintptr
	if len(optionPtrs) > 0 {
		optionPtr = uintptr(unsafe.Pointer(&optionPtrs[0]))
	}
	if err := nvrtc.call("nvrtcCompileProgram", nvrtc.compileProgram,
		uintptr(program),
		uintptr(len(optionPtrs)),
		optionPtr,
	); err != nil {
		log := nvrtc.programLog(program)
		if strings.TrimSpace(log) != "" {
			return nil, fmt.Errorf("%w\n%s", err, log)
		}
		return nil, err
	}

	if nvrtc.getCUBINSize != nil && nvrtc.getCUBIN != nil && optionsContainSM(options) {
		var size uintptr
		if err := nvrtc.call("nvrtcGetCUBINSize", nvrtc.getCUBINSize, uintptr(program), uintptr(unsafe.Pointer(&size))); err != nil {
			return nil, err
		}
		if size > 0 {
			cubin := make([]byte, size)
			if err := nvrtc.call("nvrtcGetCUBIN", nvrtc.getCUBIN, uintptr(program), uintptr(unsafe.Pointer(&cubin[0]))); err != nil {
				return nil, err
			}
			return cubin, nil
		}
	}

	var size uintptr
	if err := nvrtc.call("nvrtcGetPTXSize", nvrtc.getPTXSize, uintptr(program), uintptr(unsafe.Pointer(&size))); err != nil {
		return nil, err
	}
	if size == 0 {
		return nil, fmt.Errorf("nvrtc produced empty PTX")
	}
	ptx := make([]byte, size)
	if err := nvrtc.call("nvrtcGetPTX", nvrtc.getPTX, uintptr(program), uintptr(unsafe.Pointer(&ptx[0]))); err != nil {
		return nil, err
	}
	return ptx, nil
}

func loadNVRTC() (*nvrtcLibrary, error) {
	var lastErr error
	for _, candidate := range nvrtcCandidates() {
		lib, err := openLoadedLibrary(candidate)
		if err != nil {
			lastErr = err
			continue
		}
		nvrtc := &nvrtcLibrary{lib: lib}
		if err := nvrtc.loadProcs(); err != nil {
			lib.close()
			lastErr = err
			continue
		}
		return nvrtc, nil
	}
	if lastErr != nil {
		return nil, fmt.Errorf("load NVRTC: %w", lastErr)
	}
	return nil, fmt.Errorf("load NVRTC: library not found; install CUDA Toolkit or set HASHMINER_NVRTC_LIB")
}

func (n *nvrtcLibrary) loadProcs() error {
	var err error
	if n.createProgram, err = n.lib.proc("nvrtcCreateProgram"); err != nil {
		return err
	}
	if n.compileProgram, err = n.lib.proc("nvrtcCompileProgram"); err != nil {
		return err
	}
	if n.getProgramLogSize, err = n.lib.proc("nvrtcGetProgramLogSize"); err != nil {
		return err
	}
	if n.getProgramLog, err = n.lib.proc("nvrtcGetProgramLog"); err != nil {
		return err
	}
	if n.getPTXSize, err = n.lib.proc("nvrtcGetPTXSize"); err != nil {
		return err
	}
	if n.getPTX, err = n.lib.proc("nvrtcGetPTX"); err != nil {
		return err
	}
	n.getCUBINSize, _ = n.lib.proc("nvrtcGetCUBINSize")
	n.getCUBIN, _ = n.lib.proc("nvrtcGetCUBIN")
	if n.destroyProgram, err = n.lib.proc("nvrtcDestroyProgram"); err != nil {
		return err
	}
	n.getErrorString, _ = n.lib.proc("nvrtcGetErrorString")
	return nil
}

func (n *nvrtcLibrary) call(op string, proc *lazyProc, args ...uintptr) error {
	r1, r2, lastErr := proc.Call(args...)
	return n.check(op, r1, r2, lastErr)
}

func (n *nvrtcLibrary) check(op string, ret uintptr, _ uintptr, _ error) error {
	code := int32(ret)
	if code == success {
		return nil
	}
	if n.getErrorString != nil {
		ptr, _, _ := n.getErrorString.Call(uintptr(code))
		if ptr != 0 {
			return fmt.Errorf("%s failed with NVRTC status %d (%s)", op, code, cString(ptr))
		}
	}
	return fmt.Errorf("%s failed with NVRTC status %d", op, code)
}

func (n *nvrtcLibrary) programLog(program Program) string {
	var size uintptr
	if err := n.call("nvrtcGetProgramLogSize", n.getProgramLogSize, uintptr(program), uintptr(unsafe.Pointer(&size))); err != nil {
		return ""
	}
	if size == 0 {
		return ""
	}
	buf := make([]byte, size)
	if err := n.call("nvrtcGetProgramLog", n.getProgramLog, uintptr(program), uintptr(unsafe.Pointer(&buf[0]))); err != nil {
		return ""
	}
	return strings.TrimRight(string(buf), "\x00")
}

func cudaDriverCandidates() []string {
	var out []string
	add := func(value string) {
		value = strings.TrimSpace(value)
		if value != "" {
			out = append(out, value)
		}
	}
	add(os.Getenv("HASHMINER_CUDA_DRIVER"))
	if runtime.GOOS == "darwin" {
		for _, name := range []string{
			"libcuda.dylib",
			"/usr/local/cuda/lib/libcuda.dylib",
			"/usr/local/cuda/lib/libcuda.1.dylib",
		} {
			add(name)
		}
		return out
	}
	for _, name := range []string{
		"libcuda.so.1",
		"libcuda.so",
		"/usr/lib/x86_64-linux-gnu/libcuda.so.1",
		"/usr/lib/aarch64-linux-gnu/libcuda.so.1",
		"/usr/lib64/libcuda.so.1",
	} {
		add(name)
	}
	return out
}

func nvrtcCandidates() []string {
	seen := map[string]bool{}
	var out []string
	add := func(value string) {
		value = strings.TrimSpace(value)
		if value == "" || seen[strings.ToLower(value)] {
			return
		}
		seen[strings.ToLower(value)] = true
		out = append(out, value)
	}

	add(os.Getenv("HASHMINER_NVRTC_LIB"))
	add(os.Getenv("HASHMINER_NVRTC_DLL"))

	var dirs []string
	if cudaPath := os.Getenv("CUDA_PATH"); cudaPath != "" {
		dirs = append(dirs, filepath.Join(cudaPath, "lib64"))
		dirs = append(dirs, filepath.Join(cudaPath, "lib"))
	}
	dirs = append(dirs,
		"/usr/local/cuda/lib64",
		"/usr/local/cuda/lib",
		"/usr/lib/x86_64-linux-gnu",
		"/usr/lib/aarch64-linux-gnu",
		"/usr/lib64",
		"/usr/lib",
	)
	dirs = append(dirs, strings.Split(os.Getenv("LD_LIBRARY_PATH"), string(os.PathListSeparator))...)
	dirs = append(dirs, strings.Split(os.Getenv("DYLD_LIBRARY_PATH"), string(os.PathListSeparator))...)

	var globbed []string
	for _, dir := range dirs {
		if strings.TrimSpace(dir) == "" {
			continue
		}
		matches, _ := filepath.Glob(filepath.Join(dir, "libnvrtc.so*"))
		globbed = append(globbed, matches...)
		matches, _ = filepath.Glob(filepath.Join(dir, "libnvrtc*.dylib"))
		globbed = append(globbed, matches...)
	}
	sort.Sort(sort.Reverse(sort.StringSlice(globbed)))
	for _, match := range globbed {
		add(match)
	}

	names := []string{"libnvrtc.so", "libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so.11"}
	if runtime.GOOS == "darwin" {
		names = []string{"libnvrtc.dylib", "/usr/local/cuda/lib/libnvrtc.dylib"}
	}
	for _, name := range names {
		add(name)
	}

	return out
}

func optionsContainSM(options []string) bool {
	for _, option := range options {
		if strings.Contains(option, "sm_") {
			return true
		}
	}
	return false
}

func cString(ptr uintptr) string {
	if ptr == 0 {
		return ""
	}
	var bytes []byte
	for i := uintptr(0); ; i++ {
		b := *(*byte)(unsafe.Pointer(ptr + i))
		if b == 0 {
			return string(bytes)
		}
		bytes = append(bytes, b)
	}
}
