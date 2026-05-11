package cuda

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
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

var driver = syscall.NewLazyDLL("nvcuda.dll")

var (
	kernel32        = syscall.NewLazyDLL("kernel32.dll")
	setDllDirectory = kernel32.NewProc("SetDllDirectoryW")

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

func driverCall(op string, proc *syscall.LazyProc, args ...uintptr) error {
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

type nvrtcLibrary struct {
	dll               *syscall.DLL
	createProgram     *syscall.Proc
	compileProgram    *syscall.Proc
	getProgramLogSize *syscall.Proc
	getProgramLog     *syscall.Proc
	getPTXSize        *syscall.Proc
	getPTX            *syscall.Proc
	getCUBINSize      *syscall.Proc
	getCUBIN          *syscall.Proc
	destroyProgram    *syscall.Proc
	getErrorString    *syscall.Proc
}

func CompilePTX(source string, name string, options []string) ([]byte, error) {
	nvrtc, err := loadNVRTC()
	if err != nil {
		return nil, err
	}
	defer nvrtc.dll.Release()

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
		if filepath.IsAbs(candidate) {
			_ = setDLLDirectory(filepath.Dir(candidate))
		}
		dll, err := syscall.LoadDLL(candidate)
		if err != nil {
			lastErr = err
			continue
		}
		lib := &nvrtcLibrary{dll: dll}
		if err := lib.loadProcs(); err != nil {
			_ = dll.Release()
			lastErr = err
			continue
		}
		return lib, nil
	}
	if lastErr != nil {
		return nil, fmt.Errorf("load NVRTC: %w", lastErr)
	}
	return nil, fmt.Errorf("load NVRTC: nvrtc64_*.dll not found; install CUDA Toolkit or set HASHMINER_NVRTC_DLL")
}

func setDLLDirectory(dir string) error {
	ptr, err := syscall.UTF16PtrFromString(dir)
	if err != nil {
		return err
	}
	r1, _, lastErr := setDllDirectory.Call(uintptr(unsafe.Pointer(ptr)))
	if r1 == 0 {
		return lastErr
	}
	return nil
}

func (n *nvrtcLibrary) loadProcs() error {
	var err error
	if n.createProgram, err = n.dll.FindProc("nvrtcCreateProgram"); err != nil {
		return err
	}
	if n.compileProgram, err = n.dll.FindProc("nvrtcCompileProgram"); err != nil {
		return err
	}
	if n.getProgramLogSize, err = n.dll.FindProc("nvrtcGetProgramLogSize"); err != nil {
		return err
	}
	if n.getProgramLog, err = n.dll.FindProc("nvrtcGetProgramLog"); err != nil {
		return err
	}
	if n.getPTXSize, err = n.dll.FindProc("nvrtcGetPTXSize"); err != nil {
		return err
	}
	if n.getPTX, err = n.dll.FindProc("nvrtcGetPTX"); err != nil {
		return err
	}
	n.getCUBINSize, _ = n.dll.FindProc("nvrtcGetCUBINSize")
	n.getCUBIN, _ = n.dll.FindProc("nvrtcGetCUBIN")
	if n.destroyProgram, err = n.dll.FindProc("nvrtcDestroyProgram"); err != nil {
		return err
	}
	n.getErrorString, _ = n.dll.FindProc("nvrtcGetErrorString")
	return nil
}

func (n *nvrtcLibrary) call(op string, proc *syscall.Proc, args ...uintptr) error {
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

	add(os.Getenv("HASHMINER_NVRTC_DLL"))

	var dirs []string
	if cudaPath := os.Getenv("CUDA_PATH"); cudaPath != "" {
		dirs = append(dirs, filepath.Join(cudaPath, "bin"))
		dirs = append(dirs, filepath.Join(cudaPath, "bin", "x64"))
	}
	for _, env := range os.Environ() {
		key, value, ok := strings.Cut(env, "=")
		if !ok || !strings.HasPrefix(strings.ToUpper(key), "CUDA_PATH_V") {
			continue
		}
		dirs = append(dirs, filepath.Join(value, "bin"))
		dirs = append(dirs, filepath.Join(value, "bin", "x64"))
	}
	toolkitRoots, _ := filepath.Glob(`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*`)
	for _, root := range toolkitRoots {
		dirs = append(dirs, filepath.Join(root, "bin"))
		dirs = append(dirs, filepath.Join(root, "bin", "x64"))
	}
	dirs = append(dirs, strings.Split(os.Getenv("PATH"), string(os.PathListSeparator))...)

	var globbed []string
	for _, dir := range dirs {
		if strings.TrimSpace(dir) == "" {
			continue
		}
		matches, _ := filepath.Glob(filepath.Join(dir, "nvrtc64_*.dll"))
		globbed = append(globbed, matches...)
	}
	sort.Sort(sort.Reverse(sort.StringSlice(globbed)))
	for _, match := range globbed {
		add(match)
	}

	for _, name := range []string{
		"nvrtc64_130_0.dll",
		"nvrtc64_129_0.dll",
		"nvrtc64_128_0.dll",
		"nvrtc64_127_0.dll",
		"nvrtc64_126_0.dll",
		"nvrtc64_125_0.dll",
		"nvrtc64_124_0.dll",
		"nvrtc64_123_0.dll",
		"nvrtc64_122_0.dll",
		"nvrtc64_121_0.dll",
		"nvrtc64_120_0.dll",
		"nvrtc64_112_0.dll",
		"nvrtc64_111_0.dll",
		"nvrtc64_110_0.dll",
		"nvrtc.dll",
	} {
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
