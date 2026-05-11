package main

import (
	"bytes"
	"context"
	"crypto/ecdsa"
	"crypto/rand"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"math/big"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"hash256mining/internal/cudaminer"
	"hash256mining/internal/gpuminer"
	"hash256mining/internal/hash256"
	"hash256mining/internal/miner"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/crypto"
)

func main() {
	if err := run(os.Args); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	if err := loadDotEnv(".env"); err != nil {
		return err
	}

	if len(args) < 2 {
		usage()
		return nil
	}

	switch args[1] {
	case "status":
		return runStatus(args[2:])
	case "mine":
		return runMine(args[2:])
	case "bench":
		return runBench(args[2:])
	case "gpu-info":
		return runGPUInfo(args[2:])
	case "cuda-info":
		return runCUDAInfo(args[2:])
	case "pool-mine":
		return runPoolMine(args[2:])
	case "-h", "--help", "help":
		usage()
		return nil
	default:
		return fmt.Errorf("unknown command %q", args[1])
	}
}

func runStatus(args []string) error {
	fs := flag.NewFlagSet("status", flag.ExitOnError)
	rpcURL := fs.String("rpc", "", "Ethereum RPC URL; falls back to ETH_RPC_URL")
	contract := fs.String("contract", hash256.DefaultContract, "HASH contract address")
	address := fs.String("address", "", "optional miner address for challenge and balance")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *rpcURL == "" {
		*rpcURL = env("ETH_RPC_URL", "")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	client, err := hash256.Dial(ctx, *rpcURL, *contract)
	if err != nil {
		return err
	}
	defer client.Close()

	chainID, err := client.ChainID(ctx)
	if err != nil {
		return err
	}
	genesis, err := client.GenesisState(ctx)
	if err != nil {
		return err
	}
	state, err := client.MiningState(ctx)
	if err != nil {
		return err
	}
	totalMints, err := client.TotalMints(ctx)
	if err != nil {
		return err
	}
	expected := miner.ExpectedHashes(state.Difficulty)
	targetHex := formatTarget(state.Difficulty)

	fmt.Printf("contract: %s\n", client.Contract().Hex())
	fmt.Printf("chain id:  %s\n", chainID)
	fmt.Printf("genesis:   complete=%t minted=%s HASH remaining=%s HASH eth_raised=%s ETH\n",
		genesis.Complete,
		formatUnits(genesis.Minted, 18, 4),
		formatUnits(genesis.Remaining, 18, 4),
		formatUnits(genesis.EthRaised, 18, 6),
	)
	fmt.Printf("mining:    era=%s reward=%s HASH total_mints=%s\n",
		state.Era,
		formatUnits(state.Reward, 18, 4),
		totalMints,
	)
	fmt.Printf("supply:    mined=%s HASH remaining=%s HASH\n",
		formatUnits(state.Minted, 18, 4),
		formatUnits(state.Remaining, 18, 4),
	)
	fmt.Printf("epoch:     %s blocks_left=%s\n", state.Epoch, state.EpochBlocksLeft)
	fmt.Printf("work:      ~%s hashes/solution\n", formatHashCount(expected))
	fmt.Printf("target:    %s\n", targetHex)

	if *address != "" {
		if !common.IsHexAddress(*address) {
			return fmt.Errorf("invalid address: %s", *address)
		}
		addr := common.HexToAddress(*address)
		challenge, err := client.Challenge(ctx, addr)
		if err != nil {
			return err
		}
		balance, err := client.BalanceOf(ctx, addr)
		if err != nil {
			return err
		}
		fmt.Printf("miner:     %s\n", addr.Hex())
		fmt.Printf("challenge: %s\n", challenge.Hex())
		fmt.Printf("balance:   %s HASH\n", formatUnits(balance, 18, 4))
	}

	if !genesis.Complete {
		fmt.Println("note:      mining is not open until genesisComplete=true")
	}
	return nil
}

func runMine(args []string) error {
	fs := flag.NewFlagSet("mine", flag.ExitOnError)
	rpcURL := fs.String("rpc", "", "Ethereum RPC URL; falls back to ETH_RPC_URL")
	privateKeyHex := fs.String("private-key", "", "private key; falls back to PRIVATE_KEY")
	addressHex := fs.String("address", "", "miner address for no-submit mode")
	contract := fs.String("contract", hash256.DefaultContract, "HASH contract address")
	backend := fs.String("backend", env("HASHMINER_BACKEND", "cpu"), "mining backend: cpu, gpu/opencl, cuda, hybrid, or cuda+cpu")
	threads := fs.Int("threads", runtime.NumCPU(), "CPU worker count")
	gpuDevice := fs.Int("gpu-device", 0, "OpenCL GPU device index")
	gpuWorkgroups := fs.Uint("gpu-workgroups", defaultGPUWorkgroups, "OpenCL GPU dispatch workgroup count")
	gpuLocalSize := fs.Uint("gpu-local-size", 64, "OpenCL GPU local work size")
	gpuIterations := fs.Uint("gpu-iterations", 16, "OpenCL GPU hashes per work item")
	cudaDevice := fs.Int("cuda-device", 0, "CUDA device index")
	cudaBlocks := fs.Uint("cuda-blocks", defaultCUDABlocks, "CUDA grid blocks per dispatch")
	cudaThreads := fs.Uint("cuda-threads", defaultCUDAThreads, "CUDA threads per block")
	cudaIterations := fs.Uint("cuda-iterations", defaultCUDAIterations, "CUDA hashes per thread")
	startRaw := fs.String("start", "", "start nonce as decimal or 0x hex; random uint64 if empty")
	reportEvery := fs.Duration("report", 5*time.Second, "progress report interval")
	refreshEvery := fs.Duration("refresh", 2*time.Minute, "refresh chain mining state while searching")
	noSubmit := fs.Bool("no-submit", false, "find a nonce but do not submit a transaction")
	keepMining := fs.Bool("keep", false, "continue after a submitted solution")
	waitReceipt := fs.Bool("wait", true, "wait for transaction receipt after submit")
	gasLimit := fs.Uint64("gas-limit", 0, "manual gas limit; 0 estimates gas")
	tipGwei := fs.String("priority-tip-gwei", "6", "EIP-1559 priority fee in gwei")
	maxFeeGwei := fs.String("max-fee-gwei", "", "optional EIP-1559 max fee in gwei")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *rpcURL == "" {
		*rpcURL = env("ETH_RPC_URL", "")
	}
	if *privateKeyHex == "" {
		*privateKeyHex = env("PRIVATE_KEY", "")
	}

	key, from, err := resolveMiner(*privateKeyHex, *addressHex)
	if err != nil {
		return err
	}
	if key == nil {
		*noSubmit = true
	}
	startNonce, err := parseStartNonce(*startRaw)
	if err != nil {
		return err
	}
	tipCap, err := parseDecimalUnits(*tipGwei, 9)
	if err != nil {
		return fmt.Errorf("priority-tip-gwei: %w", err)
	}
	var feeCap *big.Int
	if *maxFeeGwei != "" {
		feeCap, err = parseDecimalUnits(*maxFeeGwei, 9)
		if err != nil {
			return fmt.Errorf("max-fee-gwei: %w", err)
		}
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	client, err := hash256.Dial(ctx, *rpcURL, *contract)
	if err != nil {
		return err
	}
	defer client.Close()

	chainID, err := client.ChainID(ctx)
	if err != nil {
		return err
	}
	if chainID.Cmp(big.NewInt(1)) != 0 {
		fmt.Printf("warning: connected chain id is %s; HASH contract is documented on Ethereum mainnet\n", chainID)
	}

	for {
		genesis, err := client.GenesisState(ctx)
		if err != nil {
			return err
		}
		state, err := client.MiningState(ctx)
		if err != nil {
			return err
		}
		if !genesis.Complete {
			return fmt.Errorf("mining is not open: genesisComplete=false, sold=%s HASH, remaining=%s HASH",
				formatUnits(genesis.Minted, 18, 4),
				formatUnits(genesis.Remaining, 18, 4),
			)
		}
		if state.Remaining.Sign() <= 0 {
			return fmt.Errorf("mining supply is exhausted")
		}
		if state.Difficulty.Sign() <= 0 {
			return fmt.Errorf("contract difficulty is zero")
		}

		challenge, err := client.Challenge(ctx, from)
		if err != nil {
			return err
		}

		targetHex := formatTarget(state.Difficulty)
		expected := miner.ExpectedHashes(state.Difficulty)
		fmt.Printf("miner:     %s\n", from.Hex())
		fmt.Printf("epoch:     %s blocks_left=%s\n", state.Epoch, state.EpochBlocksLeft)
		fmt.Printf("reward:    %s HASH\n", formatUnits(state.Reward, 18, 4))
		fmt.Printf("work:      ~%s hashes/solution\n", formatHashCount(expected))
		fmt.Printf("target:    %s\n", targetHex)
		fmt.Printf("challenge: %s\n", challenge.Hex())
		fmt.Printf("backend:   %s\n", *backend)
		if backendUsesOpenCL(*backend) {
			printOpenCLDevice(*gpuDevice)
			fmt.Printf("opencl:    workgroups=%d local=%d iterations=%d\n", *gpuWorkgroups, *gpuLocalSize, *gpuIterations)
		}
		if backendUsesCUDA(*backend) {
			printCUDADevice(*cudaDevice)
			fmt.Printf("cuda:      blocks=%d threads=%d iterations=%d\n", *cudaBlocks, *cudaThreads, *cudaIterations)
		}
		if backendUsesCPU(*backend) {
			fmt.Printf("threads:   %d\n", *threads)
		}
		fmt.Printf("start:     %d\n", startNonce)

		result, refresh, err := searchUntilRefresh(ctx, searchUntilRefreshConfig{
			Client:         client,
			Miner:          from,
			InitialState:   state,
			InitialJob:     miningJob{Challenge: challenge, Difficulty: state.Difficulty, Epoch: state.Epoch},
			Backend:        *backend,
			StartNonce:     startNonce,
			Threads:        *threads,
			GPUDevice:      *gpuDevice,
			GPUWorkgroups:  uint32(*gpuWorkgroups),
			GPULocalSize:   uint32(*gpuLocalSize),
			GPUIterations:  uint32(*gpuIterations),
			CUDADevice:     *cudaDevice,
			CUDABlocks:     uint32(*cudaBlocks),
			CUDAThreads:    uint32(*cudaThreads),
			CUDAIterations: uint32(*cudaIterations),
			ReportInterval: *reportEvery,
			RefreshEvery:   *refreshEvery,
		}, func(job miningJob, p miner.Progress) {
			expected := miner.ExpectedHashes(job.Difficulty)
			fmt.Printf("hashrate:  %s/s hashes=%d elapsed=%s eta=%s epoch=%s\n",
				formatRate(p.Hashrate),
				p.Hashes,
				p.Elapsed.Round(time.Second),
				formatETA(expected, p.Hashrate),
				job.Epoch,
			)
		})
		if err != nil {
			return err
		}
		if refresh {
			startNonce = randomUint64()
			continue
		}

		fmt.Printf("found:     backend=%s nonce=%d hash=%s hashes=%d elapsed=%s\n",
			result.Backend,
			result.Nonce,
			result.Hash.Hex(),
			result.Hashes,
			result.Elapsed.Round(time.Millisecond),
		)

		if *noSubmit {
			return nil
		}

		currentState, _, ok, err := validateSolutionFresh(ctx, client, from, result)
		if err != nil {
			return err
		}
		if !ok {
			fmt.Printf("stale:     found nonce no longer valid; refreshed epoch=%s work=~%s\n",
				currentState.Epoch,
				formatHashCount(miner.ExpectedHashes(currentState.Difficulty)),
			)
			startNonce = randomUint64()
			continue
		}

		txHash, err := client.SubmitMine(ctx, key, new(big.Int).SetUint64(result.Nonce), hash256.TxOptions{
			GasLimit: *gasLimit,
			TipCap:   tipCap,
			FeeCap:   feeCap,
		})
		if err != nil {
			return err
		}
		fmt.Printf("tx:        %s\n", txHash.Hex())

		if *waitReceipt {
			receiptCtx, cancel := context.WithTimeout(ctx, 10*time.Minute)
			receipt, err := client.WaitReceipt(receiptCtx, txHash, 5*time.Second)
			cancel()
			if err != nil {
				return fmt.Errorf("wait receipt: %w", err)
			}
			if receipt.Status == 1 {
				fmt.Printf("confirmed: block=%d gas_used=%d\n", receipt.BlockNumber.Uint64(), receipt.GasUsed)
			} else {
				fmt.Printf("reverted:  block=%d gas_used=%d\n", receipt.BlockNumber.Uint64(), receipt.GasUsed)
			}
		}

		if !*keepMining {
			return nil
		}
		startNonce = randomUint64()
	}
}

func runBench(args []string) error {
	fs := flag.NewFlagSet("bench", flag.ExitOnError)
	backend := fs.String("backend", env("HASHMINER_BACKEND", "cpu"), "mining backend: cpu, gpu/opencl, cuda, hybrid, or cuda+cpu")
	threads := fs.Int("threads", runtime.NumCPU(), "CPU worker count")
	gpuDevice := fs.Int("gpu-device", 0, "OpenCL GPU device index")
	gpuWorkgroups := fs.Uint("gpu-workgroups", defaultGPUWorkgroups, "OpenCL GPU dispatch workgroup count")
	gpuLocalSize := fs.Uint("gpu-local-size", 64, "OpenCL GPU local work size")
	gpuIterations := fs.Uint("gpu-iterations", 16, "OpenCL GPU hashes per work item")
	cudaDevice := fs.Int("cuda-device", 0, "CUDA device index")
	cudaBlocks := fs.Uint("cuda-blocks", defaultCUDABlocks, "CUDA grid blocks per dispatch")
	cudaThreads := fs.Uint("cuda-threads", defaultCUDAThreads, "CUDA threads per block")
	cudaIterations := fs.Uint("cuda-iterations", defaultCUDAIterations, "CUDA hashes per thread")
	duration := fs.Duration("duration", 10*time.Second, "benchmark duration")
	if err := fs.Parse(args); err != nil {
		return err
	}

	ctx, cancel := context.WithTimeout(context.Background(), *duration)
	defer cancel()

	challenge := common.HexToHash("0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef")
	difficulty := big.NewInt(1)

	var last miner.Progress
	if backendUsesOpenCL(*backend) {
		printOpenCLDevice(*gpuDevice)
	}
	if backendUsesCUDA(*backend) {
		printCUDADevice(*cudaDevice)
	}
	_, err := searchNonce(ctx, *backend, searchConfig{
		Challenge:      challenge,
		Difficulty:     difficulty,
		StartNonce:     randomUint64(),
		Threads:        *threads,
		GPUDevice:      *gpuDevice,
		GPUWorkgroups:  uint32(*gpuWorkgroups),
		GPULocalSize:   uint32(*gpuLocalSize),
		GPUIterations:  uint32(*gpuIterations),
		CUDADevice:     *cudaDevice,
		CUDABlocks:     uint32(*cudaBlocks),
		CUDAThreads:    uint32(*cudaThreads),
		CUDAIterations: uint32(*cudaIterations),
		ReportInterval: time.Second,
	}, func(p miner.Progress) {
		last = p
	})
	if err != nil && err != miner.ErrStopped {
		return err
	}
	if last.Hashrate == 0 {
		return fmt.Errorf("benchmark ended before first progress sample")
	}
	fmt.Printf("backend:  %s\n", *backend)
	if backendUsesCPU(*backend) {
		fmt.Printf("threads:  %d\n", *threads)
	}
	fmt.Printf("hashrate: %s/s\n", formatRate(last.Hashrate))
	return nil
}

func runGPUInfo(args []string) error {
	fs := flag.NewFlagSet("gpu-info", flag.ExitOnError)
	if err := fs.Parse(args); err != nil {
		return err
	}
	openCLDevices, openCLErr := gpuminer.ListDevices()
	cudaDevices, cudaErr := cudaminer.ListDevices()
	if openCLErr != nil && cudaErr != nil {
		return fmt.Errorf("OpenCL: %v; CUDA: %v", openCLErr, cudaErr)
	}
	if openCLErr == nil {
		fmt.Println("OpenCL:")
		for i, device := range openCLDevices {
			fmt.Printf("  [%d] %s · %s\n", i, device.Name, device.Vendor)
			fmt.Printf("      platform: %s · %s\n", device.PlatformName, device.PlatformVendor)
			fmt.Printf("      max workgroup size: %d\n", device.MaxWorkGroupSize)
		}
	} else {
		fmt.Printf("OpenCL: %v\n", openCLErr)
	}
	if cudaErr == nil {
		fmt.Println("CUDA:")
		for i, device := range cudaDevices {
			fmt.Printf("  [%d] %s\n", i, device.Name)
			fmt.Printf("      compute capability: %s\n", device.ComputeCapability)
			fmt.Printf("      multiprocessors: %d\n", device.Multiprocessors)
			fmt.Printf("      max threads/block: %d\n", device.MaxThreadsPerBlock)
		}
	} else {
		fmt.Printf("CUDA: %v\n", cudaErr)
	}
	return nil
}

func runCUDAInfo(args []string) error {
	fs := flag.NewFlagSet("cuda-info", flag.ExitOnError)
	if err := fs.Parse(args); err != nil {
		return err
	}
	devices, err := cudaminer.ListDevices()
	if err != nil {
		return err
	}
	for i, device := range devices {
		fmt.Printf("[%d] %s\n", i, device.Name)
		fmt.Printf("    compute capability: %s\n", device.ComputeCapability)
		fmt.Printf("    multiprocessors: %d\n", device.Multiprocessors)
		fmt.Printf("    max threads/block: %d\n", device.MaxThreadsPerBlock)
	}
	return nil
}

func runPoolMine(args []string) error {
	fs := flag.NewFlagSet("pool-mine", flag.ExitOnError)
	poolURL := fs.String("pool", env("HASH_POOL_URL", "http://127.0.0.1:8080"), "pool server URL")
	minerNameRaw := fs.String("miner", env("POOL_PAYOUT", env("POOL_MINER", "")), "payout address for pool stats and payouts")
	payoutRaw := fs.String("payout", "", "payout address alias; overrides --miner")
	worker := fs.String("worker", env("POOL_WORKER", defaultWorkerName()), "worker name")
	backend := fs.String("backend", env("HASHMINER_BACKEND", "cpu"), "mining backend: cpu, gpu/opencl, cuda, hybrid, or cuda+cpu")
	threads := fs.Int("threads", runtime.NumCPU(), "CPU worker count")
	gpuDevice := fs.Int("gpu-device", 0, "OpenCL GPU device index")
	gpuWorkgroups := fs.Uint("gpu-workgroups", defaultGPUWorkgroups, "OpenCL GPU dispatch workgroup count")
	gpuLocalSize := fs.Uint("gpu-local-size", 64, "OpenCL GPU local work size")
	gpuIterations := fs.Uint("gpu-iterations", 16, "OpenCL GPU hashes per work item")
	cudaDevice := fs.Int("cuda-device", 0, "CUDA device index")
	cudaBlocks := fs.Uint("cuda-blocks", defaultCUDABlocks, "CUDA grid blocks per dispatch")
	cudaThreads := fs.Uint("cuda-threads", defaultCUDAThreads, "CUDA threads per block")
	cudaIterations := fs.Uint("cuda-iterations", defaultCUDAIterations, "CUDA hashes per thread")
	startRaw := fs.String("start", "", "start nonce as decimal or 0x hex; random uint64 if empty")
	reportEvery := fs.Duration("report", 5*time.Second, "progress report interval")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if strings.TrimSpace(*payoutRaw) != "" {
		*minerNameRaw = *payoutRaw
	}
	minerName, err := resolvePoolMinerName(*minerNameRaw)
	if err != nil {
		return err
	}
	startNonce, err := parseStartNonce(*startRaw)
	if err != nil {
		return err
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	httpClient := &http.Client{Timeout: 30 * time.Second}
	lastJobID := ""

	for {
		job, err := fetchPoolJob(ctx, httpClient, *poolURL, minerName, *worker)
		if err != nil {
			if ctx.Err() != nil {
				return nil
			}
			return err
		}
		shareTarget, err := parseHexUint256(job.ShareTarget)
		if err != nil {
			return fmt.Errorf("pool share target: %w", err)
		}
		challenge := common.HexToHash(job.Challenge)
		refreshAfter := time.Duration(job.RefreshAfterSecs) * time.Second
		if refreshAfter <= 0 {
			refreshAfter = time.Minute
		}

		if job.JobID != lastJobID {
			fmt.Printf("pool:      %s\n", strings.TrimRight(*poolURL, "/"))
			fmt.Printf("payout:    %s\n", minerName)
			fmt.Printf("worker:    %s\n", *worker)
			fmt.Printf("job:       %s\n", job.JobID)
			fmt.Printf("pool addr: %s\n", job.PoolAddress)
			fmt.Printf("epoch:     %s blocks_left=%s\n", job.Epoch, job.EpochBlocksLeft)
			fmt.Printf("network:   ~%s hashes/solution\n", job.NetworkWork)
			fmt.Printf("share:     ~%s hashes/share\n", job.ShareWorkHuman)
			fmt.Printf("auth:      payout address only; no miner private key required\n")
			fmt.Printf("backend:   %s\n", *backend)
			if backendUsesOpenCL(*backend) {
				printOpenCLDevice(*gpuDevice)
				fmt.Printf("opencl:    workgroups=%d local=%d iterations=%d\n", *gpuWorkgroups, *gpuLocalSize, *gpuIterations)
			}
			if backendUsesCUDA(*backend) {
				printCUDADevice(*cudaDevice)
				fmt.Printf("cuda:      blocks=%d threads=%d iterations=%d\n", *cudaBlocks, *cudaThreads, *cudaIterations)
			}
			if backendUsesCPU(*backend) {
				fmt.Printf("threads:   %d\n", *threads)
			}
			lastJobID = job.JobID
		}

		searchCtx, cancel := context.WithTimeout(ctx, refreshAfter)
		result, err := searchNonce(searchCtx, *backend, searchConfig{
			Challenge:      challenge,
			Difficulty:     shareTarget,
			StartNonce:     startNonce,
			Threads:        *threads,
			GPUDevice:      *gpuDevice,
			GPUWorkgroups:  uint32(*gpuWorkgroups),
			GPULocalSize:   uint32(*gpuLocalSize),
			GPUIterations:  uint32(*gpuIterations),
			CUDADevice:     *cudaDevice,
			CUDABlocks:     uint32(*cudaBlocks),
			CUDAThreads:    uint32(*cudaThreads),
			CUDAIterations: uint32(*cudaIterations),
			ReportInterval: *reportEvery,
		}, func(p miner.Progress) {
			fmt.Printf("share-rate:%s/s hashes=%d elapsed=%s eta=%s job=%s\n",
				formatRate(p.Hashrate),
				p.Hashes,
				p.Elapsed.Round(time.Second),
				formatETA(miner.ExpectedHashes(shareTarget), p.Hashrate),
				shortID(job.JobID),
			)
		})
		cancel()
		if err != nil {
			if ctx.Err() != nil {
				return nil
			}
			if err == miner.ErrStopped {
				fmt.Println("refresh:   fetching latest pool job")
				startNonce = randomUint64()
				continue
			}
			return err
		}

		resp, err := submitPoolShare(ctx, httpClient, *poolURL, poolShareRequest{
			Miner:  minerName,
			Worker: *worker,
			JobID:  job.JobID,
			Nonce:  strconv.FormatUint(result.Nonce, 10),
		})
		if err != nil {
			return err
		}
		if resp.Accepted {
			fmt.Printf("accepted:  nonce=%s hash=%s score=%s", resp.Nonce, resp.Hash, formatHashCount(new(big.Float).SetInt(parseScoreWork(resp.Score))))
			if resp.BlockCandidate {
				fmt.Print(" block_candidate=true")
			}
			if resp.TxHash != "" {
				fmt.Printf(" tx=%s", resp.TxHash)
			}
			if resp.TxError != "" {
				fmt.Printf(" tx_error=%q", resp.TxError)
			}
			fmt.Println()
		} else {
			fmt.Printf("rejected:  nonce=%s reason=%s\n", resp.Nonce, resp.Reason)
		}

		startNonce = result.Nonce + 1
		if startNonce == 0 {
			startNonce = randomUint64()
		}
	}
}

type searchConfig struct {
	Challenge      common.Hash
	Difficulty     *big.Int
	StartNonce     uint64
	Threads        int
	GPUDevice      int
	GPUWorkgroups  uint32
	GPULocalSize   uint32
	GPUIterations  uint32
	CUDADevice     int
	CUDABlocks     uint32
	CUDAThreads    uint32
	CUDAIterations uint32
	ReportInterval time.Duration
}

type miningJob struct {
	Challenge  common.Hash
	Difficulty *big.Int
	Epoch      *big.Int
}

type searchUntilRefreshConfig struct {
	Client         *hash256.Client
	Miner          common.Address
	InitialState   hash256.MiningState
	InitialJob     miningJob
	Backend        string
	StartNonce     uint64
	Threads        int
	GPUDevice      int
	GPUWorkgroups  uint32
	GPULocalSize   uint32
	GPUIterations  uint32
	CUDADevice     int
	CUDABlocks     uint32
	CUDAThreads    uint32
	CUDAIterations uint32
	ReportInterval time.Duration
	RefreshEvery   time.Duration
}

func searchUntilRefresh(ctx context.Context, cfg searchUntilRefreshConfig, onProgress func(miningJob, miner.Progress)) (miner.Result, bool, error) {
	job := cfg.InitialJob
	state := cfg.InitialState
	refreshEvery := cfg.RefreshEvery
	if refreshEvery <= 0 {
		refreshEvery = 2 * time.Minute
	}
	deadline := time.Now().Add(refreshEvery)
	if d := epochRefreshDuration(state); d > 0 && time.Now().Add(d).Before(deadline) {
		deadline = time.Now().Add(d)
	}

	searchCtx, cancel := context.WithDeadline(ctx, deadline)
	defer cancel()

	result, err := searchNonce(searchCtx, cfg.Backend, searchConfig{
		Challenge:      job.Challenge,
		Difficulty:     job.Difficulty,
		StartNonce:     cfg.StartNonce,
		Threads:        cfg.Threads,
		GPUDevice:      cfg.GPUDevice,
		GPUWorkgroups:  cfg.GPUWorkgroups,
		GPULocalSize:   cfg.GPULocalSize,
		GPUIterations:  cfg.GPUIterations,
		CUDADevice:     cfg.CUDADevice,
		CUDABlocks:     cfg.CUDABlocks,
		CUDAThreads:    cfg.CUDAThreads,
		CUDAIterations: cfg.CUDAIterations,
		ReportInterval: cfg.ReportInterval,
	}, func(p miner.Progress) {
		if onProgress != nil {
			onProgress(job, p)
		}
	})
	if err == nil {
		return result, false, nil
	}
	if ctx.Err() != nil {
		return miner.Result{}, false, err
	}
	if err != miner.ErrStopped {
		return miner.Result{}, false, err
	}

	nextState, err := cfg.Client.MiningState(ctx)
	if err != nil {
		return miner.Result{}, false, err
	}
	nextChallenge, err := cfg.Client.Challenge(ctx, cfg.Miner)
	if err != nil {
		return miner.Result{}, false, err
	}
	if !sameBig(job.Epoch, nextState.Epoch) || !sameBig(job.Difficulty, nextState.Difficulty) || nextChallenge != job.Challenge {
		fmt.Printf("refresh:   epoch %s -> %s, work ~%s -> ~%s\n",
			formatBig(job.Epoch),
			formatBig(nextState.Epoch),
			formatHashCount(miner.ExpectedHashes(job.Difficulty)),
			formatHashCount(miner.ExpectedHashes(nextState.Difficulty)),
		)
		return miner.Result{}, true, nil
	}

	fmt.Printf("refresh:   chain state unchanged; continuing epoch=%s work=~%s\n",
		formatBig(nextState.Epoch),
		formatHashCount(miner.ExpectedHashes(nextState.Difficulty)),
	)
	return miner.Result{}, true, nil
}

func epochRefreshDuration(state hash256.MiningState) time.Duration {
	if state.EpochBlocksLeft == nil || state.EpochBlocksLeft.Sign() <= 0 {
		return 0
	}
	blocks := state.EpochBlocksLeft.Int64()
	if blocks <= 0 {
		return 0
	}
	if blocks > 2 {
		blocks -= 1
	}
	return time.Duration(blocks) * 12 * time.Second
}

func sameBig(a *big.Int, b *big.Int) bool {
	if a == nil || b == nil {
		return a == nil && b == nil
	}
	return a.Cmp(b) == 0
}

func formatBig(v *big.Int) string {
	if v == nil {
		return "?"
	}
	return v.String()
}

func validateSolutionFresh(ctx context.Context, client *hash256.Client, addr common.Address, result miner.Result) (hash256.MiningState, common.Hash, bool, error) {
	state, err := client.MiningState(ctx)
	if err != nil {
		return hash256.MiningState{}, common.Hash{}, false, err
	}
	challenge, err := client.Challenge(ctx, addr)
	if err != nil {
		return hash256.MiningState{}, common.Hash{}, false, err
	}
	hash := miner.HashNonce(challenge, result.Nonce)
	target, err := miner.TargetBytes(state.Difficulty)
	if err != nil {
		return hash256.MiningState{}, common.Hash{}, false, err
	}
	return state, challenge, bytes.Compare(hash[:], target[:]) < 0, nil
}

func searchNonce(ctx context.Context, backend string, cfg searchConfig, onProgress func(miner.Progress)) (miner.Result, error) {
	switch strings.ToLower(strings.TrimSpace(backend)) {
	case "", "cpu":
		result, err := miner.Search(ctx, miner.Job{
			Challenge:      cfg.Challenge,
			Difficulty:     cfg.Difficulty,
			StartNonce:     cfg.StartNonce,
			Threads:        cfg.Threads,
			ReportInterval: cfg.ReportInterval,
		}, onProgress)
		result.Backend = "cpu"
		return result, err
	case "gpu", "opencl":
		result, err := gpuminer.Search(ctx, gpuminer.Job{
			Challenge:      cfg.Challenge,
			Difficulty:     cfg.Difficulty,
			StartNonce:     cfg.StartNonce,
			DeviceIndex:    cfg.GPUDevice,
			Workgroups:     cfg.GPUWorkgroups,
			LocalSize:      cfg.GPULocalSize,
			Iterations:     cfg.GPUIterations,
			ReportInterval: cfg.ReportInterval,
		}, onProgress)
		result.Backend = "gpu"
		return result, err
	case "cuda":
		result, err := cudaminer.Search(ctx, cudaminer.Job{
			Challenge:      cfg.Challenge,
			Difficulty:     cfg.Difficulty,
			StartNonce:     cfg.StartNonce,
			DeviceIndex:    cfg.CUDADevice,
			Blocks:         cfg.CUDABlocks,
			Threads:        cfg.CUDAThreads,
			Iterations:     cfg.CUDAIterations,
			ReportInterval: cfg.ReportInterval,
		}, onProgress)
		result.Backend = "cuda"
		return result, err
	case "hybrid", "both", "all", "gpu+cpu", "cpu+gpu":
		return searchHybrid(ctx, cfg, onProgress)
	case "cuda+cpu", "cpu+cuda", "hybrid-cuda", "cuda-hybrid":
		return searchHybridCUDA(ctx, cfg, onProgress)
	default:
		return miner.Result{}, fmt.Errorf("unknown backend %q; use cpu, gpu/opencl, cuda, hybrid, or cuda+cpu", backend)
	}
}

type backendProgress struct {
	backend  string
	progress miner.Progress
}

type backendResult struct {
	backend string
	result  miner.Result
	err     error
}

func searchHybrid(ctx context.Context, cfg searchConfig, onProgress func(miner.Progress)) (miner.Result, error) {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	progressCh := make(chan backendProgress, 16)
	resultCh := make(chan backendResult, 2)

	var wg sync.WaitGroup
	wg.Add(2)

	go func() {
		defer wg.Done()
		result, err := miner.Search(ctx, miner.Job{
			Challenge:      cfg.Challenge,
			Difficulty:     cfg.Difficulty,
			StartNonce:     cfg.StartNonce,
			Threads:        cfg.Threads,
			ReportInterval: cfg.ReportInterval,
		}, func(p miner.Progress) {
			sendProgress(ctx, progressCh, backendProgress{backend: "cpu", progress: p})
		})
		result.Backend = "cpu"
		sendResult(ctx, resultCh, backendResult{backend: "cpu", result: result, err: err})
	}()

	go func() {
		defer wg.Done()
		result, err := gpuminer.Search(ctx, gpuminer.Job{
			Challenge:      cfg.Challenge,
			Difficulty:     cfg.Difficulty,
			StartNonce:     cfg.StartNonce + (uint64(1) << 63),
			DeviceIndex:    cfg.GPUDevice,
			Workgroups:     cfg.GPUWorkgroups,
			LocalSize:      cfg.GPULocalSize,
			Iterations:     cfg.GPUIterations,
			ReportInterval: cfg.ReportInterval,
		}, func(p miner.Progress) {
			sendProgress(ctx, progressCh, backendProgress{backend: "gpu", progress: p})
		})
		result.Backend = "gpu"
		sendResult(ctx, resultCh, backendResult{backend: "gpu", result: result, err: err})
	}()

	go func() {
		wg.Wait()
		close(progressCh)
		close(resultCh)
	}()

	progress := map[string]miner.Progress{}
	finished := 0

	for finished < 2 {
		select {
		case p, ok := <-progressCh:
			if !ok {
				progressCh = nil
				continue
			}
			progress[p.backend] = p.progress
			if onProgress != nil {
				onProgress(combineProgress(progress))
			}
		case r, ok := <-resultCh:
			if !ok {
				resultCh = nil
				continue
			}
			finished++
			if r.err == nil {
				cancel()
				return r.result, nil
			}
			if ctx.Err() == nil && r.err != miner.ErrStopped {
				cancel()
				return miner.Result{}, fmt.Errorf("hybrid miner %s backend failed: %w", r.backend, r.err)
			}
		case <-ctx.Done():
			return miner.Result{}, miner.ErrStopped
		}
	}

	return miner.Result{}, miner.ErrStopped
}

func searchHybridCUDA(ctx context.Context, cfg searchConfig, onProgress func(miner.Progress)) (miner.Result, error) {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	progressCh := make(chan backendProgress, 16)
	resultCh := make(chan backendResult, 2)

	var wg sync.WaitGroup
	wg.Add(2)

	go func() {
		defer wg.Done()
		result, err := miner.Search(ctx, miner.Job{
			Challenge:      cfg.Challenge,
			Difficulty:     cfg.Difficulty,
			StartNonce:     cfg.StartNonce,
			Threads:        cfg.Threads,
			ReportInterval: cfg.ReportInterval,
		}, func(p miner.Progress) {
			sendProgress(ctx, progressCh, backendProgress{backend: "cpu", progress: p})
		})
		result.Backend = "cpu"
		sendResult(ctx, resultCh, backendResult{backend: "cpu", result: result, err: err})
	}()

	go func() {
		defer wg.Done()
		result, err := cudaminer.Search(ctx, cudaminer.Job{
			Challenge:      cfg.Challenge,
			Difficulty:     cfg.Difficulty,
			StartNonce:     cfg.StartNonce + (uint64(1) << 63),
			DeviceIndex:    cfg.CUDADevice,
			Blocks:         cfg.CUDABlocks,
			Threads:        cfg.CUDAThreads,
			Iterations:     cfg.CUDAIterations,
			ReportInterval: cfg.ReportInterval,
		}, func(p miner.Progress) {
			sendProgress(ctx, progressCh, backendProgress{backend: "cuda", progress: p})
		})
		result.Backend = "cuda"
		sendResult(ctx, resultCh, backendResult{backend: "cuda", result: result, err: err})
	}()

	go func() {
		wg.Wait()
		close(progressCh)
		close(resultCh)
	}()

	progress := map[string]miner.Progress{}
	finished := 0

	for finished < 2 {
		select {
		case p, ok := <-progressCh:
			if !ok {
				progressCh = nil
				continue
			}
			progress[p.backend] = p.progress
			if onProgress != nil {
				onProgress(combineProgress(progress))
			}
		case r, ok := <-resultCh:
			if !ok {
				resultCh = nil
				continue
			}
			finished++
			if r.err == nil {
				cancel()
				return r.result, nil
			}
			if ctx.Err() == nil && r.err != miner.ErrStopped {
				cancel()
				return miner.Result{}, fmt.Errorf("hybrid miner %s backend failed: %w", r.backend, r.err)
			}
		case <-ctx.Done():
			return miner.Result{}, miner.ErrStopped
		}
	}

	return miner.Result{}, miner.ErrStopped
}

func sendProgress(ctx context.Context, ch chan<- backendProgress, p backendProgress) {
	select {
	case ch <- p:
	case <-ctx.Done():
	}
}

func sendResult(ctx context.Context, ch chan<- backendResult, r backendResult) {
	select {
	case ch <- r:
	case <-ctx.Done():
	}
}

func combineProgress(progress map[string]miner.Progress) miner.Progress {
	var out miner.Progress
	for _, p := range progress {
		out.Hashes += p.Hashes
		out.Hashrate += p.Hashrate
		if p.Elapsed > out.Elapsed {
			out.Elapsed = p.Elapsed
		}
	}
	return out
}

func backendUsesOpenCL(backend string) bool {
	switch strings.ToLower(strings.TrimSpace(backend)) {
	case "gpu", "opencl", "hybrid", "both", "all", "gpu+cpu", "cpu+gpu":
		return true
	default:
		return false
	}
}

func backendUsesCUDA(backend string) bool {
	switch strings.ToLower(strings.TrimSpace(backend)) {
	case "cuda", "cuda+cpu", "cpu+cuda", "hybrid-cuda", "cuda-hybrid":
		return true
	default:
		return false
	}
}

func backendUsesCPU(backend string) bool {
	switch strings.ToLower(strings.TrimSpace(backend)) {
	case "", "cpu", "hybrid", "both", "all", "gpu+cpu", "cpu+gpu", "cuda+cpu", "cpu+cuda", "hybrid-cuda", "cuda-hybrid":
		return true
	default:
		return false
	}
}

func printOpenCLDevice(index int) {
	devices, err := gpuminer.ListDevices()
	if err != nil {
		fmt.Printf("opencl:    %v\n", err)
		return
	}
	if index < 0 || index >= len(devices) {
		fmt.Printf("opencl:    device index %d not found\n", index)
		return
	}
	device := devices[index]
	fmt.Printf("opencl:    [%d] %s · %s\n", index, device.Name, device.Vendor)
}

func printCUDADevice(index int) {
	devices, err := cudaminer.ListDevices()
	if err != nil {
		fmt.Printf("cuda:      %v\n", err)
		return
	}
	if index < 0 || index >= len(devices) {
		fmt.Printf("cuda:      device index %d not found\n", index)
		return
	}
	device := devices[index]
	fmt.Printf("cuda:      [%d] %s cc=%s\n", index, device.Name, device.ComputeCapability)
}

type poolJobResponse struct {
	PoolAddress        string `json:"pool_address"`
	Challenge          string `json:"challenge"`
	Epoch              string `json:"epoch"`
	EpochBlocksLeft    string `json:"epoch_blocks_left"`
	RewardWei          string `json:"reward_wei"`
	RewardHASH         string `json:"reward_hash"`
	NetworkTarget      string `json:"network_target"`
	NetworkWork        string `json:"network_work"`
	ShareTarget        string `json:"share_target"`
	ShareWork          string `json:"share_work"`
	ShareWorkHuman     string `json:"share_work_human"`
	JobID              string `json:"job_id"`
	RefreshAfterSecs   int64  `json:"refresh_after_secs"`
	ServerTimeUnixSecs int64  `json:"server_time_unix_secs"`
}

type poolShareRequest struct {
	Miner  string `json:"miner"`
	Worker string `json:"worker"`
	JobID  string `json:"job_id"`
	Nonce  string `json:"nonce"`
}

type poolShareResponse struct {
	Accepted       bool   `json:"accepted"`
	Reason         string `json:"reason,omitempty"`
	Miner          string `json:"miner"`
	Worker         string `json:"worker,omitempty"`
	Nonce          string `json:"nonce"`
	Hash           string `json:"hash,omitempty"`
	Score          string `json:"score,omitempty"`
	BlockCandidate bool   `json:"block_candidate"`
	TxHash         string `json:"tx_hash,omitempty"`
	TxError        string `json:"tx_error,omitempty"`
	JobID          string `json:"job_id,omitempty"`
}

func fetchPoolJob(ctx context.Context, httpClient *http.Client, baseURL string, minerName string, worker string) (poolJobResponse, error) {
	endpoint, err := url.Parse(poolEndpoint(baseURL, "/job"))
	if err != nil {
		return poolJobResponse{}, err
	}
	q := endpoint.Query()
	q.Set("miner", minerName)
	q.Set("worker", worker)
	endpoint.RawQuery = q.Encode()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint.String(), nil)
	if err != nil {
		return poolJobResponse{}, err
	}
	resp, err := httpClient.Do(req)
	if err != nil {
		return poolJobResponse{}, err
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return poolJobResponse{}, err
	}
	if resp.StatusCode != http.StatusOK {
		return poolJobResponse{}, fmt.Errorf("pool job HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(data)))
	}
	var out poolJobResponse
	if err := json.Unmarshal(data, &out); err != nil {
		return poolJobResponse{}, err
	}
	if out.JobID == "" || out.Challenge == "" || out.ShareTarget == "" {
		return poolJobResponse{}, fmt.Errorf("pool returned incomplete job")
	}
	return out, nil
}

func submitPoolShare(ctx context.Context, httpClient *http.Client, baseURL string, reqBody poolShareRequest) (poolShareResponse, error) {
	var body bytes.Buffer
	if err := json.NewEncoder(&body).Encode(reqBody); err != nil {
		return poolShareResponse{}, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, poolEndpoint(baseURL, "/share"), &body)
	if err != nil {
		return poolShareResponse{}, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := httpClient.Do(req)
	if err != nil {
		return poolShareResponse{}, err
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return poolShareResponse{}, err
	}
	var out poolShareResponse
	if err := json.Unmarshal(data, &out); err != nil {
		return poolShareResponse{}, fmt.Errorf("pool share HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(data)))
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		if out.Reason == "" {
			out.Reason = fmt.Sprintf("HTTP %d", resp.StatusCode)
		}
		return out, nil
	}
	return out, nil
}

func poolEndpoint(baseURL string, path string) string {
	return strings.TrimRight(strings.TrimSpace(baseURL), "/") + path
}

func parseHexUint256(raw string) (*big.Int, error) {
	raw = strings.TrimSpace(strings.TrimPrefix(raw, "0x"))
	if raw == "" {
		return nil, fmt.Errorf("empty value")
	}
	n, ok := new(big.Int).SetString(raw, 16)
	if !ok {
		return nil, fmt.Errorf("invalid hex integer")
	}
	if n.Sign() <= 0 {
		return nil, fmt.Errorf("value must be positive")
	}
	if n.BitLen() > 256 {
		return nil, fmt.Errorf("value exceeds uint256")
	}
	return n, nil
}

func resolvePoolMinerName(raw string) (string, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return "", fmt.Errorf("pool mining requires --miner or --payout to be your payout address")
	}
	if !common.IsHexAddress(raw) {
		return "", fmt.Errorf("--miner or --payout must be an Ethereum address")
	}
	return common.HexToAddress(raw).Hex(), nil
}

func defaultWorkerName() string {
	name, err := os.Hostname()
	if err != nil || strings.TrimSpace(name) == "" {
		return "worker"
	}
	return strings.TrimSpace(name)
}

func parseScoreWork(raw string) *big.Int {
	n, ok := new(big.Int).SetString(strings.TrimSpace(raw), 10)
	if !ok || n.Sign() < 0 {
		return big.NewInt(0)
	}
	return n
}

func shortID(id string) string {
	if len(id) <= 10 {
		return id
	}
	return id[:10]
}

func resolveMiner(privateKeyHex string, addressHex string) (*ecdsa.PrivateKey, common.Address, error) {
	if privateKeyHex != "" {
		key, err := crypto.HexToECDSA(strings.TrimPrefix(privateKeyHex, "0x"))
		if err != nil {
			return nil, common.Address{}, fmt.Errorf("invalid private key: %w", err)
		}
		from := crypto.PubkeyToAddress(key.PublicKey)
		if addressHex != "" && common.IsHexAddress(addressHex) {
			addr := common.HexToAddress(addressHex)
			if addr != from {
				return nil, common.Address{}, fmt.Errorf("private key address %s does not match --address %s", from.Hex(), addr.Hex())
			}
		}
		return key, from, nil
	}
	if addressHex == "" {
		return nil, common.Address{}, fmt.Errorf("PRIVATE_KEY or --address is required")
	}
	if !common.IsHexAddress(addressHex) {
		return nil, common.Address{}, fmt.Errorf("invalid address: %s", addressHex)
	}
	return nil, common.HexToAddress(addressHex), nil
}

func parseStartNonce(raw string) (uint64, error) {
	if raw == "" {
		return randomUint64(), nil
	}
	v, ok := new(big.Int).SetString(strings.TrimSpace(raw), 0)
	if !ok {
		return 0, fmt.Errorf("invalid start nonce: %s", raw)
	}
	if v.Sign() < 0 || v.BitLen() > 64 {
		return 0, fmt.Errorf("start nonce must fit uint64")
	}
	return v.Uint64(), nil
}

func randomUint64() uint64 {
	var buf [8]byte
	if _, err := rand.Read(buf[:]); err != nil {
		return uint64(time.Now().UnixNano())
	}
	return binary.BigEndian.Uint64(buf[:])
}

func parseDecimalUnits(raw string, decimals int) (*big.Int, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil, fmt.Errorf("empty value")
	}
	if strings.HasPrefix(raw, "-") {
		return nil, fmt.Errorf("negative value")
	}
	parts := strings.Split(raw, ".")
	if len(parts) > 2 {
		return nil, fmt.Errorf("invalid decimal")
	}
	whole := parts[0]
	frac := ""
	if len(parts) == 2 {
		frac = parts[1]
	}
	if whole == "" {
		whole = "0"
	}
	if len(frac) > decimals {
		return nil, fmt.Errorf("too many decimal places")
	}
	for _, ch := range whole + frac {
		if ch < '0' || ch > '9' {
			return nil, fmt.Errorf("invalid digit %q", ch)
		}
	}
	frac += strings.Repeat("0", decimals-len(frac))
	n, ok := new(big.Int).SetString(whole+frac, 10)
	if !ok {
		return nil, fmt.Errorf("invalid number")
	}
	return n, nil
}

func formatUnits(v *big.Int, decimals int, places int) string {
	if v == nil {
		return "0"
	}
	sign := ""
	n := new(big.Int).Set(v)
	if n.Sign() < 0 {
		sign = "-"
		n.Abs(n)
	}
	base := new(big.Int).Exp(big.NewInt(10), big.NewInt(int64(decimals)), nil)
	whole := new(big.Int).Quo(n, base).String()
	frac := new(big.Int).Mod(n, base).String()
	frac = strings.Repeat("0", decimals-len(frac)) + frac
	if places < len(frac) {
		frac = frac[:places]
	}
	frac = strings.TrimRight(frac, "0")
	if frac == "" {
		return sign + whole
	}
	return sign + whole + "." + frac
}

func formatRate(rate float64) string {
	units := []string{"H", "KH", "MH", "GH", "TH"}
	i := 0
	for rate >= 1000 && i < len(units)-1 {
		rate /= 1000
		i++
	}
	return strconv.FormatFloat(rate, 'f', 2, 64) + " " + units[i]
}

func formatETA(expected *big.Float, rate float64) string {
	if expected == nil || rate <= 0 {
		return "unknown"
	}
	rateF := new(big.Float).SetPrec(256).SetFloat64(rate)
	seconds, _ := new(big.Float).Quo(expected, rateF).Float64()
	if math.IsInf(seconds, 0) || math.IsNaN(seconds) {
		return "unknown"
	}
	if seconds > float64(math.MaxInt64)/float64(time.Second) {
		return "very long"
	}
	return time.Duration(seconds * float64(time.Second)).Round(time.Second).String()
}

func formatHashCount(v *big.Float) string {
	if v == nil {
		return "0 H"
	}
	f, _ := v.Float64()
	if math.IsInf(f, 0) || math.IsNaN(f) {
		return v.Text('e', 3) + " H"
	}
	return formatRate(f)
}

func formatFloat(v *big.Float, places int) string {
	if v == nil {
		return "0"
	}
	f, _ := v.Float64()
	if math.IsInf(f, 0) {
		return v.Text('e', places)
	}
	return strconv.FormatFloat(f, 'f', places, 64)
}

const (
	defaultGPUWorkgroups  = 65536
	defaultCUDABlocks     = 65536
	defaultCUDAThreads    = 128
	defaultCUDAIterations = 16
)

func formatTarget(v *big.Int) string {
	if v == nil || v.Sign() <= 0 {
		return "0x" + strings.Repeat("0", 64)
	}
	out, err := miner.TargetHex(v)
	if err != nil {
		return "invalid"
	}
	return out
}

func env(key string, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}

func envBool(key string, fallback bool) bool {
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" {
		return fallback
	}
	parsed, err := strconv.ParseBool(value)
	if err != nil {
		return fallback
	}
	return parsed
}

func envDuration(key string, fallback time.Duration) time.Duration {
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" {
		return fallback
	}
	parsed, err := time.ParseDuration(value)
	if err != nil {
		return fallback
	}
	return parsed
}

func loadDotEnv(path string) error {
	content, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("read %s: %w", path, err)
	}

	for lineNo, raw := range strings.Split(string(content), "\n") {
		line := strings.TrimSpace(strings.TrimSuffix(raw, "\r"))
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		line = strings.TrimSpace(strings.TrimPrefix(line, "export "))
		key, value, ok := strings.Cut(line, "=")
		if !ok {
			return fmt.Errorf("%s:%d: expected KEY=VALUE", path, lineNo+1)
		}
		key = strings.TrimSpace(key)
		value = strings.TrimSpace(value)
		if key == "" {
			return fmt.Errorf("%s:%d: empty key", path, lineNo+1)
		}
		if strings.ContainsAny(key, " \t") {
			return fmt.Errorf("%s:%d: invalid key %q", path, lineNo+1, key)
		}
		if len(value) >= 2 {
			quote := value[0]
			if (quote == '"' || quote == '\'') && value[len(value)-1] == quote {
				value = value[1 : len(value)-1]
			}
		}
		if os.Getenv(key) == "" {
			if err := os.Setenv(key, value); err != nil {
				return fmt.Errorf("%s:%d: set %s: %w", path, lineNo+1, key, err)
			}
		}
	}
	return nil
}

func usage() {
	fmt.Print(`hashminer - HASH 本地 CLI 礦工

用法:
  hashminer status --rpc <url> [--address 0x...]
  hashminer mine   --rpc <url> --private-key <hex> [--backend cpu|gpu|cuda|hybrid|cuda+cpu]
  hashminer mine   --rpc <url> --address 0x... --no-submit
  hashminer bench  [--backend cpu|gpu|cuda|hybrid|cuda+cpu]
  hashminer gpu-info
  hashminer cuda-info
  hashminer pool-mine --pool http://host:8080 --payout 0x... [--backend gpu|cuda]

環境變數，可放在專案根目錄 .env:
  ETH_RPC_URL   Ethereum mainnet RPC URL
  PRIVATE_KEY   礦工私鑰；帳戶需要有 ETH 支付 gas
  HASHMINER_BACKEND   cpu、gpu/opencl、cuda、hybrid 或 cuda+cpu
  HASHMINER_CUDA_ARCH   NVRTC 編譯目標；不填時依裝置自動推導
  HASHMINER_NVRTC_DLL   指定 nvrtc64_*.dll 路徑；通常不需要
  HASH_POOL_URL   pool-mine 預設礦池 URL
  POOL_PAYOUT   pool-mine 收款地址；POOL_MINER 仍可作為相容別名
`)
}
