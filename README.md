# HASH Mining

Standalone HASH miner CLI. This repository contains only local mining and pool-worker code; it does not include pool-server, payout, Redis, or admin logic.

## Build

```powershell
go mod tidy
go build -o .\hashminer.exe .\cmd\hashminer
```

Linux/macOS:

```bash
# CPU only
go build -o ./hashminer ./cmd/hashminer

# OpenCL/CUDA
CGO_ENABLED=1 go build -o ./hashminer ./cmd/hashminer
```

CPU mining is pure Go. OpenCL/CUDA mining on Linux/macOS requires `CGO_ENABLED=1`, a C compiler, and the matching runtime libraries installed. If library discovery fails, set `HASHMINER_OPENCL_LIB`, `HASHMINER_CUDA_DRIVER`, or `HASHMINER_NVRTC_LIB` to the full library/framework path.

## Environment

Copy `.env.example` to `.env` and set:

```dotenv
ETH_RPC_URL=https://your-mainnet-rpc.example
PRIVATE_KEY=0xyour_private_key
HASHMINER_BACKEND=cuda
POOL_PAYOUT=0xyour_payout_address
```

`PRIVATE_KEY` is only needed for solo mining and paying mainnet gas. Pool mining only needs a payout address.

## Solo Mining

```powershell
.\hashminer.exe status --address 0x你的錢包地址
.\hashminer.exe mine --backend cuda --priority-tip-gwei 6
```

CPU/OpenCL/CUDA choices:

```powershell
.\hashminer.exe bench --backend cpu --duration 10s
.\hashminer.exe bench --backend gpu --duration 10s
.\hashminer.exe bench --backend cuda --duration 10s
.\hashminer.exe bench --backend cuda+cpu --duration 10s
```

## Pool Worker

Pool mining does not require a miner private key:

```powershell
.\hashminer.exe pool-mine --pool http://你的礦池IP:8080 --payout 0x你的收款地址 --worker rig1 --backend cuda
```

## Commands

- `status`: reads chain mining state.
- `mine`: solo mining and optional transaction submission.
- `bench`: local CPU/OpenCL/CUDA benchmark.
- `gpu-info`: OpenCL and CUDA device listing.
- `cuda-info`: CUDA device listing.
- `pool-mine`: connect to a HASH pool as a worker.

This project intentionally does not expose pool-server commands. Use the separate HASH Pool repository for server deployment.
