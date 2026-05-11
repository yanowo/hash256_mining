package hash256

import (
	"context"
	"crypto/ecdsa"
	"errors"
	"fmt"
	"math/big"
	"strings"
	"time"

	"github.com/ethereum/go-ethereum"
	"github.com/ethereum/go-ethereum/accounts/abi"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/crypto"
	"github.com/ethereum/go-ethereum/ethclient"
)

const DefaultContract = "0xAC7b5d06fa1e77D08aea40d46cB7C5923A87A0cc"

const contractABI = `[
	{"inputs":[{"internalType":"address","name":"miner","type":"address"}],"name":"getChallenge","outputs":[{"internalType":"bytes32","name":"","type":"bytes32"}],"stateMutability":"view","type":"function"},
	{"inputs":[],"name":"genesisState","outputs":[{"internalType":"uint256","name":"minted","type":"uint256"},{"internalType":"uint256","name":"remaining","type":"uint256"},{"internalType":"uint256","name":"ethRaised","type":"uint256"},{"internalType":"bool","name":"complete","type":"bool"}],"stateMutability":"view","type":"function"},
	{"inputs":[],"name":"miningState","outputs":[{"internalType":"uint256","name":"era","type":"uint256"},{"internalType":"uint256","name":"reward","type":"uint256"},{"internalType":"uint256","name":"difficulty","type":"uint256"},{"internalType":"uint256","name":"minted","type":"uint256"},{"internalType":"uint256","name":"remaining","type":"uint256"},{"internalType":"uint256","name":"epoch","type":"uint256"},{"internalType":"uint256","name":"epochBlocksLeft_","type":"uint256"}],"stateMutability":"view","type":"function"},
	{"inputs":[],"name":"totalMints","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
	{"inputs":[{"internalType":"address","name":"account","type":"address"}],"name":"balanceOf","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
	{"inputs":[{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"amount","type":"uint256"}],"name":"transfer","outputs":[{"internalType":"bool","name":"","type":"bool"}],"stateMutability":"nonpayable","type":"function"},
	{"inputs":[{"internalType":"uint256","name":"nonce","type":"uint256"}],"name":"mine","outputs":[],"stateMutability":"nonpayable","type":"function"}
]`

type Client struct {
	eth      *ethclient.Client
	abi      abi.ABI
	contract common.Address
}

type GenesisState struct {
	Minted    *big.Int
	Remaining *big.Int
	EthRaised *big.Int
	Complete  bool
}

type MiningState struct {
	Era             *big.Int
	Reward          *big.Int
	Difficulty      *big.Int
	Minted          *big.Int
	Remaining       *big.Int
	Epoch           *big.Int
	EpochBlocksLeft *big.Int
}

type TxOptions struct {
	GasLimit uint64
	TipCap   *big.Int
	FeeCap   *big.Int
}

func Dial(ctx context.Context, rpcURL string, contractAddress string) (*Client, error) {
	if rpcURL == "" {
		return nil, fmt.Errorf("rpc URL is required")
	}
	if !common.IsHexAddress(contractAddress) {
		return nil, fmt.Errorf("invalid contract address: %s", contractAddress)
	}
	parsed, err := abi.JSON(strings.NewReader(contractABI))
	if err != nil {
		return nil, err
	}
	eth, err := ethclient.DialContext(ctx, rpcURL)
	if err != nil {
		return nil, err
	}
	return &Client{
		eth:      eth,
		abi:      parsed,
		contract: common.HexToAddress(contractAddress),
	}, nil
}

func (c *Client) Close() {
	c.eth.Close()
}

func (c *Client) ChainID(ctx context.Context) (*big.Int, error) {
	return c.eth.ChainID(ctx)
}

func (c *Client) Contract() common.Address {
	return c.contract
}

func (c *Client) GenesisState(ctx context.Context) (GenesisState, error) {
	out, err := c.call(ctx, "genesisState")
	if err != nil {
		return GenesisState{}, err
	}
	return GenesisState{
		Minted:    cloneBig(out[0].(*big.Int)),
		Remaining: cloneBig(out[1].(*big.Int)),
		EthRaised: cloneBig(out[2].(*big.Int)),
		Complete:  out[3].(bool),
	}, nil
}

func (c *Client) MiningState(ctx context.Context) (MiningState, error) {
	out, err := c.call(ctx, "miningState")
	if err != nil {
		return MiningState{}, err
	}
	return MiningState{
		Era:             cloneBig(out[0].(*big.Int)),
		Reward:          cloneBig(out[1].(*big.Int)),
		Difficulty:      cloneBig(out[2].(*big.Int)),
		Minted:          cloneBig(out[3].(*big.Int)),
		Remaining:       cloneBig(out[4].(*big.Int)),
		Epoch:           cloneBig(out[5].(*big.Int)),
		EpochBlocksLeft: cloneBig(out[6].(*big.Int)),
	}, nil
}

func (c *Client) TotalMints(ctx context.Context) (*big.Int, error) {
	out, err := c.call(ctx, "totalMints")
	if err != nil {
		return nil, err
	}
	return cloneBig(out[0].(*big.Int)), nil
}

func (c *Client) BalanceOf(ctx context.Context, addr common.Address) (*big.Int, error) {
	out, err := c.call(ctx, "balanceOf", addr)
	if err != nil {
		return nil, err
	}
	return cloneBig(out[0].(*big.Int)), nil
}

func (c *Client) Challenge(ctx context.Context, miner common.Address) (common.Hash, error) {
	out, err := c.call(ctx, "getChallenge", miner)
	if err != nil {
		return common.Hash{}, err
	}
	switch v := out[0].(type) {
	case [32]byte:
		return common.BytesToHash(v[:]), nil
	case common.Hash:
		return v, nil
	default:
		return common.Hash{}, fmt.Errorf("unexpected challenge type %T", out[0])
	}
}

func (c *Client) SubmitMine(ctx context.Context, key *ecdsa.PrivateKey, nonce *big.Int, opts TxOptions) (common.Hash, error) {
	data, err := c.abi.Pack("mine", nonce)
	if err != nil {
		return common.Hash{}, err
	}
	return c.sendContractTx(ctx, key, data, opts)
}

func (c *Client) Transfer(ctx context.Context, key *ecdsa.PrivateKey, to common.Address, amount *big.Int, opts TxOptions) (common.Hash, error) {
	if amount == nil || amount.Sign() <= 0 {
		return common.Hash{}, fmt.Errorf("transfer amount must be positive")
	}
	data, err := c.abi.Pack("transfer", to, amount)
	if err != nil {
		return common.Hash{}, err
	}
	return c.sendContractTx(ctx, key, data, opts)
}

func (c *Client) sendContractTx(ctx context.Context, key *ecdsa.PrivateKey, data []byte, opts TxOptions) (common.Hash, error) {
	if key == nil {
		return common.Hash{}, fmt.Errorf("private key is required")
	}
	from := crypto.PubkeyToAddress(key.PublicKey)
	chainID, err := c.eth.ChainID(ctx)
	if err != nil {
		return common.Hash{}, err
	}
	accountNonce, err := c.eth.PendingNonceAt(ctx, from)
	if err != nil {
		return common.Hash{}, err
	}

	gasLimit := opts.GasLimit
	if gasLimit == 0 {
		estimate, err := c.eth.EstimateGas(ctx, ethereum.CallMsg{
			From: from,
			To:   &c.contract,
			Data: data,
		})
		if err != nil {
			return common.Hash{}, fmt.Errorf("estimate gas: %w", err)
		}
		gasLimit = estimate + estimate/2
	}

	tipCap := cloneBig(opts.TipCap)
	if tipCap == nil {
		tipCap, err = c.eth.SuggestGasTipCap(ctx)
		if err != nil {
			return common.Hash{}, fmt.Errorf("suggest priority fee: %w", err)
		}
	}

	feeCap := cloneBig(opts.FeeCap)
	if feeCap == nil {
		header, err := c.eth.HeaderByNumber(ctx, nil)
		if err != nil {
			return common.Hash{}, err
		}
		if header.BaseFee == nil {
			return common.Hash{}, fmt.Errorf("network does not expose EIP-1559 base fee")
		}
		feeCap = new(big.Int).Add(new(big.Int).Mul(header.BaseFee, big.NewInt(2)), tipCap)
	}
	if feeCap.Cmp(tipCap) < 0 {
		return common.Hash{}, fmt.Errorf("max fee per gas is lower than priority fee")
	}

	tx := types.NewTx(&types.DynamicFeeTx{
		ChainID:   chainID,
		Nonce:     accountNonce,
		GasTipCap: tipCap,
		GasFeeCap: feeCap,
		Gas:       gasLimit,
		To:        &c.contract,
		Value:     big.NewInt(0),
		Data:      data,
	})
	signed, err := types.SignTx(tx, types.LatestSignerForChainID(chainID), key)
	if err != nil {
		return common.Hash{}, err
	}
	if err := c.eth.SendTransaction(ctx, signed); err != nil {
		return common.Hash{}, err
	}
	return signed.Hash(), nil
}

func (c *Client) WaitReceipt(ctx context.Context, txHash common.Hash, interval time.Duration) (*types.Receipt, error) {
	if interval <= 0 {
		interval = 5 * time.Second
	}
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		receipt, err := c.eth.TransactionReceipt(ctx, txHash)
		if err == nil {
			return receipt, nil
		}
		if !errors.Is(err, ethereum.NotFound) {
			return nil, err
		}

		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-ticker.C:
		}
	}
}

func (c *Client) call(ctx context.Context, method string, params ...interface{}) ([]interface{}, error) {
	data, err := c.abi.Pack(method, params...)
	if err != nil {
		return nil, err
	}
	out, err := c.eth.CallContract(ctx, ethereum.CallMsg{
		To:   &c.contract,
		Data: data,
	}, nil)
	if err != nil {
		return nil, err
	}
	return c.abi.Unpack(method, out)
}

func cloneBig(v *big.Int) *big.Int {
	if v == nil {
		return nil
	}
	return new(big.Int).Set(v)
}
