package cudaminer

const cudaKernel = `
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

__device__ __forceinline__ uint64_t rotl64(uint64_t x, uint32_t n) {
	return (x << n) | (x >> (64u - n));
}

__device__ __forceinline__ uint32_t bswap32(uint32_t v) {
	return ((v & 0x000000ffu) << 24)
	     | ((v & 0x0000ff00u) << 8)
	     | ((v & 0x00ff0000u) >> 8)
	     | ((v & 0xff000000u) >> 24);
}

__device__ void keccak_f1600(uint64_t st[25]) {
	const uint64_t rndc[24] = {
		0x0000000000000001ULL, 0x0000000000008082ULL,
		0x800000000000808aULL, 0x8000000080008000ULL,
		0x000000000000808bULL, 0x0000000080000001ULL,
		0x8000000080008081ULL, 0x8000000000008009ULL,
		0x000000000000008aULL, 0x0000000000000088ULL,
		0x0000000080008009ULL, 0x000000008000000aULL,
		0x000000008000808bULL, 0x800000000000008bULL,
		0x8000000000008089ULL, 0x8000000000008003ULL,
		0x8000000000008002ULL, 0x8000000000000080ULL,
		0x000000000000800aULL, 0x800000008000000aULL,
		0x8000000080008081ULL, 0x8000000000008080ULL,
		0x0000000080000001ULL, 0x8000000080008008ULL
	};
	const uint32_t rotc[24] = {
		1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14,
		27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44
	};
	const uint32_t piln[24] = {
		10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4,
		15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1
	};

	uint64_t bc[5];
	for (uint32_t round = 0; round < 24; round++) {
		for (uint32_t i = 0; i < 5; i++) {
			bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
		}
		for (uint32_t i = 0; i < 5; i++) {
			uint64_t t = bc[(i + 4) % 5] ^ rotl64(bc[(i + 1) % 5], 1);
			for (uint32_t j = 0; j < 25; j += 5) {
				st[j + i] ^= t;
			}
		}

		uint64_t t = st[1];
		for (uint32_t i = 0; i < 24; i++) {
			uint32_t j = piln[i];
			uint64_t tmp = st[j];
			st[j] = rotl64(t, rotc[i]);
			t = tmp;
		}

		for (uint32_t j = 0; j < 25; j += 5) {
			for (uint32_t i = 0; i < 5; i++) {
				bc[i] = st[j + i];
			}
			for (uint32_t i = 0; i < 5; i++) {
				st[j + i] = bc[i] ^ ((~bc[(i + 1) % 5]) & bc[(i + 2) % 5]);
			}
		}

		st[0] ^= rndc[round];
	}
}

__device__ __forceinline__ int hash_lt_target(uint32_t h[8], const uint32_t *target) {
	for (uint32_t i = 0; i < 8; i++) {
		if (h[i] < target[i]) return 1;
		if (h[i] > target[i]) return 0;
	}
	return 0;
}

extern "C" __global__ void hash256_mine(
	const uint32_t *challenge,
	const uint32_t *target,
	uint64_t base_nonce,
	uint32_t iterations,
	uint32_t *result
) {
	uint64_t gid = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
	uint64_t nonce = base_nonce + gid * (uint64_t)iterations;

	for (uint32_t k = 0; k < iterations; k++) {
		uint64_t n = nonce + (uint64_t)k;
		uint32_t n_lo = (uint32_t)(n & 0xffffffffULL);
		uint32_t n_hi = (uint32_t)(n >> 32);

		uint64_t st[25];
		st[0] = ((uint64_t)challenge[1] << 32) | (uint64_t)challenge[0];
		st[1] = ((uint64_t)challenge[3] << 32) | (uint64_t)challenge[2];
		st[2] = ((uint64_t)challenge[5] << 32) | (uint64_t)challenge[4];
		st[3] = ((uint64_t)challenge[7] << 32) | (uint64_t)challenge[6];
		st[4] = 0ULL;
		st[5] = 0ULL;
		st[6] = 0ULL;
		st[7] = ((uint64_t)bswap32(n_lo) << 32) | (uint64_t)bswap32(n_hi);
		st[8] = 0x0000000000000001ULL;
		st[9] = 0ULL;
		st[10] = 0ULL;
		st[11] = 0ULL;
		st[12] = 0ULL;
		st[13] = 0ULL;
		st[14] = 0ULL;
		st[15] = 0ULL;
		st[16] = 0x8000000000000000ULL;
		st[17] = 0ULL;
		st[18] = 0ULL;
		st[19] = 0ULL;
		st[20] = 0ULL;
		st[21] = 0ULL;
		st[22] = 0ULL;
		st[23] = 0ULL;
		st[24] = 0ULL;

		keccak_f1600(st);

		uint32_t h[8];
		h[0] = bswap32((uint32_t)(st[0] & 0xffffffffULL));
		h[1] = bswap32((uint32_t)(st[0] >> 32));
		h[2] = bswap32((uint32_t)(st[1] & 0xffffffffULL));
		h[3] = bswap32((uint32_t)(st[1] >> 32));
		h[4] = bswap32((uint32_t)(st[2] & 0xffffffffULL));
		h[5] = bswap32((uint32_t)(st[2] >> 32));
		h[6] = bswap32((uint32_t)(st[3] & 0xffffffffULL));
		h[7] = bswap32((uint32_t)(st[3] >> 32));

		if (hash_lt_target(h, target)) {
			uint32_t prior = atomicCAS((unsigned int *)&result[0], 0u, 1u);
			if (prior == 0u) {
				result[1] = n_lo;
				result[2] = n_hi;
				result[3] = 0u;
				for (uint32_t i = 0; i < 8; i++) {
					result[4 + i] = h[i];
				}
			}
			return;
		}
	}
}
`
