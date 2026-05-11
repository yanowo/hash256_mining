package gpuminer

const openCLKernel = `
ulong rotl64(ulong x, uint n) {
	return (x << n) | (x >> (64u - n));
}

uint bswap32(uint v) {
	return ((v & 0x000000ffu) << 24)
	     | ((v & 0x0000ff00u) << 8)
	     | ((v & 0x00ff0000u) >> 8)
	     | ((v & 0xff000000u) >> 24);
}

void keccak_f1600(ulong st[25]) {
	const ulong rndc[24] = {
		0x0000000000000001UL, 0x0000000000008082UL,
		0x800000000000808aUL, 0x8000000080008000UL,
		0x000000000000808bUL, 0x0000000080000001UL,
		0x8000000080008081UL, 0x8000000000008009UL,
		0x000000000000008aUL, 0x0000000000000088UL,
		0x0000000080008009UL, 0x000000008000000aUL,
		0x000000008000808bUL, 0x800000000000008bUL,
		0x8000000000008089UL, 0x8000000000008003UL,
		0x8000000000008002UL, 0x8000000000000080UL,
		0x000000000000800aUL, 0x800000008000000aUL,
		0x8000000080008081UL, 0x8000000000008080UL,
		0x0000000080000001UL, 0x8000000080008008UL
	};
	const uint rotc[24] = {
		1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14,
		27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44
	};
	const uint piln[24] = {
		10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4,
		15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1
	};

	ulong bc[5];
	for (uint round = 0; round < 24; round++) {
		for (uint i = 0; i < 5; i++) {
			bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
		}
		for (uint i = 0; i < 5; i++) {
			ulong t = bc[(i + 4) % 5] ^ rotl64(bc[(i + 1) % 5], 1);
			for (uint j = 0; j < 25; j += 5) {
				st[j + i] ^= t;
			}
		}

		ulong t = st[1];
		for (uint i = 0; i < 24; i++) {
			uint j = piln[i];
			ulong tmp = st[j];
			st[j] = rotl64(t, rotc[i]);
			t = tmp;
		}

		for (uint j = 0; j < 25; j += 5) {
			for (uint i = 0; i < 5; i++) {
				bc[i] = st[j + i];
			}
			for (uint i = 0; i < 5; i++) {
				st[j + i] = bc[i] ^ ((~bc[(i + 1) % 5]) & bc[(i + 2) % 5]);
			}
		}

		st[0] ^= rndc[round];
	}
}

int hash_lt_target(uint h[8], __global const uint *target) {
	for (uint i = 0; i < 8; i++) {
		if (h[i] < target[i]) return 1;
		if (h[i] > target[i]) return 0;
	}
	return 0;
}

__kernel void hash256_mine(
	__global const uint *challenge,
	__global const uint *target,
	const ulong base_nonce,
	const uint iterations,
	__global uint *result
) {
	ulong gid = get_global_id(0);
	ulong nonce = base_nonce + gid * (ulong)iterations;

	for (uint k = 0; k < iterations; k++) {
		ulong n = nonce + (ulong)k;
		uint n_lo = (uint)(n & 0xffffffffUL);
		uint n_hi = (uint)(n >> 32);

		ulong st[25];
		st[0] = ((ulong)challenge[1] << 32) | (ulong)challenge[0];
		st[1] = ((ulong)challenge[3] << 32) | (ulong)challenge[2];
		st[2] = ((ulong)challenge[5] << 32) | (ulong)challenge[4];
		st[3] = ((ulong)challenge[7] << 32) | (ulong)challenge[6];
		st[4] = 0UL;
		st[5] = 0UL;
		st[6] = 0UL;
		st[7] = ((ulong)bswap32(n_lo) << 32) | (ulong)bswap32(n_hi);
		st[8] = 0x0000000000000001UL;
		st[9] = 0UL;
		st[10] = 0UL;
		st[11] = 0UL;
		st[12] = 0UL;
		st[13] = 0UL;
		st[14] = 0UL;
		st[15] = 0UL;
		st[16] = 0x8000000000000000UL;
		st[17] = 0UL;
		st[18] = 0UL;
		st[19] = 0UL;
		st[20] = 0UL;
		st[21] = 0UL;
		st[22] = 0UL;
		st[23] = 0UL;
		st[24] = 0UL;

		keccak_f1600(st);

		uint h[8];
		h[0] = bswap32((uint)(st[0] & 0xffffffffUL));
		h[1] = bswap32((uint)(st[0] >> 32));
		h[2] = bswap32((uint)(st[1] & 0xffffffffUL));
		h[3] = bswap32((uint)(st[1] >> 32));
		h[4] = bswap32((uint)(st[2] & 0xffffffffUL));
		h[5] = bswap32((uint)(st[2] >> 32));
		h[6] = bswap32((uint)(st[3] & 0xffffffffUL));
		h[7] = bswap32((uint)(st[3] >> 32));

		if (hash_lt_target(h, target)) {
			uint prior = atomic_cmpxchg((volatile __global uint *)&result[0], 0u, 1u);
			if (prior == 0u) {
				result[1] = n_lo;
				result[2] = n_hi;
				result[3] = 0u;
				for (uint i = 0; i < 8; i++) {
					result[4 + i] = h[i];
				}
			}
			return;
		}
	}
}
`
