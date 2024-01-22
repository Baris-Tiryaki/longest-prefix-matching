
#include <stdio.h>
#include "LpmGpuSailL.hpp"

const int KERNEL_BLOCK_SIZE=128;

__device__ inline bool GetBitCuda(const void* __restrict__ ptr, int bit_index){
    const uint8_t* __restrict__ p = (uint8_t*) ptr;
    int byte_number = bit_index/8;
    int bit_number = bit_index%8;
    uint8_t value = p[byte_number];
    return ((value >> bit_number) & 1) ;
}

__global__ void cuda_lookup(
        const uint32_t* __restrict__ addresses, int32_t* __restrict__ next_hops,
        const uint64_t* __restrict__ B16_ptr, const uint16_t* __restrict__ CN16_ptr, const int32_t* __restrict__ CN24_ptr, const uint16_t* __restrict__ N32_ptr,
        int num_entries) {
    int idx = blockIdx.x * KERNEL_BLOCK_SIZE + threadIdx.x;  // Calculate global index
    if(idx>=num_entries) return;

    uint32_t adr = addresses[idx];
    int32_t next_hop = -1;
    if( GetBitCuda( B16_ptr, adr>>16) == 0 ){
        next_hop=CN16_ptr[adr>>16];
    }
    else{
        int l2_index = (CN16_ptr[adr>>16]<<8) + (adr<<16>>24);
        int32_t value = CN24_ptr[l2_index];
        if(value > -16){ //value is next hop 
            next_hop = value;
        }
        else{ //value is a chunk id
            int l3_index = ((-(value+16))<<8) + (adr & 0xff);
            next_hop = N32_ptr[l3_index];
        }
    }
    next_hops[idx] = next_hop;
}

void LpmGpuSailL::Lookup(IPv4ADR *addresses, IFACEID *next_hops, int num_entries) {
    // *addresses and *next_hops are page_locked memory
    // They are accessible by both CPU and GPU
    int num_blocks = (num_entries+KERNEL_BLOCK_SIZE-1)/KERNEL_BLOCK_SIZE;
    //Kernel launch
    cuda_lookup<<<num_blocks, KERNEL_BLOCK_SIZE>>>(
        (uint32_t*)addresses, (int32_t*)next_hops,
        (uint64_t*)B16_ptr, (uint16_t*)CN16_ptr, (int32_t*)CN24_ptr, (uint16_t*)N32_ptr,
        num_entries);
    cudaDeviceSynchronize();
}