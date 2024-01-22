#pragma once

#include <stdio.h>
#include "utilities.hpp"
#include "LongestPrefixMatcher.hpp"
#include "LpmCpuSailL.hpp"
#include <cuda_runtime.h>


class LpmGpuSailL : public LongestPrefixMatcher {
    LpmCpuSailL sail;

    char *d_ptr=nullptr; //Device pointer

    //These 4 pointers point to inside of d_ptr
    char *B16_ptr;
    char *CN16_ptr;
    char *CN24_ptr;
    char *N32_ptr;

    int data_size = 0;

public:
    int AddEntry(IPv4ADR ip, int pref_length, IFACEID next_hop) override {
        return sail.AddEntry(ip, pref_length, next_hop);
    }

    int RemoveEntry(IPv4ADR ip, int pref_length) override {
        return sail.RemoveEntry(ip, pref_length);
    }
    void PrepareForLookup() override {
        if(d_ptr) cudaFree(d_ptr);
        int B16_size = RoundUp((1<<16)/64 * sizeof(uint64_t)); // L1.B[ (1<<16)/64 ];
        int CN16_size = RoundUp( (1<<16) * sizeof(uint16_t));    // L1.CN16[1<<16];
        int CN24_size = RoundUp(256 * sizeof(int32_t) * sail.l2s.size()); // L2.CN24[256]
        int N32_size =  RoundUp(256 * sizeof(uint16_t)* sail.l3s.size()); // L3.N32[256];

        data_size = B16_size + CN16_size + CN24_size + N32_size;
        printf("Total FIB data structure size on GPU: %.2f MB\n", data_size/1000000.0);

        //Prepare GPU buffer
        cudaMalloc(&d_ptr, data_size);
        B16_ptr = d_ptr;
        CN16_ptr = d_ptr+B16_size;
        CN24_ptr = CN16_ptr+CN16_size;
        N32_ptr = CN24_ptr+CN24_size;

        //Copy FIB arrays to gpu
        cudaMemcpy(B16_ptr, sail.l1->B, B16_size, cudaMemcpyHostToDevice);
        cudaMemcpy(CN16_ptr, sail.l1->CN16, CN16_size, cudaMemcpyHostToDevice);
        for(unsigned i=0; i<sail.l2s.size(); i++){
            cudaMemcpy(CN24_ptr + 256*sizeof(int32_t)*i, sail.l2s[i].CN24,
                       256 * sizeof(int32_t), cudaMemcpyHostToDevice);
        }
        for(unsigned i=0; i<sail.l3s.size(); i++){
            cudaMemcpy(N32_ptr + 256*sizeof(uint16_t)*i, sail.l3s[i].N32,
                       256 * sizeof(uint16_t), cudaMemcpyHostToDevice);
        }
        

    }
    void Lookup(IPv4ADR *addresses, IFACEID *next_hops, int num_entries) override;

private:
    int RoundUp(int size){
        return (size+127)/128*128; 
    }
};

