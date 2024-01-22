#include <iostream>
#include <stdio.h>

#include <random>
#include <chrono>
#include <vector>

#include "utilities.hpp"
#include "LongestPrefixMatcher.hpp"
#include "LpmCpuVector.hpp"
#include "LpmCpuTrie.hpp"
#include "LpmCpuSailL.hpp"
#include "LpmGpuSailL.hpp"
#include "RoutingExperiment.hpp"

using namespace std;
using namespace std::chrono;

std::mt19937 gen;

void GenerateIpAddresses(RoutingExperiment &exp, int N){
    //std::uniform_int_distribution<> distrib(0, 999);
    for(int i=0; i<N; i++){
        exp.AppendIP( gen() );
    }
}
double GenerateFibEntries(RoutingExperiment &exp, int N, double p=0.573, IFACEID hop_start=0){
    int long_prefix_count=0;
    std::binomial_distribution<> d(32, p);
    exp.AddEntry( 0x00000000, 0, hop_start); //Default route
    for(int i=1; i<N; i++){
        IPv4ADR adr=gen();
        int pref_length = d(gen);
        int mask = len_2_mask(pref_length);
        IFACEID next_hop = ((uint32_t) (hop_start+i)) & (0xFFFF);
        int res= exp.AddEntry( adr & mask, pref_length, next_hop );
        if(res != 0){
            //cout<<"Could not add entry"<<endl;
            i--;
        }
        else{
            if(pref_length>24){
                long_prefix_count++;
            }
        }
    }
    return (double)long_prefix_count/N;
}

void SetSeed(int seed){
    std::random_device rd; 
    if(seed==-1){
        seed = rd();
    }
    cout<<"Seed: "<<seed<<endl;
    gen = std::mt19937(seed);
}

int main(){
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    std::cout << "Cuda Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Constant memory size: " << prop.totalConstMem << " bytes" << std::endl;


    SetSeed(1240);

    RoutingExperiment exp;
    exp.AppendIPLookup(new LpmCpuVector());
    exp.AppendIPLookup(new LpmCpuTrie());
    exp.AppendIPLookup(new LpmCpuSailL());
    exp.AppendIPLookup(new LpmGpuSailL());

    //exp.AddEntry( ipv4(123,45,67,89), 32, 7777);
    double long_prefix_percentage = GenerateFibEntries(exp, 32);
    printf("FIB table completed. %5.2f%%\n", long_prefix_percentage*100);

    //for(int i=0; i<1024*1024*32; i++) exp.AppendIP(ipv4(123,45,67,89));
    GenerateIpAddresses(exp, 1024*1024*64);
    cout<<"IP addresses generated" <<endl;

    exp.PrepareForLookup();
    for(int i=0; i<50; i++){
        exp.CalculateHopsForIPLookups();
        cout<<"Validation returns: " << exp.Validate() << endl;
    }

}
