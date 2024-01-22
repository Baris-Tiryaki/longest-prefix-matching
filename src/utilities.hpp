#pragma once

#include <iostream>
#include <chrono>

using IPv4ADR = uint32_t;
using IFACEID = int32_t;
using namespace std;
using namespace std::chrono;

inline IPv4ADR ipv4(uint8_t a, uint8_t b, uint8_t c, uint8_t d){
    return (a<<24) | (b<<16) | (c<<8) | d;
}

inline string ip_2_str(IPv4ADR ip){
    uint8_t a = (ip>>24);
    uint8_t b = (ip>>16);
    uint8_t c = (ip>>8);
    uint8_t d = (ip);
    return to_string(a)+"."+to_string(b)+"."+to_string(c)+"."+to_string(d);
}

inline IPv4ADR len_2_mask(int pref_length){
    return (((uint64_t)1<<pref_length)-1) << (32-pref_length);
}