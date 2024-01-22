#pragma once

#include <vector>
#include <algorithm>
#include <chrono>
#include <typeinfo>

using namespace std;
using namespace std::chrono;

#include "FibEntry.hpp"

// Longest prefix matcher interface
class LongestPrefixMatcher {
public:
    // Add a FIB entry. On success returns 0, on failure returns -1
    virtual int AddEntry(IPv4ADR ip, int pref_length, IFACEID next_hop) = 0;

    // Remove a FIB entry
    virtual int RemoveEntry(IPv4ADR ip, int pref_length) = 0;

    virtual void Lookup(IPv4ADR *addresses, IFACEID *next_hops, int num_entries) = 0;

    virtual void PrepareForLookup() {

    }

    IFACEID Lookup(IPv4ADR address) {
        IFACEID next_hop;
        Lookup(&address, &next_hop, 1);
        return next_hop;
    }
    virtual ~LongestPrefixMatcher(){

    }
};

