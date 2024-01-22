#pragma once

class FibEntry{
public:
    IPv4ADR adr_ = 0;
    int pref_length_ = -1;
    IFACEID next_hop_ = -1;
    FibEntry() = default;
    FibEntry(IPv4ADR adr, int pref_length, IFACEID next_hop) : 
        adr_(adr), pref_length_(pref_length), next_hop_(next_hop) {
    }
};