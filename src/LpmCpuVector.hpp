#pragma once


//Simplest IP_Lookup algorithm
class LpmCpuVector : public LongestPrefixMatcher{
public:
    LpmCpuVector(){

    }

    int AddEntry(IPv4ADR ip, int pref_length, IFACEID next_hop) override {
        ip = ip & len_2_mask(pref_length);
        FibEntry entry(ip, pref_length, next_hop);
        int index = searchSimilarEntry(entry);
        if(index == -1){
            entries.push_back(entry);
            return 0;
        }
        else{
            return -1;
        }
    }
    int RemoveEntry(IPv4ADR ip, int pref_length) override {
        FibEntry entry(ip, pref_length, 0);
        int index = searchSimilarEntry(entry);
        if(index == -1){
            return -1;
        }
        else{
            //entries.erase(entries.begin() + index);
            entries[index] = entries[ entries.size()-1 ];
            entries.pop_back();
            return 0;
        }
    }
    void Lookup(IPv4ADR *addresses, IFACEID *next_hops, int num_entries) override {
        for(int k=0; k<num_entries; k++){
            IPv4ADR ip = addresses[k];
            IFACEID longest_matching_hop=-1;
            int longest_prefix_length=-1;
            for(unsigned i=0; i<entries.size(); i++){
                const FibEntry& entry = entries[i]; // FIB table entry
                IPv4ADR mask = len_2_mask(entry.pref_length_);
                if( ((entry.adr_ ^ ip) & mask) == 0 ){ //All bits in the prefix match
                    if(longest_prefix_length<entry.pref_length_){
                        longest_prefix_length = entry.pref_length_;
                        longest_matching_hop = entry.next_hop_;
                    }
                }
            }
            next_hops[k] = longest_matching_hop;
        }
    }

private:
    vector<FibEntry> entries;

    int searchSimilarEntry(const FibEntry& search){
        for(unsigned i=0; i<entries.size(); i++){
            const FibEntry& entry = entries[i];
            if(entry.adr_ == search.adr_ && entry.pref_length_ == search.pref_length_){
                return i;
            }
        }
        return -1;
    }
};
