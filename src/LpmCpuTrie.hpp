#pragma once


//Trie based IP_Lookup algorithm
class LpmCpuTrie : public LongestPrefixMatcher{
public:
    LpmCpuTrie(){

    }

    int AddEntry(IPv4ADR ip, int pref_length, IFACEID next_hop) override {
        TrieNode **current_node = &root;
        uint32_t match_bit = 0x80000000;
        uint32_t match_mask = 0x00000000;
        for(int i=0; i<pref_length; i++){
            CreateNodeIfNull(current_node, ip & match_mask, i, -1);
            if( ip & match_bit ){
                current_node = &((*current_node)->right_);
            }
            else{
                current_node = &((*current_node)->left_);
            }
            match_mask |= match_bit;
            match_bit = match_bit >> 1;
        }
        return CreateNodeIfNull(current_node, ip & match_mask, pref_length, next_hop);
    }
    int RemoveEntry(IPv4ADR ip, int pref_length) override {
        return 0;

    }
    void Lookup(IPv4ADR *addresses, IFACEID *next_hops, int num_entries) override {
        for(int k=0; k<num_entries; k++){
            IPv4ADR ip = addresses[k];
            IFACEID longest_matching_hop=-1;

            int current_pref_length = 0;
            TrieNode* current_node = root;
            uint32_t match_bit = 0x80000000;
            while(current_node != nullptr){
                if(current_node->next_hop_ != -1){
                    longest_matching_hop = current_node->next_hop_;
                }
                if(ip & match_bit){
                    current_node = current_node->right_;
                }
                else{
                    current_node = current_node->left_;
                }
                match_bit = match_bit>>1;
                current_pref_length++;

            }

            next_hops[k] = longest_matching_hop;
        }
    }

private:
    class TrieNode{
    public:
        IPv4ADR ip_;
        int pref_length_;
        IFACEID next_hop_=-1;
        TrieNode* left_=nullptr, *right_=nullptr;
        TrieNode(IPv4ADR ip, int pref_length, IFACEID next_hop) :
                     ip_(ip), pref_length_(pref_length), next_hop_(next_hop) {

        }
    };

    TrieNode *root=nullptr;

    int CreateNodeIfNull(TrieNode **node,  IPv4ADR ip, int pref_length, IFACEID next_hop){
        if(*node == nullptr){
            *node = new TrieNode(ip, pref_length, next_hop);
            return 0;
        }
        else{
            if(next_hop != -1){
                if( (*node)->next_hop_ == -1){
                    (*node)->next_hop_ =next_hop;
                    return 0;
                }
                else{ //Trying to override the next hop of an existing trie-node
                    return -1;
                }
            }
            else{
                return 0;
            }
        }
    }
    
};
