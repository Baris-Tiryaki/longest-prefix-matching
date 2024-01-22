#pragma once

// SAIL-L implementation on cpu
class LpmCpuSailL : public LongestPrefixMatcher{
public:
    class L1Chunk{ // Bits 1-16
    public:
        uint64_t  B[ (1<<16)/64 ];   //Bit array; 0=next_hop, 1=chunk_id
        uint16_t  CN16[1<<16]; //These are next hops or L2 chunk id's
        L1Chunk(){
            for(int i=0; i<((1<<16)/64 ); i++){
                B[i]=0;
            }
            for(int i=0; i<(1<<16); i++){
                CN16[i]=0;
            }
        }

    };

    class L2Chunk{ // Bits 17-24
    public:
        int32_t  CN24[256]; //These are next hops or L3 chunk id's
        L2Chunk(){
            for(int i=0; i<256; i++){
                CN24[i]=0;
            }
        }
    };
    class L3Chunk{ // Bits 25-32
    public:
        uint16_t N32[256]; //These are next hops
        L3Chunk(){
            for(int i=0; i<256; i++){
                N32[i]=0;
            }
        }
    };
    class TrieNode{
    public:
        IPv4ADR ip_;
        int pref_length_;
        IFACEID next_hop_=-1;
        TrieNode* left_=nullptr, *right_=nullptr;
        bool dirty_=true;
        IFACEID effective_hop_ = 0;
        TrieNode(IPv4ADR ip, int pref_length, IFACEID next_hop) :
                     ip_(ip), pref_length_(pref_length), next_hop_(next_hop) {

        }
    };


    TrieNode *root=nullptr;
    L1Chunk* l1;
    vector<L2Chunk> l2s;
    vector<L3Chunk> l3s;

public:
    LpmCpuSailL(){
        l1 = new L1Chunk(); 
    }

    int AddEntry(IPv4ADR ip, int pref_length, IFACEID next_hop) override {
        TrieNode **current_node = &root;
        TrieNode *parent_node = nullptr;
        uint32_t match_bit = 0x80000000;
        uint32_t match_mask = 0x00000000;
        for(int i=0; i<pref_length; i++){
            CreateNodeIfNull(current_node, ip & match_mask, i, -1, parent_node);
            parent_node = *current_node;
            if( ip & match_bit ){
                current_node = &((*current_node)->right_);
            }
            else{
                current_node = &((*current_node)->left_);
            }
            match_mask |= match_bit;
            match_bit = match_bit >> 1;
        }
        int response = CreateNodeIfNull(current_node, ip & match_mask, pref_length, next_hop, parent_node);
        SynchronizeTrieToChunks(root);
        return response;
    }

    int RemoveEntry(IPv4ADR ip, int pref_length) override {
        return 0;
    }
    void Lookup(IPv4ADR *addresses, IFACEID *next_hops, int num_entries) override {
        for(int k=0; k<num_entries; k++){
            IPv4ADR adr = addresses[k];
            IFACEID next_hop = -1;
            if( GetBit( l1->B, adr>>16) == 0 ){
                next_hop=l1->CN16[adr>>16];
            }
            else{
                int l2_index = l1->CN16[adr>>16];
                L2Chunk& l2 = l2s[l2_index];
                int32_t value = l2.CN24[adr<<16>>24];
                if(value > -16){ //value is next hop 
                    next_hop = value;
                }
                else{ //value is a chunk id
                    int l3_index = -(value+16);
                    L3Chunk& l3 = l3s[l3_index];
                    next_hop = l3.N32[adr & 0xff];
                }
            }
            next_hops[k] = next_hop;
        }
    }

private:
    void SynchronizeTrieToChunks(TrieNode* node, IFACEID next_hop_from_up=-1){
        if(node==nullptr || node->dirty_==false) return;
        node->dirty_=false;
        IFACEID next_hop = node->next_hop_;
        if(next_hop == -1){
            next_hop = next_hop_from_up;
        }
        if(node->effective_hop_ != next_hop){
            if(node->left_==nullptr && node->right_==nullptr){ //node is a leaf
                FillChunk(node->ip_, node->pref_length_, next_hop);
            }
            else{
                if(node->left_){
                    node->left_->dirty_=true;
                }
                else{
                    FillChunk(node->ip_, node->pref_length_+1, next_hop);
                }
                if(node->right_){
                    node->right_->dirty_=true;
                }
                else{
                    FillChunk(node->ip_ | (1<<(31-node->pref_length_)), 
                              node->pref_length_+1, next_hop);
                }
            }
            node->effective_hop_ = next_hop;
        }
        SynchronizeTrieToChunks(node->left_, next_hop);
        SynchronizeTrieToChunks(node->right_, next_hop);
    }
    void FillChunk(IPv4ADR adr, int depth, IFACEID next_hop){
        //printf("FillChunk: %s/%d <= %d\n", ip_2_str(adr).c_str(), depth, next_hop);
        uint16_t l1_adr = adr>>16;
        if(depth <=16 ){
            int adr_start = (l1_adr>>(16-depth))<<(16-depth);
            int adr_end = ((l1_adr>>(16-depth))+1)<<(16-depth);
            for(int i=adr_start; i<adr_end; i++){
                l1->CN16[i] = next_hop;
            }
        }
        else{
            int l2_index = l1->CN16[l1_adr];
            if(GetBit(l1->B, l1_adr) == false){
                printf("Error #1000\n");
            }
            L2Chunk &l2 = l2s[l2_index];
            uint8_t l2_adr = adr<<16>>24;

            if(depth <=24){
                int shamt = 8-(depth-16);
                int adr_start = (l2_adr>>shamt) << shamt;
                int adr_end = ((l2_adr>>shamt)+1) << shamt;
                for(int i=adr_start; i<adr_end; i++){
                    l2.CN24[i] = next_hop;
                }
            }
            else{
                int l3_index = l2.CN24[l2_adr];
                if(l3_index>-16){
                    printf("Error #1001\n");
                }
                int l3_chunk_id = -l3_index-16;
                L3Chunk &l3 = l3s[l3_chunk_id];
                uint8_t l3_adr = adr & 0xff;

                int shamt = 8-(depth-24);
                int adr_start = (l3_adr>>shamt) << shamt;
                int adr_end = ((l3_adr>>shamt)+1) << shamt;
                for(int i=adr_start; i<adr_end; i++){
                    l3.N32[i] = next_hop;
                }
            }
        }
    }
    bool GetBit(void* ptr, int bit_index){
        uint8_t *p = (uint8_t*) ptr;
        int byte_number = bit_index/8;
        int bit_number = bit_index%8;
        uint8_t value = p[byte_number];
        return ((value >> bit_number) & 1) ;
    }
    void SetBit(void* ptr, int bit_index, bool set){
        uint8_t *p = (uint8_t*) ptr;
        int byte_number = bit_index/8;
        int bit_number = bit_index%8;
        uint8_t& value = p[byte_number];
        if(set){
            value = value | (1<<bit_number);
        }
        else{
            value = value & (~(1<<bit_number));
        }
    }


    int CreateNodeIfNull(TrieNode **node,  IPv4ADR ip, int pref_length, IFACEID next_hop, TrieNode *parent_node){
        if(*node == nullptr){
            *node = new TrieNode(ip, pref_length, next_hop);
            if(pref_length==17){
                if( GetBit(l1->B, ip>>16) == 0){
                    SetBit(l1->B, ip>>16, 1);
                    l1->CN16[ip>>16] = l2s.size();
                    l2s.push_back(L2Chunk());
                    parent_node->effective_hop_ = 0;
                }
            }
            else if(pref_length==25){
                int l2_index = l1->CN16[ip>>16];
                L2Chunk& l2 = l2s[l2_index];
                uint8_t l2_adr = ip<<16>>24;
                if( l2.CN24[l2_adr] > -16){
                    l2.CN24[l2_adr] = -16 - l3s.size();
                    l3s.push_back(L3Chunk());
                    parent_node->effective_hop_ = 0;
                }
            }
            if(parent_node){
                (*node)->effective_hop_ = parent_node->effective_hop_;
            }
            else{
                (*node)->effective_hop_ = 0;
            }
            return 0;
        }
        else{
            (*node)->dirty_ = true;
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
