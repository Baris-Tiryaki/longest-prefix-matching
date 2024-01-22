#pragma once
#include "utilities.hpp"


class RoutingExperiment{ //IP addresses with next hop
public:
    //Note: Variables below are readonly
    vector<IPv4ADR> adrs_;

    vector<LongestPrefixMatcher*> lpms_;
    vector<vector<IFACEID>> next_hops_;

    RoutingExperiment(){

    }

    void AppendIP(IPv4ADR adr){
        adrs_.push_back(adr);
        for(unsigned i=0; i<next_hops_.size(); i++){
            next_hops_[i].push_back(-1);
        }
    }

    void AppendIPLookup(LongestPrefixMatcher *lpm){
        lpms_.push_back(lpm);
        next_hops_.push_back(vector<IFACEID>() );
        vector<IFACEID> &next_hops__ = next_hops_.back();
        for(unsigned i=0; i<adrs_.size(); i++){
            next_hops__.push_back(-1);
        }
    }

    int AddEntry(IPv4ADR ip, int pref_length, IFACEID next_hop){
        int failures = 0;
        for(unsigned i=0; i<lpms_.size(); i++){
            failures += lpms_[i]->AddEntry(ip, pref_length, next_hop);
        }
        if((failures != 0) && (failures != (-1 * (int)lpms_.size())) ){
            cout<<"Warning: Entry was added to some lookups, but not all."<<endl;
        }
        return failures;
    }
    void RemoveEntry(IPv4ADR ip, int pref_length){
        for(unsigned i=0; i<lpms_.size(); i++){
            lpms_[i]->RemoveEntry(ip, pref_length);
        } 
    }
    
    void PrepareForLookup(){
        cudaHostRegister( &adrs_[0], adrs_.size() * sizeof(uint32_t), 0 );
        for(unsigned i=0; i<lpms_.size(); i++){
            if( dynamic_cast<LpmGpuSailL*>(lpms_[i]) != nullptr){
                cudaHostRegister( &(next_hops_[i][0]), adrs_.size() * sizeof(int32_t), 0 );
            }
            lpms_[i]->PrepareForLookup();
        }
    }

    void CalculateHopsForIPLookups(){
        for(unsigned i=0; i<lpms_.size(); i++){
            LongestPrefixMatcher* lpm = lpms_[i];
            steady_clock::time_point t1 = steady_clock::now();
            lpm->Lookup( &adrs_[0], &(next_hops_[i][0]), adrs_.size() );
            steady_clock::time_point t2 = steady_clock::now();
            double elapsed = duration_cast<nanoseconds>(t2-t1).count() * 0.000001;
            printf("Elapsed time (%d): %.4fms\n", i, elapsed);
        }
    }

    bool Validate(){
        bool all_valid = true;
        for(unsigned i=0; i<adrs_.size(); i++){
            bool line_valid = true;
            for(unsigned j=1; j<lpms_.size(); j++){
                if( next_hops_[0][i]  !=  next_hops_[j][i] ){
                    line_valid = false;
                    break;
                }
            }
            if( !line_valid ){
                printf("Error at index %3d: IP:%s  next_hops: ", i , ip_2_str(adrs_[i]).c_str() );
                for(unsigned j=0; j<lpms_.size(); j++){
                    printf("%d", next_hops_[j][i] );
                    if( j<lpms_.size()-1 ) printf(", ");
                }
                printf("\n");
            }
            all_valid = all_valid & line_valid;
        }
        return all_valid;
    }
};

