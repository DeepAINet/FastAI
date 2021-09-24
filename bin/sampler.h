//
// Created by mengqy on 2019/1/22.
//

#ifndef FASTAI_SAMPLER_H
#define FASTAI_SAMPLER_H

#include <cmath>
#include "common.h"
#include "dictionary.h"

#define NEGATIVE_TABLE_SIZE 100000000

class EnergySampler{
    LONG *__table = 0;
    unsigned long long __random = 1;
    LONG __dict_size = 0;
public:
    EnergySampler(){}

    ~EnergySampler(){
        if (0 != __table) free(__table);
    }

    void create(dictionary& d){
        if (d.size() <= 0) return;
        __dict_size = d.size();

        __table = (LONG *)malloc(sizeof(LONG) * NEGATIVE_TABLE_SIZE);
        if (0 == __table){
            logger.error("negative_table allocate memory failure.\n");
            exit(EXIT_FAILURE);
        }

        long double __total_pow = 0.0;
        for (LONG i = 0; i < __dict_size; ++i){
            __total_pow += pow(d.count(i), 0.75);
        }

        LONG __idx = 0;
        double __accumulate_pow_rate = (double)(pow(d.count(__idx), 0.75) / __total_pow);
        for (LONG i = 0; i < NEGATIVE_TABLE_SIZE; ++i){
            if ((i+1) / (double)NEGATIVE_TABLE_SIZE > __accumulate_pow_rate){
                ++__idx;
                __accumulate_pow_rate += (double)(pow(d.count(__idx), 0.75) / __total_pow);
            }
            __table[i] = __idx;
        }
    }

    LONG energy_distribute_candidate_sampler(LONG word){
        LONG num = 0;
        while(1){
            __random = __random * (unsigned long long)25214903917 + 11;
            num = __table[(__random >> 16) % NEGATIVE_TABLE_SIZE];
            if (num != word) break;
        }
        return num;
    }

    void energy_distribute_candidate_sampler(LONG word, int num_sample, vector<LONG>& samples){
        LONG num = 0;
        while(1){
            __random = __random * (unsigned long long)25214903917 + 11;
            num = __table[(__random >> 16) % NEGATIVE_TABLE_SIZE];
            if (num != word) break;
        }
    }
};

#endif //FASTAI_SAMPLER_H
