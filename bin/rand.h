//
// Created by mengqy on 2019/1/8.
//

#ifndef FASTAI_RAND_H
#define FASTAI_RAND_H

#include <random>
#include "tensor.h"

const LONG LONG_RAND_MAX = ((long)RAND_MAX + 2) * (long)RAND_MAX;

void uniform(tensor<float>& des, float low, float high){
    std::default_random_engine random(time(NULL));
    std::uniform_real_distribution<double> dis2(low, high);

    srand((unsigned)time(0));
    float *__data = des.data();
    for (int i = 0; i < des.size(); ++i){
        float value = dis2(random);
        __data[i] = value;
    }
}

template <typename T>
void assign(tensor<T>& des, T *data, LONG size){
    assert(des.size() == size);
    T *__data = des.data();
    for (LONG i = 0; i < des.size(); ++i){
        __data[i] = data[i];
    }
}

template <typename T>
void assign(tensor<T>& des, T t){
    T *__pd = des.data();
    for (LONG i = 0; i < des.size(); ++i){
        __pd[i] = t;
    }
}

LONG get_rand(LONG seed){
    LONG limit = LONG_RAND_MAX - LONG_RAND_MAX % seed;
    LONG rnd;
    while(rnd >= limit){
        rnd = ((LONG)RAND_MAX + 1) * (LONG)rand() + (LONG)rand();
    }
    return rnd % seed;
}

template <typename T>
void shuffle(vector<T> &records){
    if (records.size() <= 2) return;
    T __temp;
    LONG j;
    size_t size = records.size();
    for (LONG i = size - 1; i >= 0; --i){
        j = get_rand(i+1);
        __temp = records[j];
        records[j] = records[i];
        records[i] = __temp;
    }
}

#endif //FASTAI_RAND_H
