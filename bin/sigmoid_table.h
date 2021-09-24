//
// Created by mengqy on 2019/1/21.
//

#ifndef FASTAI_SIGMOIDTABLE_H
#define FASTAI_SIGMOIDTABLE_H

#include <cmath>
#include "log.h"

class sigmoid_table {
    int __size = 1000;
    int __limit = 6;
    float *__data=0;

public:
    sigmoid_table(){
        __data = new float[__size];
        if (0 == __data){
            logger.error("allocate memory failure.\n");
            exit(EXIT_FAILURE);
        }
        __construct();
    }

    ~sigmoid_table(){
        if (0 != __data) delete [] __data;
    }

    void __construct();

    float get(float v);

    float operator[](int i);
};

void sigmoid_table::__construct(){
    float t;
    for (int i = 0; i < __size; ++i){
        t = ((i/(float)(__size-1))*2 - 1) * __limit;
        __data[i] = 1 /(1 + exp(-t));
    }
}

float sigmoid_table::get(float v){
    if (v >= __limit) return 1.0;
    if (v <= -__limit) return 0.0;
    double t = (v / (__limit*2) + 0.5) * (__size - 1);
    return __data[(int)(t)];
}

float sigmoid_table::operator[](int i){
    assert(i < 1000 && i >= 0);
    return __data[i];
}


#endif //FASTAI_SIGMOIDTABLE_H
