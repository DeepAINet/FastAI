//
// Created by mengqy on 2019/2/13.
//

#ifndef FASTAI_DAMP_H
#define FASTAI_DAMP_H

#include "common.h"
#include "global.h"

class damp{
public:
    virtual real get_learning_rate() = 0;
};

class stepdamp: public damp{
public:
    real get_learning_rate(){
        int a = current_epoch / every_epoch_num;
        real res = start_learning_rate * pow(0.5, a);
        if (res <= start_learning_rate * 0.01) return start_learning_rate * 0.01;
        return res;
    }
};

class expdamp:public damp{
public:
    real get_learning_rate(){
        real res = start_learning_rate / std::exp(current_epoch);
        if (res <= start_learning_rate * 0.01) return start_learning_rate * 0.01;
        return res;
    }
};

class scoredamp: public damp{
public:
    real get_learning_rate(){
        real res = start_learning_rate * (1 - global_steps / (real)total_steps);
        if (res <= start_learning_rate * 0.01) return start_learning_rate * 0.01;
        return res;
    }
};
#endif //FASTAI_DAMP_H
