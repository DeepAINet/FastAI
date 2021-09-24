//
// Created by mengqy on 2019/2/14.
//

#ifndef FASTAI_HOOKS_H
#define FASTAI_HOOKS_H

#include "common.h"
#include "tensor.h"
#include "NN.h"

class EarlyStoppingHook{
public:
    int every_steps = 1000;

    int patience = 5;

    int p = 0;

    real validation_set_error = INT64_MAX;

    int best_train_steps;

public:
    EarlyStoppingHook(){}


    void save_model(ofstream &out, vector<tensor<real>*> &params){

    }

    void evaluate(NN& model,  ofstream &out, vector<tensor<real>*> &params){
        while(p < patience){
            real v = model.evaluate();
            if (v < validation_set_error){
                p = 0;
                best_train_steps = global_steps;
                validation_set_error = v;
            }else ++p;
        }

        save_model(out, params);
    }
};

#endif //FASTAI_HOOKS_H
