//
// Created by mengqy on 2019/1/31.
//

#ifndef FASTAI_GLOBAL_H
#define FASTAI_GLOBAL_H

#define DEBUG false

#include <string>
#include <map>
#include <memory.h>
#include "tensor.h"
#include "common.h"
#include "tensor_ops.h"

using namespace std;

map<string, tensor<real>*> grad_tables;

map<string, tensor<real>*> grad_aux;

/** learning rate tables **/
map<string, tensor<real>*> epsilon_table;

/** momentum **/
map<string, tensor<real>*> grad_moves;

int global_steps = 0;

int total_steps = 0;

int current_epoch = 0;

int epoch_num = 1;

int every_epoch_num = 1;

int evaluate_steps = 1000;

int print_steps = 10;

int global_batch_size = 64;

real start_learning_rate = 0.01;

real current_learning_rate = 0.001;

tensor<real> total_loss({1});

clock_t start_clock;

clock_t end_clock;

inline void update_total_loss(real loss){
    real *__pt = total_loss.data();
    *__pt = loss;
}

real get_loss(){
    real *__pt = total_loss.data();
    return *__pt;
}

inline void add_to_grad_table(string& name, tensor<real> *grad){
    if (grad_tables.count(name) == 1){
        name = name + " has existed.";
        logger.error(name);
        exit(EXIT_FAILURE);
    }

    if (0 == grad){
        logger.error("add null pointer!");
        exit(EXIT_FAILURE);
    }

    grad_tables[name] = grad;
}

inline string partial_derivative_name(string& name){
    return "∂(loss)/∂(" + name + ")";
}

inline string get_name(string name, int i=0, int tid=0){
    return name + "(" + std::to_string(tid) + "_" +std::to_string(i) + ")";
}

bool update_grad_table(string name, tensor<real>& grad){
    tensor<real> *__des_grad = grad_tables[name];
    if (__des_grad == 0) {
        string temp = "grad table has no grad named " + name;
        logger.error(temp);
        return false;
    }
    tops::add(*__des_grad, grad, *__des_grad);
    return true;
}

void init_grad_table(){
    for (auto iter = grad_tables.begin(); iter != grad_tables.end(); ++iter){
        memset(iter->second->data(), 0, iter->second->size());
    }
}

void print_hyper_params(){
    std::cout << "start learning rate: " << start_learning_rate
              << "," << "global batch size: " << global_batch_size << std::endl;
}

#endif //FASTAI_GLOBAL_H
