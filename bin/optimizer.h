//
// Created by mengqy on 2019/2/25.
//

#ifndef FASTAI_OPTIMIZER_H
#define FASTAI_OPTIMIZER_H

#include <string>
#include "global.h"
#include "tensor_ops.h"
#include "node.h"

using namespace std;

class optimizer{
public:
    virtual void optimize(variable *n, string name) = 0;
};

class AdaGrad: optimizer{
    real delta = 0.0000001f;

public:

    void optimize(variable* n, string name){
        tensor<real> *__grad_square_sum = grad_aux[name];

        tensor<real> *__p_grad = grad_tables[name];

        tops::cut_off(*__p_grad, *__p_grad);

        tensor<real> __tmp;
        tops::multiply(*__p_grad, *__p_grad, __tmp);

        tops::add(*__grad_square_sum, __tmp, *__grad_square_sum);

        tops::sqrt0(*__grad_square_sum, __tmp);

        tops::add(__tmp, delta, __tmp);

        tops::divide(start_learning_rate / global_batch_size, __tmp, __tmp);

        tops::multiply(__tmp, *__p_grad, __tmp);

        tops::add(n->get(), __tmp,  n->get());
    }
};

class RMSProp:optimizer{
public:
    real delta = 0.000001f;
    real p = 0.9;
public:
    void optimize(variable *n, string name){
            tensor<real> *__history_aux = grad_aux[name];

            tensor<real> *__p_grad = grad_tables[name];

            tops::cut_off(*__p_grad, *__p_grad);

            tensor<real> __tmp;
            tops::multiply(*__p_grad, *__p_grad, __tmp);

            tops::multiply(*__history_aux, p, *__history_aux);

            tops::multiply(__tmp, (1 - p), __tmp);

            tops::add(*__history_aux, __tmp, *__history_aux);

            tops::add(*__history_aux, delta, __tmp);

            tops::sqrt0(__tmp, __tmp);

            tops::divide(start_learning_rate / global_batch_size, __tmp, __tmp);

            tops::multiply(__tmp, *__p_grad, __tmp);

            tops::add(n->get(), __tmp,  n->get());
    }
};

class NesterovRMSProp:optimizer{
public:
    real p = 0.9;
    real alpha = 0.9;
    real delta = 0.000001f;
public:
    void optimize(variable *n, string name){
        tensor<real> *__grad_move = grad_moves[name];
        tops::multiply(*__grad_move, alpha, *__grad_move);
        tops::subtract(n->get(), *__grad_move,  n->get());

        tensor<real> *__history_accumulation = grad_aux[name];

        tensor<real> *__p_grad = grad_tables[name];
        tops::cut_off(*__p_grad, *__p_grad);

        tensor<real> __current_grad2;
        tops::multiply(*__p_grad, *__p_grad, __current_grad2);

        tops::multiply(*__history_accumulation, p, *__history_accumulation);

        tops::multiply(__current_grad2, (1 - p), __current_grad2);

        tops::add(*__history_accumulation, __current_grad2, *__history_accumulation);

        tops::add(*__history_accumulation, delta, __current_grad2);

        tops::sqrt0(__current_grad2, __current_grad2);

        tops::divide(start_learning_rate / global_batch_size, __current_grad2, __current_grad2);

        tops::multiply(__current_grad2, *__p_grad, __current_grad2);

        tops::add(*__grad_move, __current_grad2, *__grad_move);

        tops::multiply(*__grad_move, (1+alpha), __current_grad2);

        tops::add(n->get(), __current_grad2,  n->get());
    }
};

class Adam:optimizer{
public:
    real p1 = 0.9;
    real p2 = 0.999;
//    real epsilon = 0.01f;
    real delta = 0.000000001f;

public:
    void optimize(variable *n, string name){
        tensor<real> *__history_grad = grad_aux[name];
        tensor<real> *__history_accumulation = grad_moves[name];
        tensor<real> *__p_grad = grad_tables[name];

        tops::cut_off(*__p_grad, *__p_grad);

        tensor<real> __tmp;
        tops::multiply(*__p_grad, *__p_grad, __tmp);
        tops::multiply(*__history_accumulation, p2, *__history_accumulation);
        tops::multiply(__tmp, 1 - p2, __tmp);
        tops::add(*__history_accumulation, __tmp, *__history_accumulation);

        tops::multiply(*__history_grad, p1, *__history_grad);
        tops::multiply(*__p_grad, 1-p1, *__p_grad);
        tops::add(*__history_grad, *__p_grad, *__history_grad);

        real __p1t = start_learning_rate / ((1 - pow(p1, global_steps+1)) * global_batch_size);
        tops::multiply(*__history_grad, __p1t, *__p_grad);

        real __p2t = 1 / (1-pow(p2, global_steps+1));
        tops::multiply(*__history_accumulation, __p2t, __tmp);


        tops::sqrt0(__tmp, __tmp);
        tops::add(__tmp, delta, __tmp);

        tops::divide(*__p_grad, __tmp, __tmp);

        tops::add(n->get(), __tmp,  n->get());
    }
};

#endif //FASTAI_OPTIMIZER_H
