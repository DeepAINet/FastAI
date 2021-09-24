//
// Created by mengqy on 2019/1/9.
//

#ifndef FASTAI_GLOBAL_OPS_H
#define FASTAI_GLOBAL_OPS_H

#include "node.h"
#include "ops.h"

using namespace ops;

namespace glop{

    void add_suffix(variable *a, variable *b, variable *des){
        des->add_input(a);
        des->add_input(b);
        des->operation = new ops::AddSuffix<real>(&a->get(), &b->get(), &des->get());
    }

    void add(variable *a, variable *b, variable *des){
        des->add_input(a);
        des->add_input(b);
        des->operation = new ops::add<real>(&a->get(), &b->get(), &des->get());
    }

    void dot_multiply(variable *a, variable *b, variable *des, bool transpose){
        des->add_input(a);
        des->add_input(b);
        des->operation = new ops::DotMultiply<real>(&a->get(), &b->get(), &des->get(), transpose);
    }

    void batch_norm(variable *a, variable *mean, variable *var, variable *des){
        des->add_input(a);
        des->operation = new ops::BatchNorm(&a->get(), &mean->get(), &var->get(), &des->get());
    }

    void rescale(variable *a, variable *scale, variable *bias, variable *des){
        des->add_input(a);
        des->add_input(scale);
        des->add_input(bias);
        des->operation = new ops::Redirect(&a->get(), &scale->get(), &bias->get(), &des->get());
    }

    void sigmoid(variable *a, variable *des){
        des->add_input(a);
        des->operation = new ops::sigmoid<real>(&a->get(), &des->get());
    }

    void tanh(variable *a, variable *des){
        des->add_input(a);
        des->operation = new ops::tanh<real>(&a->get(), &des->get());
    }

//    void softmax(variable& a, variable& des){
//        des.inputs.push_back(&a);
//        a.consumers.push_back(&des);
//        des.operation = new ops::softmax<real>(&a.get(), &des.get());
//    }

//    void log(variable *a, variable *des){
//        des->add_input(a);
//        des->operation = new ops::log<real>(&a->get(), &des->get());
//    }

    void softmax_cross_entropy_with_logits(variable *predict, variable *label){
        variable *loss = get_variable("loss", {1});
        loss->add_input(predict);
        loss->add_input(label);
        loss->operation = new ops::SoftmaxCrossEntropy(&predict->get(), &label->get());
    }

    void softmax_cross_entropy_with_logits(variable *predict, variable *label, variable *loss){
        loss->add_input(predict);
        loss->add_input(label);
        loss->operation = new ops::SoftmaxCrossEntropy(&predict->get(), &label->get(), &loss->get());
    }

    void multiply(variable *a, variable *b, variable *des){
        des->add_input(a);
        des->add_input(b);
        des->operation = new ops::multiply<real>(&a->get(), &b->get(), &des->get());
    }
//
//    void subtract(variable& des, variable& a, variable& b){
//            des.inputs.push_back(&a);
//            des.inputs.push_back(&b);
//            a.consumers.push_back(&des);
//            b.consumers.push_back(&des);
//            des.operation = new ops::subtract();
//    }
}

#endif //FASTAI_GLOBAL_OPS_H
