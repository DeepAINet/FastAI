//
// Created by mengqy on 2019/1/30.
//

#ifndef FASTAI_DNN_H
#define FASTAI_DNN_H

#include "tensor.h"
#include "node.h"
#include "global_ops.h"
#include "global.h"
#include "NN.h"
#include "rand.h"

using namespace glop;

class DNN: public NN{

public:
    DNN(int *a, int len, int input_dim, int output_dim, int batch_size){
        if (len <= 0 || input_dim <= 0 || output_dim <= 0) {
            std::cerr << "len <= 0 || input_dim <= 0 || output_dim <= 0";
            exit(EXIT_FAILURE);
        }

        variable *__p_batch_x = new variable({batch_size, input_dim}, "batch_x");
        variable *__p_batch_y = new variable({batch_size, output_dim}, "batch_y");

        vector<int> s;
        s.push_back(input_dim);
        for (int i = 0; i < len; ++i)
            s.push_back(a[i]);
        s.push_back(output_dim);

        variable *__p_weight, *__p_weight_x, *__p_bias, *__p_weight_x_bias, *__p_sigmoid;

        for (int i = 1; i < s.size(); ++i){

            __p_weight = new variable({s[i], s[i-1]}, "weight" + std::to_string(i));
            uniform(__p_weight->get(), -sqrt(6/(input_dim+output_dim)), sqrt(6/(input_dim+output_dim)));
            grad_tables["∂(loss)/∂(weight" + std::to_string(i) + ")"] = new tensor<real>();

            __p_weight_x = new variable({batch_size, s[i]}, "weight_x" + std::to_string(i));
            grad_tables["∂(loss)/∂(weight_x" + std::to_string(i) + ")"] = new tensor<real>();

            __p_bias = new variable({s[i]}, "bias" + std::to_string(i));
            grad_tables["∂(loss)/∂(bias" + std::to_string(i) + ")"] = new tensor<real>();

            __p_weight_x_bias = new variable({batch_size, s[i]}, "weight_x_bias" + std::to_string(i));
            grad_tables["∂(loss)/∂(weight_x_bias" + std::to_string(i) + ")"] = new tensor<real>();

            if (1 == i) glop::dot_multiply(__p_batch_x, __p_weight, __p_weight_x, true);
            else glop::dot_multiply(variable::global_variables["sigmoid" + std::to_string(i-1)],
                                    __p_weight, __p_weight_x, true);

            glop::add_suffix(__p_weight_x, __p_bias, __p_weight_x_bias);

            if (i == s.size() - 1) continue;

            __p_sigmoid = new variable({batch_size, s[i]}, "sigmoid" + std::to_string(i));
            grad_tables["∂(loss)/∂(sigmoid" + std::to_string(i) + ")"] = new tensor<real>();

            glop::sigmoid(__p_weight_x_bias, __p_sigmoid);

        }

        variable *__p_probability = new variable({batch_size, output_dim}, "probability");
        variable *__p_loss = new variable({1}, "loss");
        softmax_cross_entropy_with_logits(variable::global_variables["weight_x_bias" + std::to_string(s.size()-1)], __p_batch_y);
        grad_tables["∂(loss)/∂(loss)"] = new tensor<real>();

        construct_forward_graph();
    }

    DNN(vector<int> structure, int input_dim, int output_dim, int batch_size){
        for (int num: structure) std::cout << num;
    }

    ~DNN(){
        for (auto iter = variable::global_variables.begin(); iter != variable::global_variables.end(); ++iter){
            if (iter->second != 0) delete iter->second;
        }

        for (auto iter = grad_tables.begin(); iter != grad_tables.end(); ++iter)
            if (iter->second != 0) delete iter->second;
    }

    void iteration(tensor<real>& batch_x, tensor<real>& batch_y){
//        run();
    }
};

#endif //FASTAI_DNN_H
