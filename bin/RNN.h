//
// Created by mengqy on 2019/2/1.
//

#ifndef FASTAI_RNN_H
#define FASTAI_RNN_H

#include "NN.h"
#include "node.h"
#include "rand.h"
#include "global_ops.h"

#define DEPTH 28

using namespace glop;

class RNN: public NN{
public:
    RNN(int hidden_dim,
        int len,
        int input_dim,
        int output_dim,
        int batch_size,
        string activate_fn){

        model_name = "RNN model";

        for (int i = 0; i < len; ++i){
            if (0 != i){
                variable *__p_uxb = construct_linear_factor(batch_size, input_dim, hidden_dim,
                                       "X" + std::to_string(i), "U", "Ub", false);

                variable *__p_wh = construct_linear_factor(batch_size, hidden_dim, hidden_dim,
                        "H" + std::to_string(i-1), "W", "", true);

                variable *__p_hidden = get_variable("H" + std::to_string(i), {batch_size, hidden_dim});


                variable *__p_uxb_wh = get_variable("UXUb+WH"+std::to_string(i-1), {batch_size, hidden_dim});
                glop::add(__p_uxb, __p_wh, __p_uxb_wh);
                glop::sigmoid(__p_uxb_wh, __p_hidden);
            }else{
                construct_hidden_level(batch_size,
                        input_dim,
                        hidden_dim,
                        "X" + std::to_string(i),
                        "U",
                        "Ub",
                        activate_fn,
                        "H" + std::to_string(i),
                        false);
            }
        }

        variable *__p_batch_y = get_variable("Y", {batch_size, output_dim}, false);

        variable *__p_vxb = construct_linear_factor(batch_size,
                                hidden_dim,
                                output_dim,
                                "H" + std::to_string(len-1),
                                "V",
                                "Vb");

        glop::softmax_cross_entropy_with_logits(__p_vxb, __p_batch_y);


        construct_forward_graph();

    }

    void iteration(){
//        run();
    }
};

#endif //FASTAI_RNN_H
