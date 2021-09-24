//
// Created by mengqy on 2019/2/12.
//

#ifndef FASTAI_LSTM_H
#define FASTAI_LSTM_H

#include "NN.h"
#include "global.h"
#include "global_ops.h"
#include "corpus.h"

void test(){
    std::cout << "Hello world\n";
}


class LSTM: public NN{
public:

public:

    LSTM(int hidden_dim, int batch_size, int input_dim, int len, int output_dim, int thread_num=1, bool BN=true)
    :NN(thread_num){
        for (int k = 0; k < thread_num; ++k){
            for (int i = 0; i < len; ++i){
                if (i == 0){
                    variable *__pi = construct_hidden_level(batch_size, input_dim, hidden_dim, get_name("X", 0, k), "Wxi",
                                                            "bi", "sigmoid", get_name("I", 0, k), k, false, BN);

                    variable *__pf = construct_hidden_level(batch_size, input_dim, hidden_dim, get_name("X", 0, k), "Wxf",
                                                            "bf", "sigmoid", get_name("F", 0, k), k, false, BN);

                    variable *__po = construct_hidden_level(batch_size, input_dim, hidden_dim, get_name("X", 0, k), "Wxo",
                                                            "bo", "sigmoid", get_name("O", 0, k), k, false, BN);

                    variable *__pc = construct_hidden_level(batch_size, input_dim, hidden_dim, get_name("X", 0, k), "Wxc",
                                                            "bc", "tanh", get_name("cin", 0, k), k, false, BN);

                    variable *__pC = get_variable(get_name("C", 0, k), {batch_size, hidden_dim}, k);
                    glop::multiply(__pi, __pc, __pC);

                    variable *__p_tanhC = get_variable(get_name("tanhC", 0, k), {batch_size, hidden_dim}, k);
                    glop::tanh(__pC, __p_tanhC);

                    variable *__ph = get_variable(get_name("H", 0, k), {batch_size, hidden_dim}, k);
                    glop::multiply(__po, __p_tanhC, __ph);

                    continue;
                }

                variable *__pi = construct_control_gate(batch_size, input_dim, hidden_dim, get_name("X", i, k),
                                                        "Wxi", "bi", get_name("H", i-1, k), "Whi", get_name("Iin", i, k), get_name("I", i, k), "sigmoid", k, BN);

                variable *__pf = construct_control_gate(batch_size, input_dim, hidden_dim, get_name("X", i, k),
                                                        "Wxf", "bf", get_name("H", i-1, k), "Whf", get_name("Fin", i, k), get_name("F", i, k), "sigmoid", k, BN);

                variable *__po = construct_control_gate(batch_size, input_dim, hidden_dim, get_name("X", i, k),
                                                        "Wxo", "bo", get_name("H", i-1, k), "Who", get_name("Oin", i, k), get_name("O", i, k), "sigmoid", k, BN);

                variable *__pc = construct_control_gate(batch_size, input_dim, hidden_dim, get_name("X", i, k),
                                                        "Wxc", "bc", get_name("H", i-1, k), "Whc", get_name("cin", i, k), get_name("c", i, k), "tanh", k, BN);

                variable *__pci = get_variable(get_name("CI", i, k), {batch_size, hidden_dim}, k);
                glop::multiply(__pi, __pc, __pci);

                variable *__pCl = get_variable(get_name("C", i-1, k), {batch_size, hidden_dim}, k);

                variable *__pcf = get_variable(get_name("CF", i, k), {batch_size, hidden_dim}, k);
                glop::multiply(__pf, __pCl, __pcf);

                variable *__pC = get_variable(get_name("C", i, k), {batch_size, hidden_dim}, k);
                glop::add(__pci, __pcf, __pC);

                variable *__p_tanhC = get_variable(get_name("tanhC", i, k), {batch_size, hidden_dim}, k);
                glop::tanh(__pC, __p_tanhC);

                variable *__ph = get_variable(get_name("H", i, k), {batch_size, hidden_dim}, k);
                glop::multiply(__po, __p_tanhC, __ph);
            }

            variable *__p_batch_y = get_variable(get_name("Y", 0, k), {batch_size, output_dim}, k, false);

            variable *__p_vxb = construct_linear_factor(batch_size,
                                                        hidden_dim,
                                                        output_dim,
                                                        get_name("H", len-1, k),
                                                        "V",
                                                        "Vb",
                                                        k);
            variable *__p_loss = get_variable(get_name("loss", 0, k), {batch_size, output_dim}, k, false);

            glop::softmax_cross_entropy_with_logits(__p_vxb, __p_batch_y, __p_loss);
        }

        construct_forward_graph();
    }

    variable *construct_control_gate(int batch_size,
            int input_dim,
            int hidden_dim,
            string input_name,
            string input_weight_name,
            string bias_name,
            string hidden_name,
            string hidden_weight_name,
            string gate_input_name,
            string gate_name,
            string activate_name,
            int tid=0,
            bool BN=true){

        variable *__p_wh = construct_linear_factor(batch_size,
                hidden_dim, hidden_dim, hidden_name, hidden_weight_name, "", tid);

        variable *__p_wxb = construct_linear_factor(batch_size,
                input_dim, hidden_dim, input_name, input_weight_name, bias_name, tid, false);

        variable *__p_wh_wxb = get_variable(gate_input_name, {batch_size, hidden_dim}, tid);

        glop::add(__p_wh, __p_wxb, __p_wh_wxb);

        variable *__p_new_gate = get_variable(gate_name, {batch_size, hidden_dim}, tid);


        if (BN) {
            string name = __p_wh_wxb->name;
            variable *__p_mean = get_variable("norm_mean_" + name, {hidden_dim}, tid, false);
            variable *__p_var = get_variable("norm_var_" + name, {hidden_dim}, tid, false);
            variable *__p_norm = get_variable("norm_" + name, {batch_size, hidden_dim}, tid);
            variable *__p_scale = get_variable("scale_" + name, {hidden_dim}, tid);
            uniform(__p_scale->get(), -sqrt(6/(input_dim+hidden_dim)), sqrt(6/(input_dim+hidden_dim)));
            variable *__p_bias = get_variable("bias_" + name, {hidden_dim}, tid);

            variable *__p_direct = get_variable("direct_" + name, {batch_size, hidden_dim}, tid);

            glop::batch_norm(__p_wh_wxb, __p_mean, __p_var, __p_norm);
            glop::rescale(__p_norm, __p_scale, __p_bias, __p_direct);

            if (activate_name == "sigmoid") {
                glop::sigmoid(__p_direct, __p_new_gate);
            } else if (activate_name == "tanh") {
                glop::tanh(__p_direct, __p_new_gate);;
            }
        } else {

            if (activate_name == "sigmoid")
                glop::sigmoid(__p_wh_wxb, __p_new_gate);

            if (activate_name == "tanh")
                glop::tanh(__p_wh_wxb, __p_new_gate);

        }



        return __p_new_gate;
    }



};

#endif //FASTAI_LSTM_H
