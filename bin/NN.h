//
// Created by mengqy on 2019/1/28.
//

#ifndef FASTAI_NN_H
#define FASTAI_NN_H

#include <string>
#include <map>
#include <queue>
#include <set>
#include <thread>
#include "node.h"
#include "rand.h"
#include "global_ops.h"
#include "damp.h"
#include "optimizer.h"

using namespace std;


void topological_sort_forward_graph(vector<variable*> &forward_graph){
    map<string, int> in_degrees;
    map<string, set<string>> in_degree_nodes;

    for (variable *n: forward_graph){
        for (node *input: n->inputs){
            ++in_degrees[n->name];
            in_degree_nodes[input->name].insert(n->name);
        }
    }

    queue<variable*> compute_nodes;

    for (auto iter = forward_graph.begin(); iter != forward_graph.end(); ++iter){
        if (in_degrees[(*iter)->name] == 0)
            compute_nodes.push(*iter);
    }

    vector<variable*> res;
    size_t size, i;
    variable *__current_node;
    while(!compute_nodes.empty()){
        size = compute_nodes.size();
        for (i = 0; i < size; ++i){
            __current_node = compute_nodes.front();
            compute_nodes.pop();
            res.push_back(__current_node);
            for (auto e: in_degree_nodes[__current_node->name]){
                --in_degrees[e];
                if (in_degrees[e] == 0) compute_nodes.push(variable::global_variables[e]);
            }
        }
    }

    if (DEBUG){
        std::cout << res.size() << " nodes\n";
        for(auto e: res)
            std::cout << e->to_str() << std::endl;
    }

    forward_graph = res;
}

void init_grad_table(vector<variable*> &forward_graph){
    for (variable *n: forward_graph){
        if (grad_tables.count(partial_derivative_name(n->name)) == 1){
            tensor<real> *grad = grad_tables[partial_derivative_name(n->name)];
            memset(grad->data(), 0, grad->size());
        }
    }
}

void evaluate(){

}

void train_thread(Adam* opt, int tid){

    vector<variable*> forward_graph = variable::forward_graph[tid];

    //前向传播
    for (variable *n: forward_graph)
        n->forward();

    // 清零
    init_grad_table(forward_graph);

    //后向传播
    for (auto iter = forward_graph.end() - 1; iter != forward_graph.begin() - 1; --iter)
        (*iter)->back_prop();


    //梯度更新
    for (variable *n: forward_graph){
        if (grad_tables.count(partial_derivative_name(n->name)) == 1){
            opt->optimize(n, partial_derivative_name(n->name));
        }
    }

    ++global_steps;
    if (global_steps % print_steps == 0) {
        variable *loss = get_variable(get_name("loss", 0, tid), {1}, tid);
        real *total_loss = loss->get().data();
        logger.info(std::to_string(current_epoch) + " epoch - total loss - " + std::to_string(*total_loss) + " - " + std::to_string(print_steps) +" steps - " + std::to_string(logger.get_diff_time()) + " s");
    }

    if (global_steps % evaluate_steps == 0){
        evaluate();
    }
}

class NN{

public:
    string model_name;
    vector<variable*> forward_compute_graph;

    vector<thread*> threads;

    AdaGrad ada_grad;
    RMSProp rms_prop;
    NesterovRMSProp nesterov_rms_prop;
    Adam adam;

    scoredamp sd;

public:
    NN(){}

    NN(int thread_num):threads(thread_num, 0){
        print_hyper_params();
    }

    void construct_forward_graph(){
        for (vector<variable*> graph: variable::forward_graph)
            topological_sort_forward_graph(graph);

//        for (vector<variable*> graph: variable::forward_graph){
//            std::cout << graph.size() << " nodes\n";
//            for (variable* n: graph){
//                std::cout << n->to_str() << std::endl;
//            }
//
//            std::cout << "----------------------------\n";
//        }
    }



//    void run(){
//        if (global_steps == 0) start = clock();
//
//        for (variable *n: forward_compute_graph)
//            n->forward();
//
//        init_grad_table();
//
//        for (auto iter = forward_compute_graph.end() - 1; iter != forward_compute_graph.begin() - 1; --iter)
//            (*iter)->back_prop();
//
//        for (variable *n: forward_compute_graph){
//            if (grad_tables.count(partial_derivative_name(n->name)) == 1){
////                ada_grad.optimize(n, partial_derivative_name(n->name));
////                rms_prop.optimize(n, partial_derivative_name(n->name));
////                nesterov_rms_prop.optimize(n, partial_derivative_name(n->name));
//                adam.optimize(n, partial_derivative_name(n->name));
////                tensor<real> lr({1});
////                real *t = lr.data();
////                *t = sd.get_learning_rate() / global_batch_size;
////                tops::multiply_prefix(*grad_tables[partial_derivative_name(n->name)], lr, *grad_tables[partial_derivative_name(n->name)]);
////                tops::add(n->get(), *grad_tables[partial_derivative_name(n->name)],  n->get());
//            }
//        }
//
//        ++global_steps;
//
//        if (global_steps % print_steps == 0) {
//            end = clock();
//            logger.info(std::to_string(current_epoch) + " epoch - total loss - " + std::to_string(get_loss()) + " - " + std::to_string(print_steps) +" steps - " + std::to_string((end - start) / CLOCKS_PER_SEC) + " s");
//            start = end;
//        }
//    }

    variable* construct_linear_factor(int batch_size,
                                      int input_dim,
                                      int output_dim,
                                      string input_name,
                                      string weight_name,
                                      string bias_name="",
                                      int tid=0,
                                      bool input_add_grad=true){
        variable *__p_batch_x = get_variable(input_name, {batch_size, input_dim}, tid, input_add_grad);

        variable *__p_weight = get_variable(weight_name, {output_dim, input_dim}, tid);
        uniform(__p_weight->get(), -sqrt(6.0 / (input_dim + output_dim)), sqrt(6.0 / (input_dim + output_dim)));

        variable *__p_weight_batch_x = get_variable(weight_name + input_name , {batch_size, output_dim}, tid);

        glop::dot_multiply(__p_batch_x, __p_weight, __p_weight_batch_x, true);

        if (bias_name == "") return __p_weight_batch_x;

        variable *__p_bias = get_variable(bias_name, {output_dim}, tid);

        variable *__p_weight_batch_x_bias = get_variable(weight_name + input_name + bias_name, {batch_size, output_dim}, tid);

        glop::add_suffix(__p_weight_batch_x, __p_bias, __p_weight_batch_x_bias);

        return __p_weight_batch_x_bias;
    }

    variable* construct_hidden_level(int batch_size,
                                     int input_dim,
                                     int hidden_dim,
                                     string input_name,
                                     string weight_name,
                                     string bias_name,
                                     string activate_fn,
                                     string activate_name,
                                     int tid=0,
                                     bool input_add_grad=true,
                                     bool BN=true){

        variable * __p_weight_batch_x_bias = construct_linear_factor(batch_size, input_dim, hidden_dim, input_name, weight_name, bias_name, tid, input_add_grad);

        variable *__p_activate = get_variable(activate_name, {batch_size, hidden_dim}, tid);

        if (BN){
            string name = __p_weight_batch_x_bias->name;
            variable *__p_mean = get_variable("norm_mean_" + name, {hidden_dim}, tid, false);
            variable *__p_var = get_variable("norm_var_" + name, {hidden_dim}, tid, false);
            variable *__p_norm = get_variable("norm_" + name, {batch_size, hidden_dim}, tid);
            variable *__p_scale = get_variable("scale_" + name, {hidden_dim}, tid);
            uniform(__p_scale->get(), -sqrt(6/(input_dim+hidden_dim)), sqrt(6/(input_dim+hidden_dim)));

            variable *__p_bias = get_variable("bias_" + name, {hidden_dim}, tid);

            variable *__p_direct = get_variable("direct_" + name, {batch_size, hidden_dim}, tid);

            glop::batch_norm(__p_weight_batch_x_bias, __p_mean, __p_var, __p_norm);
            glop::rescale(__p_norm, __p_scale, __p_bias, __p_direct);

            if (activate_fn == "sigmoid"){
                glop::sigmoid(__p_direct, __p_activate);
            } else if (activate_fn == "tanh"){
                glop::tanh(__p_direct, __p_activate);;
            }

        }else{
            if (activate_fn == "sigmoid"){
                glop::sigmoid(__p_weight_batch_x_bias, __p_activate);
            } else if (activate_fn == "tanh"){
                glop::tanh(__p_weight_batch_x_bias, __p_activate);;
            }
        }


        return __p_activate;
    }

    void iteration(){

        for (int i = 0; i < threads.size(); ++i){
            threads[i] = new thread(train_thread, &adam, i);
        }

        for (int i = 0; i < threads.size(); ++i){
            threads[i]->join();
        }

        for (int i = 0; i < threads.size(); ++i){
            delete threads[i];
        }
    }

    real evaluate(){
        return 0.0;
    }
};

#endif //FASTAI_NN_H
