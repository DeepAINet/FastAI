//
// Created by mengqy on 2019/1/9.
//

#ifndef FASTAI_NODE_H
#define FASTAI_NODE_H

#include <iostream>
#include <map>
#include <set>
#include <string>
#include "tensor.h"
#include "ops.h"

using namespace std;
using namespace ops;

class node{

public:
    /** the son nodes **/
    vector<node*> consumers;

    /** the parent nodes **/
    vector<node*> inputs;

    /** the name of node **/
    string name;

    /** the operation of node, operation(inputs) = _v  **/
    op* operation = 0;

    /** the tensor of node **/
    tensor<real> __v;

public:
    node(string& name, const shape& sp)
    :name(name), __v(sp){
        __v.name = name;
    }

    node(string& name, vector<int>& sp)
    :name(name), __v(sp){
        __v.name = name;
    }

    ~node(){
        if (0 != operation) delete operation;
    }

    void forward(){
        if (0 == operation) {
            return;
        }
        operation->forward();
    }

    void back_prop(){
        if (0 == operation) {
            return;
        }
        operation->backward();
    }

    tensor<real>& get(){
        return __v;
    }

    void add_input(node *input){
        if (0 == input){
            std::cout <<"input = 0\n";
            return;
        }

        inputs.push_back(input);
        input->add_consumer(this);
    }

private:
    void add_consumer(node *consumer){
        consumers.push_back(consumer);
    }
};


class variable:public node{

public:
    static int idx;

    /** the forward compute graph **/
    static vector<vector<variable *>> forward_graph;

    /** global variables **/
    static map<string, variable *> global_variables;

public:
    variable(const shape& sp, string name="variable-" + std::to_string(idx++)):node(name, sp){}

    variable(vector<int> sp, string name="variable-" + std::to_string(idx++)):node(name, sp){}

    ~variable(){}

    string to_str(){
        string __tmp = __v.to_str();
        __tmp = __tmp.substr(1, __tmp.size()-2);
        string __s = "[name=\'" + name + "\', " + __tmp + ",";

        if (inputs.size() == 0){
            __s += "\ninputs=None,";
        } else {
            __s += "\ninputs={";
            for (node* input: inputs){
                __s += input->name + ",";
            }
            __s += "},";
        }

        if (operation == 0){
            __s += "\top=None,";
        }else{
            __s += "\top=" + operation->name + ",";
        }

        if (consumers.size() == 0){
            __s += "\nconsumers=None,";
        }else{
            __s += "\nconsumers={";
            for (node* consumer: consumers){
                __s += consumer->name + ",";
            }
            __s += "}";
        }
        __s += "]\n";
        return __s;
    }
};

int variable::idx = 0;

vector<vector<variable*>> variable::forward_graph;

map<string, variable*> variable::global_variables;

variable* get_variable(string name, vector<int> sp, int tid=0, bool add_grad=true){
    variable *res = 0;
    if (variable::global_variables.count(name) == 1)
        return variable::global_variables[name];

    res = new variable(sp, name);
    variable::global_variables[name] = res;
    if (tid == variable::forward_graph.size()){
        vector<variable*> __new_forward_graph;
        variable::forward_graph.push_back(__new_forward_graph);
    }
    variable::forward_graph[tid].push_back(res);
    if (add_grad) {
        grad_tables[partial_derivative_name(name)] = new tensor<real>(sp);
        grad_aux[partial_derivative_name(name)] = new tensor<real>(sp);
//        epsilon_table[partial_derivative_name(name)] = new tensor<real>(sp);
        grad_moves[partial_derivative_name(name)] = new tensor<real>(sp);
    }
    return res;
}

#endif //FASTAI_NODE_H
