//
// Created by mengqy on 2019/4/25.
//

#ifndef FASTAI_CRF_H
#define FASTAI_CRF_H

#include <unordered_map>
#include "corn.h"


namespace crf{
struct Node;
struct Path{
public:
    struct Node * lnode;
    struct Node * rnode;
    real weight;
    real cost;

public:
    Path(Node* lnode, Node* rnode)
    :lnode(lnode), rnode(rnode), weight(0.0f), cost(0.0f){

    }
};

struct Node{
public:
    int idx;
    int obs;
    int state;
    real alpha;
    real beta;
    real cost;
    std::vector<Path*> inputs;
    std::vector<Path*> outputs;

public:
    Node(int idx, int obs, int state)
    :idx(idx), obs(obs), state(state), alpha(0.0f), beta(0.0f), cost(0.0f){

    }

    void forward_compute_alpha(){

    }

    void backward_compute_beta(){

    }

    void compute_expectation(){

    };
};

class FeatureAddition{
public:
    unordered_map<string, int> feature_idx_map;
    unordered_map<int, string> idx_feature_map;
    unordered_map<string, real> inner_feature_weights;
    unordered_map<string, real> state_transfer_weights;
    unordered_map<string, real> state_weights;

    unordered_map<string, real> delta_inner_feature_weights;
    unordered_map<string, real> delta_state_transfer_weights;
    unordered_map<string, real> delta_state_weights;
    lc::FeatureTemplate ft;
    lc::corn *corn_ = 0;
    int fid=0;

public:
    FeatureAddition(){}

    FeatureAddition(string train_file, string template_file){
        read_templates(template_file);
        load_corn_corpus(train_file);
    }

    void read_templates(string template_file){
        ft.read_templates(template_file.c_str());
        logger.info("Load corn feature templates successfully!");
    }

    string get_name(string a, int b){
        return a + "[" + std::to_string(b) + "]";
    }

    void load_corn_corpus(string train_file){
        if (0 != corn_) delete corn_;
        corn_ = new lc::corn(train_file);
        logger.info("Load corn train file successfully!");
    }

    void active_features(){
        vector<lc::NODE> nodes = ft.get_nodes();
        vector<lc::TRANSFER> trans = ft.get_trans();
        int col, row;
        for(LONG i = 0; i < corn_->num_samples; ++i){
            for (int j = 0; j < corn_->sentences[i].m; ++j){
                for (lc::NODE node_: nodes){
                    col = std::get<1>(node_);
                    if (col >= 0 && col < corn_->sentences[i].n-1){
                        string temp = get_name(corn_->sentences[i][j][col], col) + "@" + corn_->sentences[i][j].back();
                        if (feature_idx_map.count(temp) == 0){
                            feature_idx_map[temp] = fid;
                            inner_feature_weights[temp] = 0.0f;
                            idx_feature_map[fid] = temp;
                            ++fid;
                        }
                        corn_->sentences[i].active_feature(temp);
                    }
                }

                for (lc::TRANSFER transfer: trans){
                    string temp = "/";
                    for (tuple<int, int> &t: transfer){
                        row = std::get<0>(t), col = std::get<1>(t);
                        if (row + j >= 0 && row + j < corn_->sentences[i].m && col >= 0 && col < corn_->sentences[i].n-1){
                            temp += get_name(corn_->sentences[i][row+j][col], col) + "@" + corn_->sentences[i][row+j].back() + "/";
                        }
                    }
                    if (feature_idx_map.count(temp) == 0) {
                        feature_idx_map[temp] = fid;
                        inner_feature_weights[temp] = 0.0f;
                        idx_feature_map[fid] = temp;
                        ++fid;
                    }
                    corn_->sentences[i].active_feature(temp);
                }
            }
        }

        if (DEBUG){
            for (auto iter = feature_idx_map.begin(); iter != feature_idx_map.end(); ++iter){
                std::cout << iter->first << ":" << iter->second << std::endl;
            }
            for(LONG i = 0; i < corn_->num_samples; ++i){
                corn_->sentences[i].print_sparse_features();
            }
        }

        logger.info("There have generated " + std::to_string(feature_idx_map.size()) + " features.");
    }

    ~FeatureAddition(){
        if (0 != corn_) delete corn_;
    }
};

class CRF{
public:
    FeatureAddition fa;
    unordered_map<string, real> deltas;
    vector<unordered_map<string, real>> feature_conditional_expectation;
    int fid=0;
public:
    CRF(){}

    void get_loss(){

    }

    void update_parameters(){
//        for (auto param: fa.weights){
//            param.second += deltas[param.first];
//        }
    }

    void train(string train_file, string template_file){
        fa.read_templates(template_file);
        fa.load_corn_corpus(train_file);
        fa.active_features();
//        feature_conditional_expectation.resize(fa.corn_->num_samples);
//        train_thread(0, 0, 6000);


    }

    void set_start(tensor<real>& ts, vector<int>& init_states){
        real *pt = ts.data();
        for (int i = 0; i < init_states.size(); ++i)
            *(pt + init_states[i]) = exp(fa.state_weights[fa.corn_->labels[init_states[i]]]);
    }

    string get_transfer_name(int last_state, int current_state){
        return fa.corn_->labels[last_state] + "/" + fa.corn_->labels[current_state];
    }

    void get_solution_with_newton_method(){
        real total=0.0f;
        for (LONG i = 0; i < fa.corn_->num_samples; ++i){
            total += fa.corn_->sentences[i].total;
        }
        total /= fa.corn_->num_samples;

        real fxy, conditional_fxy, initial_value, old_value, new_value, d;
        for (auto w: fa.state_weights){
            fxy=0.0f, conditional_fxy=0.0f;
            for (LONG i = 0; i < fa.corn_->num_samples; ++i){
                fxy += fa.corn_->sentences[i].state_features[w.first];
                conditional_fxy += feature_conditional_expectation[i][w.first];
            }
            initial_value = std::log(fxy / conditional_fxy) / total;
            new_value = initial_value;
            while(1){
                old_value = fxy;
                d = 0.0f;
                for (LONG i = 0; i < fa.corn_->num_samples; ++i){
                    old_value -= feature_conditional_expectation[i][w.first] * std::exp(new_value * fa.corn_->sentences[i].total);
                    d -= fa.corn_->sentences[i].total * feature_conditional_expectation[i][w.first] * std::exp(new_value * fa.corn_->sentences[i].total);
                }

                real last_value = new_value;
                new_value = new_value - old_value / d;
                std::cout << new_value << " " << last_value << std::endl;
                if (abs(new_value - last_value) < 0.000001f) break;
            }
            std::cout << "------------------\n";
            fa.delta_state_weights[w.first] = new_value;
        }


        for (auto w: fa.state_transfer_weights){
            fxy=0.0f, conditional_fxy=0.0f;
            for (LONG i = 0; i < fa.corn_->num_samples; ++i){
                fxy += fa.corn_->sentences[i].state_transfer_features[w.first];
                conditional_fxy += feature_conditional_expectation[i][w.first];
            }
            initial_value = std::log(fxy / conditional_fxy) / total;
            new_value = initial_value;
            std::cout << "fxy:" << fxy << " initial_value:" << initial_value << std::endl;

            while(1){
                old_value = fxy;
                d = 0.0f;
                for (LONG i = 0; i < fa.corn_->num_samples; ++i){
                    if (feature_conditional_expectation[i].find(w.first) != feature_conditional_expectation[i].end()){
                        old_value -= feature_conditional_expectation[i][w.first] * exp(new_value * fa.corn_->sentences[i].total);
                        d -= fa.corn_->sentences[i].total * feature_conditional_expectation[i][w.first] * exp(new_value * fa.corn_->sentences[i].total);
                    }
                }
                real last_value = new_value;
                new_value = new_value - old_value / d;
                std::cout << new_value << " " << last_value << std::endl;
                if (abs(new_value - last_value) < 0.000001f) break;
            }
            std::cout << "------------------\n";
            fa.delta_state_transfer_weights[w.first] = new_value;
        }

        for (auto w: fa.inner_feature_weights){
            fxy=0.0f;
            for (LONG i = 0; i < fa.corn_->num_samples; ++i){
                fxy += fa.corn_->sentences[i].state_transfer_features[w.first];
            }
            initial_value = std::log(fxy / conditional_fxy) / total;
            new_value = initial_value;
            std::cout << "fxy:" << fxy << " initial_value:" << initial_value << std::endl;

            while(1){
                old_value = fxy;
                d = 0.0f;
                for (LONG i = 0; i < fa.corn_->num_samples; ++i){
                    if (feature_conditional_expectation[i].find(w.first) != feature_conditional_expectation[i].end()){
                        old_value -= feature_conditional_expectation[i][w.first] * exp(new_value * fa.corn_->sentences[i].total);
                        d -= fa.corn_->sentences[i].total * feature_conditional_expectation[i][w.first] * exp(new_value * fa.corn_->sentences[i].total);
                    }
                }
                real last_value = new_value;
                new_value = new_value - old_value / d;
                std::cout << new_value << " " << last_value << std::endl;
                if (abs(new_value - last_value) < 0.000001f) break;
            }
            std::cout << "------------------\n";
            fa.delta_state_transfer_weights[w.first] = new_value;
        }

    }

//    void set_mid(tensor<real>& ts, LONG sid, int i){
//        assert(i > 0);
//        assert(i < fa.corn_->sentences[sid].words.size());
//        vector<lc::NODE> nodes = fa.ft.get_nodes();
//        vector<lc::TRANSFER> trans = fa.ft.get_trans();
//
//        real node_feature = 0.0f;
//        int col, n = fa.corn_->sentences[sid].n;
//        vector<real> node_features(fa.corn_->num_states, 0.0f);
//        for (int k = 0; k < fa.corn_->num_states; ++k){
//            for (lc::NODE node_: nodes){
//                col = std::get<1>(node_);
//                if (col >= 0 && col < n-1){
//                    string tmp = fa.corn_->sentences[sid][i][col] + "[" + std::to_string(col) + "]" + "@" + fa.corn_->labels[k];
//                    if (fa.state_weights.count(tmp) == 0){
//                        fa.state_weights[tmp] = 0.0f;
//                    }
//                    node_features[k] += fa.state_weights[tmp];
//                }
//            }
//        }


//        real *pt = ts.data();
//        for (int l = 0; l < fa.corn_->num_states; ++l){
//            for (int j = 0; j < fa.corn_->num_states; ++j){
//                *(pt + l * fa.corn_->num_states + j) = exp(fa.state_transfer_weights[get_transfer_name(last_states[i], current_states[j])]);
//            }
//        }
//    }

//    void set_stop(tensor<real>& ts){
//        ts.init(1.0f);
//    }

//    void get_node_expectation(int sid, vector<tensor<real>>& alpha, vector<tensor<real>>& beta){
//        real *pa, *pb;
//        int state;
//        for (int i = 0; i < fa.corn_->sentences[sid].words.size(); ++i){
//            pa = alpha[i].data(), pb = beta[i].data();
//            for(int j = 0; j < fa.corn_->potential_states[fa.corn_->sentences[sid].words[i]].size(); ++j){
//                state = fa.corn_->potential_states[fa.corn_->sentences[sid].words[i]][j];
//                if (*(pa + state) != 0.0f && *(pb + state) != 0.0f){
//                    feature_conditional_expectation[sid][fa.corn_->labels[state]] += ((*(pa + state)) * (*(pb + state)))/fa.corn_->sentences[sid].total_probability;
//                }
//            }
//        }
//    }
//
//    void get_transfer_expectation(int sid, vector<tensor<real>>& alpha, vector<tensor<real>>& beta, vector<tensor<real>>& M){
//        real *pa, *pb, *pt;
//        int last_state, current_state;
//        for (int i = 0; i < fa.corn_->sentences[sid].words.size()-1; ++i){
//            pa = alpha[i].data(), pb = beta[i+1].data();
//            for(int j = 0; j < fa.corn_->potential_states[fa.corn_->sentences[sid].words[i]].size(); ++j){
//                for (int k = 0; k < fa.corn_->potential_states[fa.corn_->sentences[sid].words[i+1]].size(); ++k){
//                    last_state = fa.corn_->potential_states[fa.corn_->sentences[sid].words[i]][j];
//                    current_state = fa.corn_->potential_states[fa.corn_->sentences[sid].words[i+1]][k];
//                    pt = M[i+1].data() + last_state * fa.corn_->num_states + current_state;
//                    if (*(pa + last_state) != 0.0f && *(pb + current_state) != 0.0f && *pt != 0.0f){
//                        feature_conditional_expectation[sid][get_transfer_name(last_state, current_state)] += ((*((pa + last_state))) * (*(pb + current_state) * (*pt)))/fa.corn_->sentences[sid].total_probability;
//                    }
//                }
//
//            }
//        }
//    }

//    void train_thread(int tid, LONG start, LONG ending){
//        vector<tensor<real>> M;
//        /** forward probability **/
//        vector<tensor<real>> alpha;
//        /** backward probability **/
//        vector<tensor<real>> beta;
//        int size=-1;
//        tensor<real> total_probability;
//        for (LONG i = start; i <= ending; ++i){
//            if (size < fa.corn_->sentences[i].m+1){
//                M.resize(fa.corn_->sentences[i].m+1);
//                alpha.resize(fa.corn_->sentences[i].m);
//                beta.resize(fa.corn_->sentences[i].m);
//            }
//            size = fa.corn_->sentences[i].m+1;
//
//            M[0].reshape({1, fa.corn_->num_states});
//            M[0].init(0.0f);
//            M[size-1].reshape({fa.corn_->num_states, 1});
//            M[size-1].init(0.0f);
//            for (int j = 1; j < size-1; ++j){
//                M[j].reshape({fa.corn_->num_states, fa.corn_->num_states});
//                M[j].init(0.0f);
//            }

//            set_start(M[0], fa.corn_->potential_states[fa.corn_->sentences[i].words[0]]);
//            for (int j = 1; j < size-1; ++j){
//                set_mid(M[j], fa.corn_->sentences[i].words, j);
//            }
//            set_stop(M[size-1]);

//            for (int j = 0; j < size-1; ++j){
//                alpha[j].reshape({1, fa.corn_->num_states});
//                beta[size-2-j].reshape({fa.corn_->num_states, 1});
//                if (j == 0){
//                    alpha[0] = M[0];
//                    beta[size-2-j] = M[size-1];
////                    std::cout << alpha[j].show() << std::endl;
////                    std::cout << beta[size-2-j].show() << std::endl;
//                }else{
//                    tops::dot_mul(alpha[j-1], M[j], alpha[j], false, false);
//                    tops::dot_mul(M[size-1-j], beta[size-1-j], beta[size-2-j], false, false);
////                    std::cout << alpha[j].show() << std::endl;
////                    std::cout << beta[size-2-j].show() << std::endl;
//                }
//
//            }

//            tops::dot_mul(alpha.back(), M.back(), total_probability, false, false);
//            real *pt = total_probability.data();
//            fa.corn_->sentences[i].total_probability = *pt;
//
//            get_node_expectation(i, alpha, beta);
//
//            get_transfer_expectation(i, alpha, beta, M);

//            for (auto a: feature_conditional_expectation[i]){
//                std::cout << a.first << ":" << a.second << ", ";
//            }
//            std::cout << endl;
//        }


//        vector<real> test;
//        for (auto w: fa.state_transfer_weights){
//            real t=0.0f;
//            for (LONG i = 0; i < fa.corn_->num_samples; ++i){
//                if (feature_conditional_expectation[i].count(w.first)){
//                    t += feature_conditional_expectation[i][w.first];
////                    std::cout << w.first << ":" << feature_conditional_expectation[i][w.first] << std::endl;
//                }
//            }
//            if (t != 0.0f){
//                test.push_back(t);
//            }
////            if (t == 0.0f) std::cout << w.first << std::endl;
//        }

//        std::cout << fa.state_transfer_weights.size() << ":" << test.size() << std::endl;
//        get_solution_with_newton_method();
//    }



//    void active(LONG i){
//        vector<lc::NODE> nodes = fa.ft.get_nodes();
//        vector<lc::TRANSFER> trans = fa.ft.get_trans();
//        int col, row_bias;
//        vector<int>& words = fa.corn_->sentences[i].words;
//        int n = fa.corn_->sentences[i].n;
//        for (int row = 0; row < words.size(); ++row){
//            vector<int>& labels = fa.corn_->potential_states[words[row]];
//            for (int k = 0; k < labels.size(); ++k){
//                for (lc::NODE node_: nodes){
//                    col = std::get<1>(node_);
//                    if (col >= 0 && col < n-1){
//                        string tmp = fa.corn_->sentences[i][row][col] + std::to_string(col) + fa.corn_->labels[labels[k]];
//                        if (fa.state_weights.count(tmp) == 0){
//                            fa.state_weights[tmp] = 0.0f;
//                        }
//                    }
//                }
//            }
//
//
//            for (lc::TRANSFER transfer: trans){
//                string temp = "/";
//                vector<int> ttt;
//                for (tuple<int, int> &t: transfer){
//                    row_bias = std::get<0>(t), col = std::get<1>(t);
//                    if (row + row_bias < 0){
//                        temp += "START/";
//                    }else if (row + row_bias >= words.size()){
//                        temp += "END/";
//                    }else{
//                        if (col < n-1){
//                            temp += fa.corn_->sentences[i][row + row_bias][col] + "/";
//                            ttt.push_back(row+row_bias);
//                        }
//                    }
//                }
//
//            }
//        }
//    }

      void active(LONG sid){
        lc::sentence *sp = &fa.corn_->sentences[sid];
        // m为句子词汇长度, 每个词汇有num_states个状态
        vector<vector<Node*>> nodes(sp->m, vector<Node*>(fa.corn_->num_states, 0));
        for (int i = 0; i < sp->m; ++i){
            for (int state_idx = 0; state_idx < fa.corn_->num_states; ++state_idx){
                if (i == 0){
                    nodes[i][state_idx] = new Node(i, sp->words[i], state_idx);
                }else{
                    Node *node = new Node(i, sp->words[i], state_idx);
                    for (Node *n: nodes[i-1]){
                        Path *path = new Path(n, node);
                        node->inputs.push_back(path);
                    }
                    nodes[i][state_idx] = node;
                }
            }
        }


        for (size_t i = 0; i < sp->m; ++i){
          for (size_t j = 0; j < fa.corn_->num_states; ++j)
              nodes[i][j]->forward_compute_alpha();
        }

      for (size_t i = sp->m-1; i >= 0; --i){
          for (size_t j = 0; j < fa.corn_->num_states; ++j)
              nodes[i][j]->forward_compute_alpha();
      }



        for (size_t i = 0; i < sp->m; ++i){
            for (size_t j = 0; j < fa.corn_->num_states; ++j)
                if (0 != nodes[i][j]) delete nodes[i][j];
        }
    }

    void train_thread(int tid, int start, int ending){
        lc::sentence *sp = fa.corn_->sentences;
//        LONG size = fa.corn_->num_samples;
        for (LONG i = start; i <= ending; ++i){

        }
    }
};
}

#endif //FASTAI_CRF_H
