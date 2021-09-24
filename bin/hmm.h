//
// Created by mengqy on 2018/12/13.
//

#ifndef FASTAI_HMM_H
#define FASTAI_HMM_H

#include <map>
#include "common.h"
#include "corn.h"

using namespace std;
using namespace lc;
class HMM{
public:
    lc::LabelCorpus *data = nullptr;
    vector<real> init_state_matrix;
    map<LONG, STATE> idx_states_map;
    map<LONG, map<LONG, real>> state_transfer_matrix;
    map<LONG, map<LONG, real>> observation_matrix;

public:
    HMM(){}

    ~HMM(){}

    void set_idx_states_map(){
        for (LONG i = 0; i < data->label_size(); ++i){
            STATE state_;
            state_.idx = i;
            state_.count = 0;
            state_.init_count = 0;
            idx_states_map[i] = state_;
        }
    }

    void count(){
        LONG state_idx;
        LONG obs_idx;
        for (LONG i = 0; i != data->corpus.size(); ++i)
            for (LONG j = 0; j != data->corpus[i].size(); ++j) {
                state_idx = data->labels[(*data)[i][j].label];
                obs_idx = data->words[(*data)[i][j].word];

                // 状态总计数
                idx_states_map[state_idx].count += 1;
                // 该状态下的观测计数
                observation_matrix[state_idx][obs_idx] += 1;

                // 初始状态计数
                if (j == 0){
                    idx_states_map[state_idx].init_count += 1;
                }

                // 状态转移计数
                if (j != data->corpus[i].size() - 1){
                    state_transfer_matrix[state_idx][data->labels[data->corpus[i][j+1].label]] += 1;
                }
            }
    }

    void supervise_learn(lc::LabelCorpus *label){
        time_t start, end;
        data = label;

        time(&start);
        set_idx_states_map();
        count();
        init_state_matrix.resize(idx_states_map.size());
        float temp;
        for (auto iter = idx_states_map.begin(); iter != idx_states_map.end(); ++iter){
            temp = iter->second.init_count / (real)data->corpus.size();
            init_state_matrix[iter->second.idx] = temp;
        }

        for (auto subject = state_transfer_matrix.begin(); subject != state_transfer_matrix.end(); ++subject){
            for (auto object = subject->second.begin(); object != subject->second.end(); ++object){
                object->second /= idx_states_map[subject->first].count;
            }
        }

        for (auto state_obs = observation_matrix.begin(); state_obs != observation_matrix.end(); ++state_obs){
            temp = 0;
            for (auto obs = state_obs->second.begin(); obs != state_obs->second.end(); ++obs){
                obs->second /= idx_states_map[state_obs->first].count;
                temp += obs->second;
            }
            observation_matrix[state_obs->first][-1] = temp / (real)observation_matrix[state_obs->first].size();
        }

        time(&end);
        double cost = difftime(end, start);
        printf("%f s", cost);
        logger.info("supervise learn successfully.");
    }

    void predict(vector<string>& sentence){
        if (sentence.empty()) return;
        vector<LONG> ids(sentence.size(), -1);
        for (int i = 0; i < sentence.size(); ++i) {
            ids[i] = data->words[sentence[i]];
        }

        vector<vector<real>> max_prob(idx_states_map.size(), vector<real>(ids.size(),0.0f));
        vector<vector<real>> last_state_bayes(idx_states_map.size(), vector<real>(ids.size(),0.0f));
        real prob = 0;
        int last_optimal_state = -1;
        real bayes, temp;

        for (int32_t t = 0; t < ids.size(); ++t){
            for (int32_t current_state = 0; current_state < idx_states_map.size(); ++current_state) {
                if (t == 0){
                    prob = init_state_matrix[current_state] * observation_matrix[current_state][ids[0]];
                    max_prob[current_state][0] = prob;
                    last_state_bayes[current_state][0] = current_state;
                    continue;
                }
                bayes = -MAXFLOAT;
                for (int32_t last_state = 0; last_state < idx_states_map.size(); ++last_state){
                    temp = state_transfer_matrix[last_state][current_state] * max_prob[last_state][t-1];
                    if (bayes < temp){
                        bayes = temp;
                        last_optimal_state = last_state;
                    }
                }
                bayes *= observation_matrix[current_state][ids[t]];
                max_prob[current_state][t] = bayes;
                last_state_bayes[current_state][t] = last_optimal_state;
            }
        }

        bayes = -MAXFLOAT;
        last_optimal_state = 0;
        for (int i = 0; i < idx_states_map.size(); ++i){
            if (bayes < max_prob[ids.size()-1][i]){
                bayes = max_prob[ids.size()-1][i];
                last_optimal_state = i;
            }
        }

        ids[ids.size()-1] = last_optimal_state;
        for (int i = ids.size()-1; i >= 1; --i){
            ids[i-1] = last_state_bayes[ids[i]][i];
        }

        for (LONG i = 0; i < sentence.size(); ++i)
            sentence[i] = sentence[i] + '/' + data->labels[ids[i]];

    }

    void predict(const char *input_file, const char *output_file){
        ifstream in(input_file);
        ofstream out(output_file);
        string temp;
        vector<string> v;
        while(getline(in, temp)){
            v = split(temp);
//            predict(v);
            for (string &term: v){
                out << term << ' ';
            }
            out << std::endl;
            v.clear();
        }
        in.close();
        out.close();
    }

//    float _forward(vector<LINT>& obs_values, matrix& forward_prob){
//        float temp;
//        for (int32_t t = 0; t < obs_values.size(); ++t){
//            for (LINT __current_state = 0; __current_state < idx_states_map.size(); ++__current_state) {
//                if (t == 0){
//                    temp = init_state_matrix.get(0, __current_state) * observation_matrix[__current_state][obs_values[t]];
//                    forward_prob.set(__current_state, t, temp);
//                    continue;
//                }
//
//                temp = 0;
//                for (LINT __last_state = 0; __last_state < idx_states_map.size(); ++__last_state){
//                    temp += forward_prob.get(__last_state, t-1) * state_transfer_matrix[__last_state][__current_state] * observation_matrix[__current_state][obs_values[t]];
//                }
//
//                forward_prob.set(__current_state, t, temp);
//            }
//        }
//
//        float prob = 0.0;
//        for (int32_t i = 0; i < idx_states_map.size(); ++i){
//            prob += forward_prob.get(i, forward_prob.col-1);
//        }
//        return prob;
//
//    }

//    float forward(vector<string>& sentence){
//        // 观测数据
//        vector<LINT> obs_values(sentence.size(), -1);
//        for (int i = 0; i < sentence.size(); ++i) {
//            obs_values[i] = data->word_dict[sentence[i]];
//        }
//
//        //前向概率矩阵
//        matrix forward_prob((int32_t)idx_states_map.size(), (int32_t)sentence.size());
//
//
//        return _forward(obs_values, forward_prob);
//    }

//    float _backward(vector<LINT>& obs_values, matrix& backward_prob){
//        float temp;
//        for (LINT t = obs_values.size()-2; t >= 0; --t){
//            for (LINT __current_state = 0; __current_state < idx_states_map.size(); ++__current_state){
//                temp = 0;
//                for (LINT __next_state = 0; __next_state < idx_states_map.size(); ++__next_state){
//                    temp += state_transfer_matrix[__current_state][__next_state] * observation_matrix[__next_state][obs_values[t+1]] * backward_prob.get(__next_state, t+1);
//                }
//                backward_prob.set(__current_state, t, temp);
//            }
//        }
//
//        temp = 0.0;
//        for (LINT i = 0; i < idx_states_map.size(); ++i){
//            temp += init_state_matrix.get(0, i) * observation_matrix[i][obs_values[0]] * backward_prob.get(i, 0);
//        }
//
//        return temp;
//    }

//    float backward(vector<string>& sentence){
//        vector<LINT> obs_values(sentence.size(), -1);
//        for (int i = 0; i < sentence.size(); ++i){
//            obs_values[i] = data->word_dict[sentence[i]];
//        }
//
//        matrix backward_prob((int32_t)idx_states_map.size(), (int32_t)sentence.size(), 1.0);
//        return _backward(obs_values, backward_prob);
//    }

};

#endif //FASTAI_HMM_H
