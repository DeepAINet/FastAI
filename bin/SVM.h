//
// Created by mengqy on 2019/5/13.
//

#ifndef FASTAI_SVM_H
#define FASTAI_SVM_H

#include <time.h>
#include <unordered_set>
#include <unordered_map>
#include "common.h"
#include "corpus.h"


class SVM {
public:
    corpus *corp;
    real C = 10.0f;
    real old_b = 0.0f;
    real new_b = 0.0f;
    vector<real> w;
    vector<real> alphas;
    vector<real> error_table;
    vector<real> pres;

    unordered_set<int> supports;
    unordered_set<int> softs;
    unordered_map<int, unordered_map<int, real>> kernel_table;
    size_t iter_num=0;
    real k11 = 0.0f;
    real k22 = 0.0f;
    real k12 = 0.0f;
    real new_alpha1 = 0.0f;
    real new_alpha2 = 0.0f;
    real old_alpha1 = 0.0f;
    real old_alpha2 = 0.0f;
    real y1 = 0.0f;
    real y2 = 0.0f;
    bool stop = false;
    int idx1;
    int idx2;
public:
    SVM(){}

    ~SVM(){
        if (0 != corp) delete corp;
    }

    real dot_mul(real *a, real *b, int size){
        real res = 0.0f;
        for (int i = 0; i < size; ++i, ++a, ++b) {
            res += (*a) * (*b);
        }
        return res;
    }

    real kernel_fn(int i, int j){
        if (kernel_table.count(i) == 1 && kernel_table[i].count(j) == 1){
            return kernel_table[i][j];
        }
        if (kernel_table.count(j) == 1 && kernel_table[j].count(i) == 1){
            return kernel_table[j][i];
        }
        real *a = corp->get_x(i), *b = corp->get_x(j);
        kernel_table[i][j] = dot_mul(a, b, corp->x_dim);
        kernel_table[j][i] =  kernel_table[i][j];
        return kernel_table[j][i];
    }

    void init_params(){
        alphas.resize(corp->size);
        std::fill(alphas.begin(), alphas.end(), 0.0f);

        error_table.resize(corp->size);
        std::fill(error_table.begin(), error_table.end(), 0.0f);
        for (size_t i = 0; i < corp->size; ++i){
            error_table[i] -= corp->get_label(i);
        }
    }

    void select_variables(){
        if (iter_num == 0){
            srand((unsigned)time(NULL));
            idx1 = rand() % (int)corp->size;
            real y1 = corp->get_label(idx1);
            while(true){
                idx2 = rand() % (int)corp->size;
                real y2 = corp->get_label(idx2);
                if (y1 != y2) break;
            }
        } else {
            idx1 = -1;
            idx2 = -1;
            for (int idx: supports){
                real c = check_kkt(idx);
                if (c != 1.0f) {
                    idx1 = idx;
                }
            }

            if (idx1 == -1){
                for (size_t i = 0; i < corp->size; ++i){
                    if (supports.count(i)) continue;
                    real c = check_kkt(i);
                    if (alphas[i] == 0.0f){
                        if (c < 1) idx1 = i;
                    }else if (alphas[i] == C) {
                        if (c > 1) idx1 = i;
                    }
                }
            }

            if (idx1 == -1) {
                stop = true;
                return;
            }

            real error1 = error_table[idx1], error2;
            real mx = -MAXFLOAT;
            for (size_t i = 0; i < corp->size; ++i){
                if (i == idx1) continue;
                error2 = error_table[i];
                if (mx < abs(error1 - error2)){
                    mx = abs(error1 - error2);
                    idx2 = i;
                }
            }
        }

        ++iter_num;
    }

    real check_kkt(LONG idx){
        real pre = 0.0f, y;
//        for (int i = 0; i < corp->size; ++i){
//            if (alphas[i] == 0.0f) continue;
//            y = corp->get_label(i);
//            pre += alphas[i] * y * kernel_fn(i, idx);
//        }
        for (int i: supports){
            y = corp->get_label(i);
            pre += alphas[i] * y * kernel_fn(i, idx);
        }

        for (int i: softs){
            y = corp->get_label(i);
            pre += C * y * kernel_fn(i, idx);
        }
        pre += old_b;
        y = corp->get_label(idx);
        return y * pre;
    }

    bool update_alphas(){
        old_alpha1 = alphas[idx1];
        old_alpha2 = alphas[idx2];
        y1 = corp->get_label(idx1);
        y2 = corp->get_label(idx2);

        k11 = kernel_fn(idx1, idx1);
        k22 = kernel_fn(idx2, idx2);
        k12 = kernel_fn(idx1, idx2);
        if ((k11 + k22 - 2 * k12) == 0.0f) {
            std::cout << k11 + k22 - 2 * k12 << std::endl;
            return false;
        }

        real L, H;
        if (y1 == y2){
            L = max(0.0f, old_alpha1 + old_alpha2 - C);
            H = min(C, old_alpha1 + old_alpha2);
        } else {
            L = max(0.0f, old_alpha2 - old_alpha1);
            H = min(C, C + old_alpha2 - old_alpha1);
        }

        real bias = y2 * (error_table[idx1] - error_table[idx2]) / (k11 + k22 - 2 * k12);

        alphas[idx2] += bias;
        new_alpha2 = alphas[idx2];

        if (alphas[idx2] < L){
            alphas[idx2] = L;
            new_alpha2 = L;
        }else if (alphas[idx2] > H){
            alphas[idx2] = H;
            new_alpha2 = H;
        }

        alphas[idx1] += y1 * y2 * (-bias);
        new_alpha1 = alphas[idx1];

        std::cout << alphas[idx1] << ":" << alphas[idx2] << std::endl;
        if (alphas[idx1] > 0 && alphas[idx1] < C){
            supports.insert(idx1);
            if (softs.count(idx1))
                softs.erase(idx1);
        }
        if (alphas[idx1] == C){
            if (supports.count(idx1))
                supports.erase(idx1);
            softs.insert(idx1);
        }
        if (alphas[idx2] > 0 && alphas[idx2] < C){
            supports.insert(idx2);
            if (softs.count(idx2))
                softs.erase(idx2);
        }
        if (alphas[idx2] == C){
            if (supports.count(idx2))
                supports.erase(idx2);
            softs.insert(idx2);
        }
        return true;
    }

    void update_bias(){
        if (new_alpha1 > 0 && new_alpha1 < C && new_alpha2 > 0 && new_alpha2 < C){
            new_b = -error_table[idx1] - y1 * k11 *(new_alpha1 - old_alpha1) - y2 * k12 * (new_alpha2 - old_alpha2) + old_b;
        }else if (new_alpha1 == 0.0f || new_alpha1 == C || new_alpha2 == 0.0f || new_alpha2 == C){
            real b0 = -error_table[idx1] - y1 * k11 *(new_alpha1 - old_alpha1) - y2 * k12 * (new_alpha2 - old_alpha2) + old_b;
            real b1 = -error_table[idx2] - y1 * k12 *(new_alpha1 - old_alpha1) - y2 * k22 * (new_alpha2 - old_alpha2) + old_b;
            new_b = (b0 + b1) / 2.0f;
        }
    }

    void update_error_table(){
        for (size_t i = 0; i < corp->size; ++i){
            error_table[i] += new_b - old_b;
            real y1 = corp->get_label(idx1);
            real y2 = corp->get_label(idx2);
            real k1 = kernel_fn(idx1, i);
            real k2 = kernel_fn(idx2, i);
            error_table[i] += (new_alpha1 - old_alpha1) * y1 * k1 + (new_alpha2 - old_alpha2) * y2 * k2;
        }
        old_b = new_b;
    }

    bool should_stop(){
        return stop;
    }

    void train(string filename){
        corp = new corpus(filename, 128, false, false);
        corp->get_data();
        logger.info("Load data successfully, size=[" + std::to_string(corp->size) + "," +  std::to_string(corp->x_dim) + "].");
        init_params();

        select_variables();
        while (!should_stop()){
            if (!update_alphas()) continue;
            update_bias();
            update_error_table();
            select_variables();
        }
    }
};


#endif //FASTAI_SVM_H
