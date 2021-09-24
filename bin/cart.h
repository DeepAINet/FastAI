//
// Created by mengqy on 2019/1/24.
//

#ifndef FASTAI_CART_H
#define FASTAI_CART_H

#include <stdlib.h>
#include <sstream>
#include "DecisionTree.h"


class cart: public DecisionTree{
    bool regression = false;

public:
    cart(bool regression):regression(regression){}

    cart(){}

    void prune() {}

    /**
     * get the label of a node.
     * @param node
     */
    void get_label(TreeNode *node){
        if (regression){
            real sum = 0.0;
            for (LONG idx: node->x){
                sum += __corpus->get_label(idx);
            }
            node->y = sum / (real)node->x.size();
        }else{
            LONG most_count = 0;
            most_common(node);
        }
    }

    real __compute_gini_index(const vector<LONG>& ids) const{
        map<real, LONG> y_map;
        for(LONG idx: ids){
            y_map[__corpus->get_label(idx)]++;
        }
        return compute_gini_index(y_map, (LONG)ids.size());
    }

    real __compute_gini_index(RECORD *records, LONG start, LONG end){
        map<real, LONG> y_map;
        for(LONG i = start; i <= end; ++i){
            y_map[records[i].y]++;
        }
        return compute_gini_index(y_map, end-start+1);
    }

    real compute_purity(TreeNode *node){
        if (regression){
            node->purity = __compute_sqrt_error(node->x);
        } else {
            node->purity = __compute_gini_index(node->x);
        }
        return node->purity;
    }


    real __compute_sqrt_error(const vector<LONG>& ids) const{
        real sum = 0.0, avg = 0.0f, error;
        for (LONG idx: ids){
            error = __corpus->get_label(idx);
            sum += error * error;
            avg += error;
        }
        avg /= (real)ids.size();
        sum /= (real)ids.size();
        return sum - avg * avg;
    }

    real __compute_sqrt_error(RECORD *records, LONG start, LONG end){
        real sum = 0.0, avg, error;
        for (LONG i = start; i <= end; ++i){
            error = records[start].y;
            sum += error * error;
            avg += error;
        }
        avg /= (real)(end-start+1);
        sum /= (real)(end-start+1);
        return sum - avg * avg;
    }

    void __compute_best_point_regression(TreeNode *node, int32_t fid){
        map<real, vector<LONG>> x_map;

        for (LONG idx: node->x)
            x_map[__corpus->get_x(idx, node->feature_remainder[fid])].push_back(idx);

        if (x_map.size() == 1){
            node->purities[fid] = MAXFLOAT;
            return;
        }

        map<real, real> sum_map;
        map<real, real> pow_sum_map;
        real temp, sum, pow_sum, total_sum=0.0f, total_pow_sum=0.0f;
        for (auto iter = x_map.begin(); iter != x_map.end(); ++iter){
            sum = 0.0, pow_sum = 0.0;
            for (LONG idx : iter->second){
                temp = __corpus->get_label(idx);
                sum += temp;
                pow_sum += temp * temp;
            }
            sum_map[iter->first] = sum;
            pow_sum_map[iter->first] = pow_sum;
            total_pow_sum += pow_sum;
            total_sum += sum;
        }

        real t;
        real __min_purity=node->purity, __best_point=0.0f;
        real left, right;
        for (auto iter = x_map.begin(); iter != x_map.end(); ++iter) {
            sum = sum_map[iter->first] / (real)iter->second.size();
            pow_sum = pow_sum_map[iter->first] /  (real)iter->second.size();
            temp = node->x.size() - iter->second.size();
            t = (total_sum - sum) / temp;
            left = pow_sum - sum * sum;
            right = (total_pow_sum - pow_sum_map[iter->first]) / temp - t * t;
            t =  left + right;
            if (__min_purity > t){
                __min_purity = t;
                __best_point = iter->first;
                node->feature_purity_table[fid][0] = left;
                node->feature_purity_table[fid][1] = right;
            }
        };
        node->purities[fid] = __min_purity;
        node->__span[fid] = __best_point;
    }

    void __compute_best_point_classification(TreeNode *node, int32_t fid){
        map<real, pair<map<real, LONG>, LONG>> __x_point_map;

        map<real, LONG> __total_label_map;
        map<real, LONG> __remain;
        int32_t feature = node->feature_remainder[fid];
        real __x_point, __label;
        for (LONG idx: node->x){
            __x_point = __corpus->get_x(idx, feature);
            __label = __corpus->get_label(idx);
            ++__x_point_map[__x_point].first[__label];
            ++__x_point_map[__x_point].second;
            ++__total_label_map[__label];
        }

        if (__x_point_map.size() == 1){
            node->purities[fid] = MAXFLOAT;
            return;
        }

        real __left, __right, __weight, __purity, __min_purity=node->purity, __best_point=0.0f;
        for (auto iter = __x_point_map.begin(); iter != __x_point_map.end(); ++iter){
            __remain.clear();

            for (auto item = __total_label_map.begin(); item != __total_label_map.end(); ++item){
                if (iter->second.first.count(item->first)){
                    __remain[item->first] = __total_label_map[item->first] - iter->second.first[item->first];
                }else __remain[item->first] = __total_label_map[item->first];
            }

            __weight = (real)iter->second.second / (real)node->x.size();
            __left = compute_gini_index(iter->second.first, iter->second.second);
            __right =  compute_gini_index(__remain, node->x.size() - iter->second.second);
            __purity = __weight * __left + (1.0f - __weight) * __right;

            if (__min_purity > __purity){
                __min_purity = __purity;
                __best_point = iter->first;
                node->feature_purity_table[fid][0] = __left;
                node->feature_purity_table[fid][1] = __right;
            }
        }

        node->purities[fid] = __min_purity;
        node->__span[fid] = __best_point;
    }

    void __compute_best_split_classification(TreeNode *node, int32_t fid, RECORD *records, size_t size){
        map<real, LONG> left, right;
        for (size_t i = 0; i < size; ++i){
            ++right[records[i].y];
        }
        real __weight, __left, __right, __purity;
        real __min_purity = node->purity;
        real __best_split;
        size_t i = 0;
        while (i < size){
            if (i == 0 || (i > 0 && records[i].y == records[i-1].y)){
                left[records[i].y]++;
                right[records[i].y]--;
                ++i;
                continue;
            }

            __left = compute_gini_index(left, i);
            __right = compute_gini_index(right, size-i);
            __weight = (real)i / (real)size;
            __purity = __weight * __left + (1 - __weight) * __right;
            if (__min_purity > __purity){
                if (records[i].x != records[i-1].x){
                    __min_purity = __purity;
                    __best_split = (records[i-1].x + records[i].x)/2.0f;
                }else{
                    __min_purity = MAXFLOAT;
                    __best_split = MAXFLOAT;
                }
                node->feature_purity_table[fid][0] = __left;
                node->feature_purity_table[fid][01] = __right;
            }
            left[records[i].y]++;
            right[records[i].y]--;
            ++i;
        }

        node->purities[fid] = __min_purity;
        node->__span[fid] = __best_split;
    }

    void __compute_best_split_regression(TreeNode *node, int32_t fid, RECORD *records, size_t size){
        real right_sum=0.0, right_pow_sum=0.0;
        for (size_t i = 0; i < size; ++i){
            right_sum += records[i].y;
            right_pow_sum += records[i].y * records[i].y;
        }

        real left_sum = 0.0f, left_pow_sum = 0.0f;
        real r0, r1, l0, l1, left, right;
        real purity;
        real min_purity = node->purity, best_split = 0.0f;
        for (size_t i = 0; i < size - 1; ++i){
            left_sum += records[i].y;
            l0 = records[i].y * records[i].y;
            left_pow_sum += l0;
            right_sum -= records[i].y;
            right_pow_sum -= l0;

            l0 = left_pow_sum / (real)(i+1);
            l1 = left_sum / (real)(i+1);
            r0 = right_pow_sum / (real)(size - i - 1);
            r1 = right_sum / (real)(size - i - 1);
            left = l0 - l1 * l1;
            right = r0 - r1 * r1;
            purity = left + right;
            if (min_purity > purity){
                if (records[i].x == records[i-1].x){
                    min_purity = MAXFLOAT;
                    best_split = MAXFLOAT;
                }else{
                    min_purity = purity;
                    best_split = (records[i].x + records[i-1].x)/2.0f;
                }
                node->feature_purity_table[fid][0] = left;
                node->feature_purity_table[fid][1] = right;
            }
        }

        node->purities[fid] = min_purity;
        node->__span[fid] = best_split;
    }

    void compute_best_split(TreeNode *node, int32_t fid){
        assert(fid >= 0);
        assert(fid < node->feature_remainder.size());

        size_t size = node->x.size();
        auto *records = (RECORD *)malloc(sizeof(RECORD) * size);
        if (0 == records){
            std::cerr << "records allocate failure!";
            exit(EXIT_FAILURE);
        }
        LONG i = 0;
        for (LONG idx: node->x){
            records[i].idx = idx;
            records[i].x = __corpus->get_x(idx, node->feature_remainder[fid]);
            records[i].y = __corpus->get_label(idx);
            ++i;
        }

        qsort(records, size, sizeof(RECORD), cmp);

        if (records[0].x == records[size-1].x){
            node->purities[fid] = MAXFLOAT;
            free(records);
            return;
        }

        if (regression){
            __compute_best_split_regression(node, fid, records, size);
        }else{
            __compute_best_split_classification(node, fid, records, size);
        }

        if (0 != records) free(records);
    }

    void compute_best_point(TreeNode *node, int32_t fid){
        assert(fid >= 0);
        assert(fid < node->feature_remainder.size());
        if (regression)
            __compute_best_point_regression(node, fid);
        else
            __compute_best_point_classification(node, fid);
    }

    void select_feature_thread(TreeNode* node, const vector<int32_t>& fids){
        for (int32_t fid: fids){
            if (__corpus->feature_types[node->feature_remainder[fid]]){
                compute_best_split(node, fid);
            }else{
                compute_best_point(node, fid);
            }
        }
    }

    bool can_split(TreeNode *node){
        if (node->purity <= MIN_ERROR || node->x.size() <= MIN_SAMPLES || node->feature_remainder.empty()){
            get_label(node);
            node->leaf = true;
            return false;
        }
        return true;
    }

    void terminate(TreeNode *node){
        get_label(node);
        node->leaf = true;
    }

    void push_sons(TreeNode *node){
        pair<size_t, real> idx_purities = argmin(node->purities);
        if (idx_purities.second >= node->purity){
            terminate(node);
            return;
        }
        int32_t best_feature = node->feature_remainder[idx_purities.first];

        auto *__son0 = new TreeNode();
        auto *__son1 = new TreeNode();
        __son0->subject = __corpus->feature_names[best_feature];
        __son1->subject = __corpus->feature_names[best_feature];
        __update_sons(node, __son0, __son1, idx_purities.first);
    }

    void __update_sons(TreeNode *parent, TreeNode *son0, TreeNode *son1, size_t fid){
        if (__corpus->feature_types[parent->feature_remainder[fid]])
            __update_continuous_sons(parent, son0, son1, fid);
        else __update_discrete_sons(parent, son0, son1, fid);
    }

    void __update_continuous_sons(TreeNode *parent, TreeNode *son0, TreeNode *son1, int32_t fid){
        real __point = parent->__span[fid];
        real __min_value = __point, __max_value = __point, __value;
        int32_t best_feature = parent->feature_remainder[fid];
        for (LONG idx: parent->x){
            __value = __corpus->get_x(idx, best_feature);
            if (__value <= __point) son0->x.push_back(idx);
            else son1->x.push_back(idx);
            __min_value = min(__min_value, __value);
            __max_value = max(__max_value, __value);
        }

        son0->purity = parent->feature_purity_table[fid][0];
        son1->purity = parent->feature_purity_table[fid][1];
        parent->span.push_back(__min_value);
        parent->span.push_back(__point);
        parent->span.push_back(__max_value);
        parent->son_map[0] = son0;
        parent->son_map[1] = son1;
        parent->best_feature = best_feature;
        for (int i = 0; i < parent->feature_remainder.size(); ++i){
            if (parent->purities[i] != MAXFLOAT) {
                son0->feature_remainder.push_back(parent->feature_remainder[i]);
                son1->feature_remainder.push_back(parent->feature_remainder[i]);
            }
        }
//        std::cout << "[" << parent->x.size() << "," << son0->x.size() << "," << son1->x.size() << "]" << std::endl;
        son0->resize();
        son1->resize();
        parent->sons.push_back(son0);
        parent->sons.push_back(son1);
        if (!regression)
            most_common(parent);
        parent->shrink_to_fit();
    }

    void __update_discrete_sons(TreeNode *parent, TreeNode *son0, TreeNode *son1, int32_t fid){
        real __point = parent->__span[fid];
        real __value;
        int32_t best_feature = parent->feature_remainder[fid];
        set<real> __set;
        for (LONG idx: parent->x){
            __value = __corpus->get_x(idx, best_feature);
            if (__value == __point) son0->x.push_back(idx);
            else {
                son1->x.push_back(idx);
                __set.insert(__value);
            }
        }
        son0->purity = parent->feature_purity_table[fid][0];
        son1->purity = parent->feature_purity_table[fid][1];
        parent->span.push_back(__point);
        for(real v:__set) parent->span.push_back(v);
        parent->son_map[0] = son0;
        parent->son_map[1] = son1;
        parent->best_feature = best_feature;

        for (int i = 0; i < parent->feature_remainder.size(); ++i){
            if (parent->purities[i] != MAXFLOAT && parent->feature_remainder[i] != best_feature) {
                son0->feature_remainder.push_back(parent->feature_remainder[i]);
                son1->feature_remainder.push_back(parent->feature_remainder[i]);
            }
        }
        if (__set.size() > 1)
            son1->feature_remainder.push_back(best_feature);

        son0->resize();
        son1->resize();
        parent->sons.push_back(son0);
        parent->sons.push_back(son1);
        if (!regression)
            most_common(parent);
        parent->shrink_to_fit();
    }

    void predict(corpus *corp, bool train_data){
        int count = 0;
        for (int i = 0; i < corp->size; ++i){
            TreeNode *node = root;
            while(!node->leaf){
                real v = corp->get_x(i, node->best_feature);
                if (__corpus->feature_types[node->best_feature]){
                    if (v <= node->span[1]) node = node->son_map[0];
                    else node = node->son_map[1];
                }else {
                    if (v == node->span[0]) {
                        node = node->son_map[0];
                    }else{
                        node = node->son_map[1];
                    }
                }
            }
            if (corp->get_label(i) == node->y)
                ++count;

//            std::cout << node->y << ":" << corp->get_label(i) << std::endl;
        }

        if (train_data){
            std::cout << "train accuracy:" << (real)count / (real)corp->size
                      << "\ncorpus size:" << corp->size << std::endl;
        } else {
            std::cout << "test accuracy:" << (real)count / (real)corp->size << std::endl;
        }
    }
};


#endif //FASTAI_CART_H
