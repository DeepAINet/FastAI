//
// Created by mengqy on 2019/4/11.
//

#ifndef FASTAI_BASE_TREES_H
#define FASTAI_BASE_TREES_H

#include "corpus.h"
#include "DecisionTree.h"
#include "tensor_ops.h"
#include "util.h"

class BaseCart: public DecisionTree{
public:
    bool regression = false;
    bool binary_class = true;
    bool multi_class = false;

public:
    BaseCart(bool regression):regression(regression){}

    BaseCart(){}

    void prune() {}

    void get_label(TreeNode *node){}

    real compute_purity(TreeNode *node){
        node->purity = compute_sqrt_error(node->x);
        return node->purity;
    }

    real compute_sqrt_error(const vector<LONG>& ids) const{
        real sum = 0.0f, pow_sum = 0.0f, t;
        for (LONG idx: ids){
            t = __corpus->get_label(idx);
            sum += t;
            pow_sum += t * t;

        }
//        sum /= (real)ids.size();
        return pow_sum  - sum * sum / (real)(ids.size());
    }

    vector<TreeNode*> get_terminate_nodes(){
        vector<TreeNode*> nodes;
        queue<TreeNode*> q;
        q.push(root);
        while(!q.empty()){
            TreeNode *node = q.front();
            q.pop();
            if (node->leaf) nodes.push_back(node);
            else{
                for (TreeNode *n: node->sons) q.push(n);
            }
        }
        return nodes;
    }

    void compute_best_point_regression(TreeNode *node, int32_t fid){
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

        real purity;
        real min_purity=node->purity, best_point=0.0f;
        real left, right;
        for (auto iter = x_map.begin(); iter != x_map.end(); ++iter) {
            sum = sum_map[iter->first] / (real)iter->second.size();
            pow_sum = pow_sum_map[iter->first] /  (real)iter->second.size();
            temp = node->x.size() - iter->second.size();
            purity = (total_sum - sum) / temp;
            left = pow_sum - sum * sum;
            right = (total_pow_sum - pow_sum_map[iter->first]) / temp - purity * purity;
            purity =  left + right;
            if (min_purity > purity){
                min_purity = purity;
                best_point = iter->first;
                node->feature_purity_table[fid][0] = left;
                node->feature_purity_table[fid][1] = right;
            }
        };
        node->purities[fid] = min_purity;
        node->__span[fid] = best_point;
    }

    void compute_best_split_regression(TreeNode *node, int32_t fid, RECORD *records, size_t size){
        real total_sum=0.0f, total_pow_sum=0.0f;
        for (size_t i = 0; i < size; ++i){
            total_sum += records[i].y;
            total_pow_sum += records[i].y * records[i].y;
        }

        real left_sum = 0.0f, left_pow_sum = 0.0f;
        real r0, r1, l0, l1, left, right;
        real purity;
        real min_purity = node->purity, best_split = 0.0f;
        for (LONG i = 0; i < size - 1; ++i){
            if (i == 0 || (records[i].x == records[i-1].x)) {
                left_sum += records[i].y;
                total_sum -= records[i].y;

                l0 = records[i].y * records[i].y;
                left_pow_sum += l0;
                total_pow_sum -= l0;
                continue;
            }
            left = left_pow_sum - (left_sum * left_sum) / i;
            right = total_pow_sum - (total_sum * total_sum) / (size - i);
            purity = left + right;
            if (min_purity > purity){
                min_purity = purity;
                best_split = (records[i].x + records[i-1].x)/2.0f;
                node->feature_purity_table[fid][0] = left;
                node->feature_purity_table[fid][1] = right;
            }
            left_sum += records[i].y;
            total_sum -= records[i].y;
            l0 = records[i].y * records[i].y;
            left_pow_sum += l0;
            total_pow_sum -= l0;
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

        compute_best_split_regression(node, fid, records, size);
        free(records);
    }

    void compute_best_point(TreeNode *node, int32_t fid){
        assert(fid >= 0);
        assert(fid < node->feature_remainder.size());
        compute_best_point_regression(node, fid);
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
        if (node->height == MAX_HEIGHT
        || node->purity <= MIN_ERROR
        || node->x.size() <= MIN_SAMPLES
        || node->feature_remainder.empty()){
            get_label(node);
//            std::cout << "terminate node purity:"<< node->purity << std::endl;
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
//        std::cout << idx_purities.second  << ":" <<  node->purity << std::endl;
        if (idx_purities.second >= node->purity){
            terminate(node);
            return;
        }
        int32_t best_feature = node->feature_remainder[idx_purities.first];

        auto *son0 = new TreeNode();
        auto *son1 = new TreeNode();
        son0->subject = __corpus->feature_names[best_feature];
        son1->subject = __corpus->feature_names[best_feature];
        son0->height = node->height + 1;
        son1->height = node->height + 1;
        update_sons(node, son0, son1, idx_purities.first);
    }

    void update_sons(TreeNode *parent, TreeNode *son0, TreeNode *son1, size_t fid){
        if (__corpus->feature_types[parent->feature_remainder[fid]])
            update_continuous_sons(parent, son0, son1, fid);
        else update_discrete_sons(parent, son0, son1, fid);
    }

    void update_continuous_sons(TreeNode *parent, TreeNode *son0, TreeNode *son1, int32_t fid){
        real point = parent->__span[fid];
//        std::cout << "split-point:" << point << std::endl;
        real min_value = point, max_value = point, value;
        int32_t best_feature = parent->feature_remainder[fid];
        for (LONG idx: parent->x){
            value = __corpus->get_x(idx, best_feature);
            if (value < point) son0->x.push_back(idx);
            else son1->x.push_back(idx);
            min_value = min(min_value, value);
            max_value = max(max_value, value);
        }

//        std::cout << "son-X ["<< son0->x.size() << ":" << son1->x.size() << "]" << std::endl;

        son0->purity = parent->feature_purity_table[fid][0];
        son1->purity = parent->feature_purity_table[fid][1];
        parent->span.push_back(min_value);
        parent->span.push_back(point);
        parent->span.push_back(max_value);
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
        parent->shrink_to_fit();
    }

    void update_discrete_sons(TreeNode *parent, TreeNode *son0, TreeNode *son1, int32_t fid){
        real point = parent->__span[fid];
        real value;
        int32_t best_feature = parent->feature_remainder[fid];
        set<real> value_set;
        for (LONG idx: parent->x){
            value = __corpus->get_x(idx, best_feature);
            if (value == point) son0->x.push_back(idx);
            else {
                son1->x.push_back(idx);
                value_set.insert(value);
            }
        }
        son0->purity = parent->feature_purity_table[fid][0];
        son1->purity = parent->feature_purity_table[fid][1];
        parent->span.push_back(point);
        for(real v:value_set) parent->span.push_back(v);
        parent->son_map[0] = son0;
        parent->son_map[1] = son1;
        parent->best_feature = best_feature;

        for (int i = 0; i < parent->feature_remainder.size(); ++i){
            if (parent->purities[i] != MAXFLOAT && parent->feature_remainder[i] != best_feature) {
                son0->feature_remainder.push_back(parent->feature_remainder[i]);
                son1->feature_remainder.push_back(parent->feature_remainder[i]);
            }
        }
        if (value_set.size() > 1)
            son1->feature_remainder.push_back(best_feature);

        son0->resize();
        son1->resize();
        parent->sons.push_back(son0);
        parent->sons.push_back(son1);
        parent->shrink_to_fit();
    }

    vector<real> predict(corpus *corp){
        TreeNode *node;
        vector<real> labels;
        real v;
        LONG size = corp->size;
        for (int i = 0; i < size; ++i){
            node = root;
            while(!node->leaf){
                v = corp->get_x(i, node->best_feature);
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
            labels.push_back(node->y);
        }
        return labels;
    }
};



#endif //FASTAI_BASE_TREES_H
