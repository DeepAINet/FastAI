//
// Created by mengqy on 2019/1/24.
//

#ifndef FASTAI_C4_5_H
#define FASTAI_C4_5_H

#include <cstdlib>
#include "DecisionTree.h"

#define MIN_PURITY 0.0f
enum class prune_name : int {PEP= 1, REP};

class C45: public DecisionTree{
private:
    int strategy = 1;
    prune_name pname = prune_name::PEP;
public:
    C45(){}

    real compute_purity(const vector<LONG>& ids){
        map<real, LONG> count;
        for (LONG id: ids){
            count[__corpus->get_label(id)]++;
        }
        return compute_entropy(count, (LONG)ids.size());
    }

    real compute_purity(TreeNode *node){
        return node->purity = compute_purity(node->x);
    }

    void compute_discrete_feature(TreeNode* node, int32_t fid){
        assert(fid >= 0);
        assert(fid < node->feature_remainder.size());;

        int feature = node->feature_remainder[fid];
        map<real, vector<LONG>> x_map;
        for (LONG x: node->x){
            x_map[__corpus->get_x(x, feature)].push_back(x);
        }
        if (x_map.size() == 1) {
            node->purities[fid] = -MAXFLOAT;
            return;
        }

        int j = 0;
        real __feature_value_entropy, purity=0.0, tmp;
        size_t size = node->x.size();
        vector<real> __ratios(x_map.size(), 0.0);
        for (auto iter = x_map.cbegin(); iter != x_map.cend(); ++iter){
            __ratios[j] = (real)iter->second.size() / (real)size;
            tmp = compute_purity(iter->second);
            node->feature_purity_table[fid][iter->first] = tmp;
            purity +=  __ratios[j] * tmp;
            ++j;
        }

        __feature_value_entropy = compute_entropy(__ratios);
        if (std::isnan(__feature_value_entropy) || std::isinf(__feature_value_entropy)){
            std::cerr << "entropy is NaN or InF";
            exit(EXIT_FAILURE);
        }
        assert(__feature_value_entropy != 0.0f);
        node->purities[fid] = (node->purity - purity) / __feature_value_entropy;
    }

    void compute_continuous_feature(TreeNode *node, int fid){
        assert(fid >= 0);
        assert(fid < node->feature_remainder.size());

        size_t size = node->x.size();
        int feature = node->feature_remainder[fid];
        RECORD* records = (RECORD *)malloc(size * sizeof(RECORD));
        for (size_t i = 0; i < size; ++i){
            records[i].idx = node->x[i];
            records[i].x = __corpus->get_x(node->x[i], feature);
            records[i].y = __corpus->get_label(node->x[i]);
        }
        qsort(records, size, sizeof(RECORD), cmp);
        if (records[0].x == records[size-1].x){
            node->purities[fid] = -MAXFLOAT;
            free(records);
            return;
        }

        map<real, LONG> left, right;
        for (size_t i = 0; i < size; ++i){
            ++right[records[i].y];
        }

        real t, purity, w, left_entropy, right_entropy, max_purity=-MAXFLOAT;
        size_t count = 1, num_values=1;
        size_t i = 0;
        while (i < size){
            if (i == 0 || (i > 0 && records[i].y == records[i-1].y)){
                left[records[i].y]++;
                right[records[i].y]--;
                ++i;
                continue;
            }
            if (i > 0 && records[i].x != records[i-1].x) ++num_values;
            left_entropy = compute_entropy(left, i);
            right_entropy = compute_entropy(right, size-i);
            w = (real)i/(real)size;
            purity = node->purity - w * left_entropy - (1-w) * right_entropy;
            if (strategy == 0){
                w = compute_entropy(w);;
                purity /= w;
            }
            if (max_purity < purity){
                if (records[i-1].x != records[i].x){
                    max_purity = purity;
                    node->__span[fid] = (records[i-1].x + records[i].x) / 2.0f;
                }else {
                    max_purity = -MAXFLOAT;
                    node->__span[fid] = -MAXFLOAT;
                }
                node->feature_purity_table[fid][0.0f] = left_entropy;
                node->feature_purity_table[fid][1.0f] = right_entropy;
                count = i;
            }
            left[records[i].y]++;
            right[records[i].y]--;
            ++i;
        }
        if (strategy == 0){
            node->purities[fid] = max_purity;
        }else if (strategy == 1){
            w = (real)count / (real)size;
            w = compute_entropy(w);
//            node->purities[fid] = (max_purity - std::log((real)(num_values)/(real)size)) / w;
            node->purities[fid] = max_purity / w;
        }
        if (0 != records) free(records);
    }

    void select_feature_thread(TreeNode* node, const vector<int32_t>& fids){
        for (int32_t fid: fids){
            if (__corpus->feature_types[node->feature_remainder[fid]]){
                compute_continuous_feature(node, fid);
            }else{
                compute_discrete_feature(node, fid);
            }
        }
    }

    void update_feature_remainder(TreeNode *p, TreeNode *s, int best_feature){
        for (int32_t i = 0; i < p->feature_remainder.size(); ++i){
            if (p->purities[i] == -MAXFLOAT)
                continue;
            if (!__corpus->feature_types[best_feature] && p->feature_remainder[i] == best_feature)
                continue;
            s->feature_remainder.push_back(p->feature_remainder[i]);
        }
    }

    bool can_split(TreeNode *node){
        if (0 == node) return false;
        if (node->purity <= MIN_PURITY || node->x.size() == 1){
            most_common(node);
            node->leaf = true;
            return false;
        }
        if (node->feature_remainder.empty()){
            most_common(node);
            node->leaf = true;
            return false;
        }
        return true;
    }

    void terminate(TreeNode *node){
        most_common(node);
        node->leaf = true;
    }

    void most_common(TreeNode *node){
        LONG most_count = 0;
        node->y = __corpus->most_common(node->x, most_count);
        node->most_common = most_count;
        node->size = node->x.size();
    }

    void push_sons(TreeNode *node){
        pair<size_t, real> id_purities = argmax(node->purities);
        int32_t best_feature = node->feature_remainder[id_purities.first];
        if (node->purities[id_purities.first] <= 0) {
            terminate(node);
            return;
        }
        map<real, vector<LONG>> x_map;
        real __min, __max;
        if (!__corpus->feature_types[best_feature]){
            for (LONG idx: node->x){
                x_map[__corpus->get_x(idx, best_feature)].push_back(idx);
            }
        }else{
            __min = MAXFLOAT, __max=-MAXFLOAT;
            real tmp;
            for (LONG idx: node->x){
                tmp = __corpus->get_x(idx, best_feature);
                __min = min(tmp, __min);
                __max = max(tmp, __max);
                if (tmp <= node->__span[id_purities.first])
                    x_map[0.0f].push_back(idx);
                else x_map[1.0f].push_back(idx);
            }
            assert(x_map[0.0f].size() > 0);
            assert(x_map[1.0f].size() > 0);
        }

        node->sons.clear();
        string temp = "[";
        for (auto iter = x_map.cbegin(); iter != x_map.cend(); ++iter){
            TreeNode *son = new TreeNode();
            son->feature_remainder.clear();
            update_feature_remainder(node, son, best_feature);
            son->subject = __corpus->feature_names[best_feature];
            son->x = iter->second;
            node->sons.push_back(son);
            temp += std::to_string(iter->second.size()) + ",";
            son->resize();

            if (!__corpus->feature_types[best_feature]){
                node->span.push_back(iter->first);
                son->purity = node->feature_purity_table[id_purities.first][iter->first];
                node->son_map[iter->first] = son;
            }else{
                if (iter->first == 0.0f){
                    node->span.push_back(__min);
                    node->span.push_back(node->__span[id_purities.first]);
                    node->son_map[0.0f] = son;
                    son->purity = node->feature_purity_table[id_purities.first][0.0f];
                }
                if (iter->first == 1.0f){
                    node->span.push_back(__max);
                    node->son_map[1.0f] = son;
                    son->purity = node->feature_purity_table[id_purities.first][1.0f];
                }
            }
        }
        most_common(node);
        temp = temp.substr(0, temp.size()-1) + "]";
        if (DEBUG) std::cout << temp << std::endl;
        node->best_feature = best_feature;
        node->shrink_to_fit();
    }

    void prune(){
        logger.info("start to prune ... ");
        int before_height = get_height(root);
        if (pname == prune_name::PEP){
            pep_prune(root);
        }else if (pname == prune_name::REP){
            ;
        }
        int after_height = get_height(root);
//        if (DEBUG)
        std::cout << "original-height:" << before_height << ", pruned-height:" << after_height << std::endl;
        logger.info("prune ending ... ");
    }

    void pep_prune(TreeNode *node){
        vector<TreeNode*> leaves;
        __pep_prune(node, leaves);
    }

    void __pep_prune(TreeNode *node, vector<TreeNode*>& leaves){
        leaves.clear();
        for (TreeNode *son: node->sons){
            if (!son->leaf) {
                vector<TreeNode*> l;
                __pep_prune(son, l);
                for (TreeNode *n: l)
                    leaves.push_back(n);
            }else leaves.push_back(son);
        }
        real after = node->size - node->most_common + 0.5f;
        real before = 0.0f;
        for (TreeNode *son: leaves){
            before += son->size - son->most_common + 0.5f;
        }
        real var = sqrt(before * (1 - before/(real)node->size));
        if (after < before + var){
            for (TreeNode *son: node->sons){
                delete son;
            }
            node->sons.clear();
            node->son_map.clear();
            node->leaf = true;
            leaves.clear();
            leaves.push_back(node);
        }
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
                }else node = node->son_map[v];
            }
            if (corp->get_label(i) == node->y)
                ++count;
        }

        if (train_data)
            std::cout << "train accuracy:" << (real)count / (real)corp->size << std::endl;
        else std::cout << "test accuracy:" << (real)count / (real)corp->size << std::endl;
    }

    int get_height(TreeNode *node){
        if (node->leaf) return 1;
        int m;
        for (int i = 0; i < node->sons.size(); ++i){
            if (i == 0) m = get_height(node->sons[i]);
            else {
                int n = get_height(node->sons[i]);
                if (m < n) m = n;
            }
        }
        return m+1;
    }

    void test(){
        queue<TreeNode*> q;
        q.push(root);

        while(!q.empty()){
            TreeNode *t = q.front();
            std::cout << t->size << " ==== "<< t->most_common << std::endl;
            q.pop();
            for (TreeNode *n: t->sons){
                q.push(n);
            }
        }
    }
};
#endif //FASTAI_C4_5_H
