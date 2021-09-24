//
// Created by mengqy on 2019/1/24.
//

#ifndef FASTAI_ID3_H
#define FASTAI_ID3_H

#include <thread>
#include <future>
#include "DecisionTree.h"

class ID3: public DecisionTree{

public:
    ID3(){}

    real compute_purity(const vector<LONG>& ids){
        real res = 0;
        if (ids.size() <= 1) return res;
        map<real, LONG> count;
        for (LONG id: ids){
            count[__corpus->get_label(id)]++;
        }
        return compute_entropy(count, (LONG)ids.size());
    }

    real compute_purity(TreeNode *node){
        node->purity = compute_purity(node->x);
        return node->purity;
    }

    void select_feature_thread(TreeNode *node, const vector<int32_t>& fids){
        map<real, vector<LONG>> x_map;
        size_t total = node->x.size();
        real purity, weight, tmp;
        for (int32_t fid: fids){
            purity = 0.0;
            x_map.clear();
            for (LONG x: node->x){
                x_map[__corpus->get_x(x, node->feature_remainder[fid])].push_back(x);
            }
            for (auto iter = x_map.begin(); iter != x_map.end(); ++iter){
                tmp = compute_purity(iter->second);
                weight = (real)iter->second.size() / (real)total;
                purity += weight * tmp;
                node->feature_purity_table[node->feature_remainder[fid]][iter->first] = tmp;
            }
            node->purities[fid] = node->purity - purity;
        }
    }

    bool can_split(TreeNode *node){
        if (0 == node) return false;
        if (node->purity == 0.0) {
            node->y = __corpus->get_label(node->x[0]);
            node->leaf = true;
            return false;
        }else if (node->feature_remainder.empty()) {
            node->y = __corpus->most_common(node->x);
            node->leaf = true;
            return false;
        }
        return true;
    }

    void terminate(TreeNode *node){
        node->y = __corpus->most_common(node->x);
        node->leaf = true;
    }

    void push_sons(TreeNode *node){
        pair<int32_t, real> id_purities = argmax(node->purities);
        int32_t best_feature = node->feature_remainder[id_purities.first];
        if (id_purities.second <= 0.0) {
            terminate(node);
            return;
        }

        if (DEBUG) std::cout << "|" << __corpus->feature_names[best_feature] << "-purity|:" << id_purities.second << std::endl;

        node->sons.clear();
        map<real, vector<LONG> > x_map;
        for (LONG x: node->x){
            x_map[__corpus->get_x(x, best_feature)].push_back(x);
        }

        for (auto iter = x_map.begin(); iter != x_map.end(); ++iter){
            TreeNode *son = new TreeNode();
            son->x = iter->second;
            son->span.push_back(iter->first);
            son->subject = __corpus->feature_names[best_feature];
            son->feature_remainder.clear();
            for (int i = 0; i < node->feature_remainder.size(); ++i){
                if (node->feature_remainder[i] != best_feature) {
                    son->feature_remainder.push_back(node->feature_remainder[i]);
                }
            }
            son->purities.resize(son->feature_remainder.size());
            son->purity = node->feature_purity_table[best_feature][iter->first];
            node->sons.push_back(son);
            node->son_map[iter->first] = son;
        }
        node->best_feature = best_feature;
        node->shrink_to_fit();
    }

    void predict(corpus *corp){
        for (int i = 0; i < corp->size; ++i){
            TreeNode *node = root;
            while(!node->leaf){
                real v = corp->get_x(i, node->best_feature);
                node = node->son_map[v];
            }
            std::cout << i << ':' << corp->idx_value_map[node->y] << std::endl;
        }
    }

    void prune(){}
};

#endif //FASTAI_ID3_H
