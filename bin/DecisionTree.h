//
// Created by mengqy on 2019/3/18.
//

#ifndef FASTAI_DECISIONTREE_H
#define FASTAI_DECISIONTREE_H

#include <string>
#include <list>
#include <queue>
#include <iomanip>
#include "common.h"
#include "corpus.h"

#define MIN_ERROR 0.0
#define MIN_SAMPLES 1
#define MAX_HEIGHT 10

using namespace std;

class TreeNode{
public:
    string subject;
    int32_t best_feature;
    int32_t height=0;
    map<real, TreeNode*> son_map;
    vector<real> span;
    vector<real> __span;
    vector<LONG> x;
    vector<int32_t> feature_remainder;

    // 纯度表
    vector<real> purities;

    vector<real> one_hot_label;
    vector<TreeNode*> sons;
    vector<map<real, real>> feature_purity_table;
    real purity=0.0f;
    bool leaf=false;
    real y=0.0f;
    LONG most_common=0;
    LONG size=0;
public:
    TreeNode(){}

    virtual ~TreeNode(){
        for (TreeNode *son: sons)
            delete son;
    }

    void reset(){
        span.clear();
        __span.clear();
        x.clear();
        feature_remainder.clear();
        purities.clear();
        for (TreeNode *son: sons)
            if (son != 0) delete son;
        sons.clear();
        leaf = false;
    }

    void resize(){
        __span.resize(feature_remainder.size());
        purities.resize(feature_remainder.size());
        feature_purity_table.resize(feature_remainder.size());
    }

    void resize(int n){
        __span.resize(n);
        purities.resize(n);
        feature_purity_table.resize(n);
    }

    void shrink_to_fit(){
        x.clear();
        x.shrink_to_fit();
        __span.clear();
        __span.shrink_to_fit();
        feature_purity_table.clear();
    }
};

struct record{
    LONG idx;
    real x;
    real y;
};

typedef struct record RECORD;

int cmp(const void *a, const void *b){
    if (((RECORD*)a)->x - ((RECORD*)b)->x >= 0) return 1;
    else return -1;
}

class DecisionTree {
protected:
    /** 根节点 **/
    TreeNode *root;
    /* 语料 */
    corpus *__corpus;
    /* 线程数目 */
    int thread_num;

public:
    DecisionTree(){}

    virtual ~DecisionTree(){
        if(root != 0) delete root;
    }

    /* 计算纯度 */
    virtual real compute_purity(TreeNode *node) = 0;

    virtual bool can_split(TreeNode *node) = 0;

    virtual void push_sons(TreeNode *node) = 0;

    void start_threads(TreeNode *node){
        size_t feature_dim = node->feature_remainder.size();

        // 获得可获得的线程数
        int thread_num = std::thread::hardware_concurrency();
        if (thread_num == 0){
            std::cerr << "No available threads!\n";
            exit(EXIT_FAILURE);
        }
        thread_num = this->thread_num < thread_num ? this->thread_num:thread_num;

        int num = (int)feature_dim / thread_num;
        thread_num = (num == 0) ? (int)feature_dim : thread_num;

        // 根据条件选择单线程
        if (node->x.size() * node->feature_remainder.size() < 200000) thread_num = 1;

        vector<vector<int32_t>> params(thread_num);
        if (num == 0) num = 1;
        int allocation = (int)feature_dim % thread_num;

        shuffle(node->feature_remainder);

        vector<thread*> threads(thread_num, 0);
        int left, right=-1;
        for (int i = 0; i < thread_num; ++i){
            left = right+1;
            right = (i == thread_num-1)?(int)feature_dim-1:right + num;
            right = i < allocation?(right+1):right;
            for (int32_t k = left; k <= right; ++k){
                params[i].push_back(k);
            }
            threads[i] = new thread([=](){select_feature_thread(node, params[i]);});
        }
        for (int i = 0; i < thread_num; ++i){
            threads[i]->join();
        }
        for (int i = 0; i < thread_num; ++i){
            delete threads[i];
        }
    }

    void select_feature(TreeNode *node){
        if (!can_split(node)) return;
        start_threads(node);
        push_sons(node);
    }

    virtual void prune() = 0;
    virtual void select_feature_thread(TreeNode* node, const vector<int32_t>& features) = 0;

    void init() {
        // 语料的数量确定大于1
        assert(__corpus->size > 0);
        root = new TreeNode();
        root->x = vector<LONG>(__corpus->size, 0);
        std::iota(root->x.begin(), root->x.end(), 0);
        root->feature_remainder = vector<int32_t>(__corpus->feature_names.size(), 0);
        std::iota(root->feature_remainder.begin(), root->feature_remainder.end(), 0);
        root->resize();
        root->subject = "ROOT";
        root->sons.clear();
        root->span.clear();
        compute_purity(root);
        for (int i = 0; i < root->feature_remainder.size(); ++i){
            root->purities[i] = root->purity;
        }
    }

    void most_common(TreeNode *node){
        LONG most_count = 0;
        node->y = __corpus->most_common(node->x, most_count);
        node->most_common = most_count;
        node->size = node->x.size();
    }

    void grow(){
//        logger.info("start to generate tree.");
        queue<TreeNode*> nodes;
        nodes.push(root);

        TreeNode *t = 0;
        size_t size, i, level=1;
        while(!nodes.empty()){
            size = nodes.size();
            for (i = 0; i < size; ++i){
                t = nodes.front();
                logger.info(std::to_string(level) + " depth - " + std::to_string(i+1) + " node.", true);
                select_feature(t);
                nodes.pop();
                for (TreeNode *son: t->sons){
                    nodes.push(son);
                }
            }
            level++;
        }
//        logger.info("tree has been generated!");
    }

    void generate(corpus& corpus1, int thread_num=8){
        __corpus = &corpus1;
        this->thread_num = thread_num;
        init();
        grow();
//        prune();
        if (DEBUG){
            print(root, 0);
        }
    }

    void print(TreeNode *t, int h){
        for (int i = 0; i < h; ++i)
            std::cout << '\t';

        string temp;
        for (size_t i = 0; i < t->span.size(); ++i){
            if (__corpus->idx_value_map.count(t->span[i]) == 0)
                temp += (std::to_string(t->span[i])) + ",";
            else temp += __corpus->idx_value_map[t->span[i]] + ",";
        }
        temp = "(" + temp.substr(0, temp.size()-1) + ")";

        std::cout << "[SUBJECT:" << t->subject << "]:" << temp
                  << "\t[PURITY:" << std::setprecision(4) << t->purity << "]" << "\t[X:" << t->x.size() << "]:";

        if (!t->x.empty()){
            for (LONG i: t->x)
                std::cout << ' ' << i;
            std::cout << '\t';
        }else std::cout << "None\t";

        std::cout << "[FEATURES:" << t->feature_remainder.size() <<  "]:";
        for (int i: t->feature_remainder)
            std::cout << ' ' << __corpus->feature_names[i];
        std::cout << '\t';

        if (t->leaf){
            std::cout << "[Y:"<< t->y << "]" ;
        }
        std::cout << std::endl;
        for(TreeNode *son: t->sons)
            print(son, h+1);
    }
};

#endif //FASTAI_DECISIONTREE_H
