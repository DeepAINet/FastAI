//
// Created by mengqy on 2019/4/17.
//

#ifndef FASTAI_XGBOOST_H
#define FASTAI_XGBOOST_H

#include "DecisionTree.h"

namespace xgboost{

struct record{
    LONG idx;
    real x;
    real g;
    real h;
};

typedef struct record RECORD;

int compare(const void *a, const void *b){
    if (((RECORD*)a)->x - ((RECORD*)b)->x >= 0) return 1;
    else return -1;
}

class cart {
    public:
        real lambda = 1.0f;
        real gamma = 1.0f;
        TreeNode *root=0;
        int thread_num;
        corpus *corp;
        bool row_sample = false;
        bool col_sample = true;
        int random_k;
        real select_feature_ratio = 0.6f;
        int max_height=200;
        int max_leaf_node_num=20;
        int leaf_node_num=0;

    public:
        cart(real lambda, real gamma, int max_height=200, int max_leaf_node_num=20)
        :lambda(lambda), gamma(gamma), max_height(max_height), max_leaf_node_num(max_leaf_node_num){}

        void update_terminate_node(TreeNode* node){
            real first=0.0f, second=0.0f;
            pair<real, real> p;
            for (LONG idx: node->x){
                p = corp->get_data(idx);
                first += p.first;
                second += p.second;
            }
            node->y = -1.0f * first / (second + lambda);
            node->leaf = true;
            leaf_node_num += 1;
        }

        real compute_purity(TreeNode *node){
            real g=0.0f, h=0.0f;
            pair<real, real> p;
            for (LONG idx: node->x){
                p = corp->get_data(idx);
                g += p.first;
                h += p.second;
            }
            return node->purity = -0.5f * (g * g)/ (h + lambda);
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
                    for (TreeNode *n: node->sons) {
                        if (0 != n) q.push(n);
                    }
                }
            }
            return nodes;
        }

        void compute_best_split_regression(TreeNode *node, int32_t fid, RECORD *records, size_t size){
            real g_sum=0.0f, h_sum=0.0f;
            for (size_t i = 0; i < size; ++i){
                g_sum += records[i].g;
                h_sum += records[i].h;
            }

            real left_g_sum = 0.0f, left_h_sum = 0.0f;
            real left, right;
            real purity;
            real max_gain = 0.0f, best_split = 0.0f;
            for (LONG i = 0; i < size - 1; ++i){
                if (i == 0 || (records[i].x == records[i-1].x)) {
                    left_g_sum += records[i].g;
                    g_sum -= records[i].g;

                    left_h_sum += records[i].h;
                    h_sum -= records[i].h;
                    continue;
                }
                left = -0.5f * left_g_sum * left_g_sum / (left_h_sum + lambda);
                right = -0.5f * g_sum * g_sum / (h_sum + lambda);
                purity = node->purity - left - right - gamma;
//                std::cout << "purity:" << purity << std::endl;
                if (max_gain < purity){
                    max_gain = purity;
                    best_split = (records[i].x + records[i-1].x)/2.0f;
                    node->feature_purity_table[fid][0] = left;
                    node->feature_purity_table[fid][1] = right;
                }
                left_g_sum += records[i].g;
                g_sum -= records[i].g;
                left_h_sum += records[i].h;
                h_sum -= records[i].h;
            }
            node->purities[fid] = max_gain;
            node->__span[fid] = best_split;
        }

        void compute_best_split(TreeNode *node, int32_t fid){
            assert(fid >= 0);
            assert(fid < node->purities.size());

            size_t size = node->x.size();
            auto *records = (RECORD *)malloc(sizeof(RECORD) * size);
            if (0 == records){
                std::cerr << "records allocate failure!";
                exit(EXIT_FAILURE);
            }
            LONG i = 0;
            pair<real, real> p;
            for (LONG idx: node->x){
                records[i].idx = idx;
                records[i].x = corp->get_x(idx, node->feature_remainder[fid]);
                p = corp->get_data(idx);
                records[i].g = p.first;
                records[i].h = p.second;
                ++i;
            }
            qsort(records, size, sizeof(RECORD), compare);

            if (records[0].x == records[size-1].x){
                node->purities[fid] = -MAXFLOAT;
                free(records);
                return;
            }

            compute_best_split_regression(node, fid, records, size);
            free(records);
        }


        void select_feature_thread(TreeNode* node, const vector<int32_t>& fids){
            for (int32_t fid: fids){
                compute_best_split(node, fid);
            }
        }

        bool can_split(TreeNode *node){
            if (node->height == max_height
                || node->x.size() <= MIN_SAMPLES
                || node->feature_remainder.empty()
                || leaf_node_num == max_leaf_node_num){
                update_terminate_node(node);
                return false;
            }
            return true;
        }


        void push_sons(TreeNode *node){
            pair<size_t, real> idx_purities = argmax(node->purities);
//        std::cout << idx_purities.second  << ":" <<  node->purity << std::endl;
            if (idx_purities.second <= 0.0f){
                update_terminate_node(node);
                return;
            }
            int32_t best_feature = node->feature_remainder[idx_purities.first];

            auto *son0 = new TreeNode();
            auto *son1 = new TreeNode();
            son0->subject = corp->feature_names[best_feature];
            son1->subject = corp->feature_names[best_feature];
            son0->height = node->height + 1;
            son1->height = node->height + 1;
            update_sons(node, son0, son1, idx_purities.first);
        }

        void update_sons(TreeNode *parent, TreeNode *son0, TreeNode *son1, int32_t fid){
            real point = parent->__span[fid];
//        std::cout << "split-point:" << point << std::endl;
            real min_value = point, max_value = point, value;
            int32_t best_feature = parent->feature_remainder[fid];
            for (LONG idx: parent->x){
                value = corp->get_x(idx, best_feature);
                if (value < point) son0->x.push_back(idx);
                else son1->x.push_back(idx);
                min_value = min(min_value, value);
                max_value = max(max_value, value);
            }

//            std::cout << "["<< son0->x.size() << ":" << son1->x.size() << "]" << std::endl;

            son0->purity = parent->feature_purity_table[fid][0];
            son1->purity = parent->feature_purity_table[fid][1];
            parent->span.push_back(min_value);
            parent->span.push_back(point);
            parent->span.push_back(max_value);
            parent->son_map[0] = son0;
            parent->son_map[1] = son1;
            parent->best_feature = best_feature;
            for (int i = 0; i < parent->feature_remainder.size(); ++i){
                if (parent->purities[i] != -MAXFLOAT) {
                    son0->feature_remainder.push_back(parent->feature_remainder[i]);
                    son1->feature_remainder.push_back(parent->feature_remainder[i]);
                }
            }

            int k = random_k < son0->feature_remainder.size() ? random_k : (int)son0->feature_remainder.size();
            son0->resize(k);

            k = random_k < son1->feature_remainder.size() ? random_k : (int)son1->feature_remainder.size();
            son1->resize(k);
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
                    if (corp->feature_types[node->best_feature]){
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


        void start_threads(TreeNode *node){
            vector<int>::iterator iter = random_k < node->feature_remainder.size()?node->feature_remainder.begin() + random_k:node->feature_remainder.end();
            vector<int> features(node->feature_remainder.begin(), iter);
            size_t feature_dim = features.size();
            int thread_num = std::thread::hardware_concurrency();
            if (thread_num == 0){
                std::cerr << "No available threads!\n";
                exit(EXIT_FAILURE);
            }
            thread_num = this->thread_num < thread_num ? this->thread_num:thread_num;

            int num = (int)feature_dim / thread_num;
            thread_num = (num == 0) ? (int)feature_dim : thread_num;
            if (node->x.size() * features.size() < 200000) thread_num = 1;
            vector<vector<int32_t>> params(thread_num);
            if (num == 0) num = 1;
            int allocation = (int)feature_dim % thread_num;

            shuffle(features);

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

        void init() {
            assert(corp->size > 0);
            root = new TreeNode();
            if (!row_sample){
                root->x = vector<LONG>(corp->size, 0);
                std::iota(root->x.begin(), root->x.end(), 0);
            }else{
                //row sample.
            }

            root->feature_remainder = vector<int32_t>(corp->feature_names.size(), 0);
            std::iota(root->feature_remainder.begin(), root->feature_remainder.end(), 0);
            if (col_sample) shuffle(root->feature_remainder);
            random_k = (int)(corp->feature_names.size() * select_feature_ratio);

            assert(random_k >= 1);
            root->resize(random_k);
            root->subject = "ROOT";
            root->sons.clear();
            root->span.clear();
            compute_purity(root);
            for (int i = 0; i < random_k; ++i){
                root->purities[i] = 0.0f;
            }
        }


        void grow(){
            queue<TreeNode*> nodes;
            if (0 != root) nodes.push(root);
            TreeNode *t = 0;
            size_t size, i, h=1;
            while(!nodes.empty()){
                size = nodes.size();
                for (i = 0; i < size; ++i){
                    t = nodes.front();
                    select_feature(t);
                    logger.info(std::to_string(h) + " depth - " + std::to_string(i) + " node", true);
                    nodes.pop();
                    for (TreeNode *son: t->sons){
                        nodes.push(son);
                    }
                }
                ++h;
            }
        }

        void generate(corpus& corpus1, int thread_num=8){
            corp = &corpus1;
            this->thread_num = thread_num;
            init();
            grow();
        }

        void draw(){
            print(root, 0);
        }


        void print(TreeNode *t, int h){
            if (0 == root || h < 0) return;
            for (int i = 0; i < h; ++i)
                std::cout << '\t';

            string temp;
            for (size_t i = 0; i < t->span.size(); ++i){
                if (corp->idx_value_map.count(t->span[i]) == 0)
                    temp += (std::to_string(t->span[i])) + ",";
                else temp += corp->idx_value_map[t->span[i]] + ",";
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
                std::cout << ' ' << corp->feature_names[i];
            std::cout << '\t';

            if (t->leaf){
                std::cout << "[Y:"<< t->y << "]" ;
            }
            std::cout << std::endl;
            for(TreeNode *son: t->sons)
                print(son, h+1);
        }
};

class XGBoost {
private:
    int num_trees=5;
    bool regression=false;
    corpus *corp;
    vector<vector<cart*>> trees;

    real epsilon = 0.1f;
    real lambda = 0.0f;
    real gamma = 0.0f;
    int max_height = 6;
    int max_leaf_node_num=100;

    tensor<real> one_hot_label;
    tensor<real> y;
    tensor<real> y_softmax;
    tensor<real> G;
    tensor<real> H;
public:
    XGBoost(){}

    ~XGBoost(){
        for (vector<cart*> k_trees: trees){
            for (cart *tree: k_trees)
                delete tree;
        }
    }

    void compute_residual_error(){
        if (!regression){
            tops::softmax(y, y_softmax);
            tensor<real> temp;
            tops::subtract(1.0f, y_softmax, temp);
            tops::multiply(y_softmax, temp, H);
            tops::subtract(y_softmax, one_hot_label, G);
        }else{

        }
    }

    void convert_to_one_hot(){
        tensor<real> d({(int)corp->size});
        real *pd = d.data();
        for (LONG i = 0; i < corp->size; ++i)
            *pd++ = corp->get_label(i);
        tops::convert_to_one_hot(d, one_hot_label, corp->num_classes);

        std::cout << one_hot_label.show() << std::endl;
        y.reshape({(int)corp->size, corp->num_classes});
        y_softmax.reshape({(int)corp->size, corp->num_classes});
        G.reshape({(int)corp->size, corp->num_classes});
        H.reshape({(int)corp->size, corp->num_classes});
        corp->buffer = std::make_shared<vector<real>>(corp->size, 0.0);
        corp->buffer->resize(corp->size);
    }

    void init(){
        if (!regression){
            convert_to_one_hot();
            compute_residual_error();
        }else {

        }
    }

    void predict_update(TreeNode *node, int bias=0){
       if (!regression){
            real *py = y.data() + bias;
            for (LONG idx: node->x){
                *(py + idx * corp->num_classes)  += epsilon * node->y;
            }
        }else{

        }
    }

    void generate_tree(int idx){
       if (!regression){
            vector<cart*> tree;
            for (int i = 0; i < corp->num_classes; ++i){
                real *pg = G.data() + i;
                real *ph = H.data() + i;
                real *py = corp->data_y->data();
                real *pb = corp->buffer->data();
                for (LONG j = 0; j < corp->size; ++j){
                    *py++ = *pg;
                    *pb++ = *ph;
                    pg += corp->num_classes;
                    ph += corp->num_classes;
                }

                cart *cart_ = new cart(lambda, gamma, max_height, max_leaf_node_num);
                logger.info(std::to_string(idx+1) + "[" + std::to_string(i+1) + "]" + " tree.");
                cart_->generate(*corp);
                vector<TreeNode*> nodes = cart_->get_terminate_nodes();
                for (TreeNode *node : nodes){
                    predict_update(node, i);
                }
//                cart_->draw();
                std::cout << std::endl;
                tree.push_back(cart_);
            }
            if (tree.size() == corp->num_classes)
                trees.push_back(tree);
        }else{

        }
    }

    void train(corpus *corp){
        this->corp = corp;
        init();
        int i = 0;
        while(i < num_trees){
            generate_tree(i);
            compute_residual_error();
            ++i;
        }
    }

    void trim(){

    }

    void predict(corpus *corp){
        if (!regression){
            tensor<real> f;
            f.reshape({(int)corp->size, corp->num_classes});
            real *pf;
            for (int i = 0; i < trees.size(); ++i){
                for (int j = 0; j < trees[i].size(); ++j){
                    pf = f.data() + j;
                    vector<real> increments = trees[i][j]->predict(corp);
                    for (int k = 0; k < corp->size; ++k){
                        *pf += epsilon * increments[k];
                        pf += corp->num_classes;
                    }
                }
            }
            int total = 0;
            tensor<real> soft;
            tops::softmax(f, soft);
            tensor<real> pre;
            tops::argmax(soft, pre);
            real *pp = pre.data();

            std::cout << pre.show() << std::endl;
            for (LONG i = 0; i < corp->size; ++i){
                std::cout << *pp << ":" << corp->get_label(i) << std::endl;
                if (*pp == corp->get_label(i)) ++total;
                ++pp;
            }
            std::cout << "\naccuracy:";
            std::cout << (real)total / (real)corp->size << std::endl;
        }
    }
};
}


#endif //FASTAI_XGBOOST_H
