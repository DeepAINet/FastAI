//
// Created by mengqy on 2019/4/11.
//

#ifndef FASTAI_GBDT_H
#define FASTAI_GBDT_H

#include "corpus.h"
#include "base_trees.h"

class GBDT {
private:
    int num_trees=5;
    bool regression=false;
    corpus *corp;
    vector<BaseCart*> binary_trees;
    vector<vector<BaseCart*>> multi_trees;

    tensor<real> data_one_hot_label;
    tensor<real> y;
    tensor<real> data_predict;
    tensor<real> one_hot_error;
    real init_predict=0.0f;
    real epsilon = 0.1f;

public:
    GBDT(){}

    ~GBDT(){
        for (BaseCart *cart1: binary_trees)
            delete cart1;
        for (vector<BaseCart*> k_trees: multi_trees){
            for (BaseCart *tree: k_trees)
                delete tree;
        }
    }

    void compute_residual_error(){
        if (corp->binary_class){
            real *pdp = data_predict.data();
            real *py = y.data();
            for (LONG i = 0; i < this->corp->size; ++i){
//                std::cout << *pdp << '\t';
                (*this->corp->data_y)[i] = (2 * (*py)) /(1 + exp(2 * (*py) * (*pdp++)));
//                std::cout << (*this->corp->data_y)[i] << std::endl;
                ++py;
            }

        } else if (corp->multi_class){
            tops::softmax(y, data_predict);
            tops::subtract(data_one_hot_label, data_predict, one_hot_error);
            std::cout << one_hot_error.show() << std::endl;
        } else {

        }
    }

    void convert_to_one_hot(){
        tensor<real> d({(int)corp->size});
        real *pd = d.data();
        for (LONG i = 0; i < corp->size; ++i)
            *pd++ = corp->get_label(i);
        tops::convert_to_one_hot(d, data_one_hot_label, corp->num_classes);
        y.reshape({(int)corp->size, corp->num_classes});
        data_predict.reshape({(int)corp->size, corp->num_classes});
        one_hot_error.reshape({(int)corp->size, corp->num_classes});
    }

    void init(){
        if (this->corp->binary_class) {
            real avg = 0.0f;
            for (LONG i = 0; i < this->corp->size; ++i) {
                avg += this->corp->get_label(i);
            }
            avg /= (real)this->corp->size;
            avg = 0.5f * std::log((1+avg)/(1-avg));
            init_predict = avg;
            real tmp;
            y.reshape({(int)corp->size, 1});
            data_predict.reshape({(int)corp->size, 1});
            real *py = y.data(), *pdp = data_predict.data();
            for (LONG i = 0; i < this->corp->size; ++i) {
                tmp = this->corp->get_label(i);
                *py++ = tmp;
                *pdp++ = avg;
            }
            logger.info("初始值:" + std::to_string(avg));

            pdp = data_predict.data();
            py = y.data();
            for (LONG i = 0; i < this->corp->size; ++i){
                (*this->corp->data_y)[i] = (2 * (*py)) /(1 + exp(2 * (*py) * (*pdp++)));
                if (i <= 100) {
                    std::cout << *py << ":" << this->corp->get_label(i) << std::endl;
                }
                ++py;
            }
        }else if (this->corp->multi_class){
            // 计算one_hot_label - softmax
            convert_to_one_hot();
            compute_residual_error();
        }else {

        }
    }

    void update_terminate_nodes(TreeNode *node){
        if (corp->binary_class){
            real sum = 0.0f, var = 0.0f;
            for (LONG idx: node->x){
                sum += corp->get_label(idx);
                var += abs(corp->get_label(idx)) * (2.0f - abs(corp->get_label(idx)));
            }
            node->y = sum / (var + 0.000000001f);
        } else if (corp->multi_class){
            real sum = 0.0f, var = 0.0000001f;
            for (LONG idx: node->x){
                sum += corp->get_label(idx);
                var += abs(corp->get_label(idx)) * (1.0f - abs(corp->get_label(idx)));
            }
            node->y = (sum / var) * (1.0f - 1.0f/(real)corp->num_classes);
        } else {
            real sum = 0.0f;
            for (LONG idx: node->x){
                sum += corp->get_label(idx);
            }
            node->y = sum / (real)node->x.size();
        }
    }

    void predict_update(TreeNode *node, int bias=0){
        if (corp->binary_class){
            real *pdp = data_predict.data();
            for (LONG idx: node->x){
                *(pdp + idx) += epsilon * node->y;
            }
        }else if (corp->multi_class){
            real *py = y.data() + bias;
            for (LONG idx: node->x){
                *(py + idx * corp->num_classes)  += node->y;
            }
        }else{

        }
    }

    bool generate_tree(int idx){
        if (corp->binary_class){
            BaseCart *cart_ = new BaseCart();
            cart_->generate(*corp, 1);
            vector<TreeNode*> nodes = cart_->get_terminate_nodes();
            for (TreeNode *node : nodes){
                update_terminate_nodes(node);
                predict_update(node);
            }

            if (nodes.size() == 1) return false;
            binary_trees.push_back(cart_);
        }else if (corp->multi_class){
            vector<BaseCart*> tree;
            for (int i = 0; i < corp->num_classes; ++i){
                //
                real *po = one_hot_error.data() + i;
                real *py = corp->data_y->data();


                for (LONG j = 0; j < corp->size; ++j){
                    *py++ = *po;
                    po += corp->num_classes;
                }

                BaseCart *cart_ = new BaseCart();
                logger.info(std::to_string(idx+1) + "[" + std::to_string(i+1) + "] tree.");
                cart_->generate(*corp);
                vector<TreeNode*> nodes = cart_->get_terminate_nodes();
                for (TreeNode *node : nodes){
                    update_terminate_nodes(node);
                    predict_update(node, i);
                }
                tree.push_back(cart_);
            }
            if (tree.size() == corp->num_classes)
                multi_trees.push_back(tree);
        }else{

        }
        return true;
    }

    void train(corpus *corp){
        this->corp = corp;
        init();
        int i = 0;
        bool con;
        while(i < num_trees){
            con = generate_tree(i);
            if (!con) break;
            compute_residual_error();
            ++i;
        }
    }

    void trim(){

    }

    void predict(corpus *corp){
        if (corp->binary_class){
            int total = 0;
            LONG size = corp->size;
            real *py = corp->data_y->data();

            vector<real> f(corp->size, init_predict);
            for (BaseCart *cart1: binary_trees){
                vector<real> increments = cart1->predict(corp);
                for (int i = 0; i < size; ++i){
                    f[i] += epsilon * increments[i];
                }
            }

            for (int i = 0; i < size; ++i){
                f[i] = 1.0f / (1.0f + std::exp(-2 * f[i]));
                if ((f[i] >= 0.5f && (*py) == 1.0f) ||
                    (f[i] < 0.5f && (*py) == -1.0f)){
                    ++total;
                }
                ++py;
            }
            std::cout << "\naccuracy:";
            std::cout << (real)total / (real)size << std::endl;
        }else if (corp->multi_class){
            std::cout << "multi_class\n";
            tensor<real> f;
            f.reshape({(int)corp->size, corp->num_classes});
            real *pf;
            for (int i = 0; i < multi_trees.size(); ++i){
                for (int j = 0; j < multi_trees[i].size(); ++j){
                    pf = f.data() + j;
                    vector<real> increments = multi_trees[i][j]->predict(corp);
                    for (int k = 0; k < corp->size; ++k){
                        *pf += increments[k];
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
#endif //FASTAI_GBDT_H
