//
// Created by mengqy on 2019/1/12.
//

#ifndef FASTAI_WORD2VEC_H
#define FASTAI_WORD2VEC_H

#include <sys/stat.h>
#include "dictionary.h"
#include "tensor.h"
#include "rand.h"
#include "constant.h"
#include "common.h"
#include "sampler.h"

class vocab{
public:
    int length = 0;
    LONG *path = 0;
    bool *codes = 0;
    LONG trained_count = 0;

public:
    vocab(){}
    vocab(int len){
        length = len;
        path = new LONG[length];
        codes = new bool[length];
    }

    void resize(int len){
        if (0 != path) {
            delete [] path;
            path = 0;
        }
        if (0 != codes) {
            delete [] codes;
            codes = 0;
        }
        length = len;
        path = new LONG[len];
        codes = new bool[len];
    }

    void assign(LONG *p, bool *c){
        for (int i = 0; i < length; ++i){
            path[i] = p[i];
            codes[i] = c[i];
        }
    }

    ~vocab(){
        if (0 != path) {
            delete [] path;
            path = 0;
        }
        if (0 != codes) {
            delete [] codes;
            codes = 0;
        }
    }
};


class word2vec {
public:
    dictionary __dict;
    int __context_size;
    int __model;
    int __loss_fn;
    int __embedding_dim;
    int __train_steps;
    int __every_steps;
    int __epoch_num;
    int __min_count;
    int __batch_size;

    tensor<real> __embeddings;
    tensor<real> __coefficients;

    vocab *__vocabs=0;
    EnergySampler __es;

public:
    word2vec(){}

    void train(const char *corpus_filename,
               const char *model_dir,
               int embedding_dim=128,
               int loss_fn=0,
               int model=0,
               int window_size=5,
               int epoch_num=1,
               int min_count=1,
               int batch_size=128,
               int train_steps=10000,
               int every_steps=10000);

    void initialize();

    void save_model();

    void construct_dict(const char *corpus_filename, const char *model_dir);

    void create_huffman_tree();



//    void optimize();

};

void word2vec::train(const char *corpus_filename,
                     const char *model_dir,
                     int embedding_dim,
                     int loss_fn,
                     int model,
                     int window_size,
                     int epoch_num,
                     int min_count,
                     int batch_size,
                     int train_steps,
                     int every_steps){

    __embedding_dim = embedding_dim;
    __context_size = window_size;
    __epoch_num = epoch_num;
    __min_count = min_count;
    __batch_size = batch_size;
    __loss_fn = loss_fn;
    __model = model;
    __train_steps = train_steps;
    __every_steps = every_steps;

    std::cout << "[word2vec-model-parameters]:"
              << " embedding_dim:" << __embedding_dim
              << ", context_size:" << __context_size
              << ", epoch_num:" << __epoch_num
              << ", loss_fn:" << __loss_fn
              << ", train_steps:" << __train_steps
              << ", every_steps:" << __every_steps << std::endl;

    construct_dict(corpus_filename, model_dir);

    initialize();
    logger.info("initialize parameters.");

    if (__loss_fn == 0){
        create_huffman_tree();
    }else if(__loss_fn == 1){
        __es.create(__dict);
    }else{
        logger.error("loss function should be 0 or 1!");
        exit(EXIT_FAILURE);
    }

//    optimize();

}

void word2vec::construct_dict(const char *corpus_filename, const char *model_dir){
    clock_t start_time = clock();
    char temp[LENGTH];
    mkdir(model_dir, 0755);
    __dict.construct_dict(corpus_filename, __min_count);
    clock_t end_time = clock();
    sprintf(temp, "%s/%s", model_dir, "words.txt");
    __dict.save(temp);

    sprintf(temp, "construct dictionary with its size is %ld, and save words to %s/%s, exhaust %ld s.", __dict.size(), model_dir, "words.txt", (end_time-start_time)/CLOCKS_PER_SEC);
    logger.info(temp);
}

void word2vec::initialize(){
    int s[2] = {(int)(__dict.size()), __embedding_dim};
    shape __shape(s, 2);
    __embeddings.reshape(__shape);
    uniform(__embeddings, (float)(-0.5/sqrt(__embedding_dim)), (float)(0.5/sqrt(__embedding_dim)));
    __coefficients.reshape(__shape);
}

void word2vec::create_huffman_tree() {
    clock_t start_time = clock();
    LONG __dict_size = (LONG)__dict.size();
    LONG __len = (__dict_size << 1) - 1;
    LONG *__counts = 0, *__paths = 0;
    bool *__codes = 0;
    __counts = new LONG[__len];
    __paths = new LONG[__len];
    __codes = new bool[__len];
    __vocabs = new vocab[__dict_size];
    if (0 == __codes || 0 == __paths || 0 == __codes || 0 == __vocabs){
        logger.error("allocate memory failure.\n");
        exit(EXIT_FAILURE);
    }

    for (LONG i = 0; i < __dict_size; ++i){
        __counts[i] = __dict.count(i);
        __codes[i] = false;
    }
    for (LONG i = __dict_size; i < __len; ++i){
        __counts[i] = (LONG)1e15;
        __codes[i] = false;
    }

    LONG __left = __dict_size-1, __right = __dict_size;
    LONG __first, __second;
    for (LONG i = 0; i < __dict_size - 1; ++i){
        if (__left >= 0){
            if (__counts[__left] < __counts[__right]) __first = __left--;
            else __first = __right++;
        }else __first = __right++;
        if (__left >= 0){
            if (__counts[__left] < __counts[__right]) __second = __left--;
            else __second = __right++;
        }else __second = __right++;
        __counts[__dict_size + i] = __counts[__first] + __counts[__second];
        __paths[__first] = __dict_size + i;
        __paths[__second] = __dict_size + i;
        __codes[__second] = true;
    }

    LONG p = 0, j = 0;
    bool __code[LENGTH];
    LONG __parent[LENGTH];
    int __code_idx = 0;
    for (LONG i = 0; i < __dict_size; ++i){
        __code_idx = LENGTH;
        p = i;
        while(p < __len){
            --__code_idx;
            __code[__code_idx] = __codes[p];
            p = __paths[p];
            __parent[__code_idx] = p;
        }

        for (j = __code_idx; j < LENGTH; ++j)
            __parent[j] = (__len-1) - __parent[j];
        __vocabs[i].resize(LENGTH - __code_idx);
        __vocabs[i].assign(__parent + __code_idx, __code + __code_idx);
    }

    delete __counts;
    delete __paths;
    delete __codes;
    clock_t end_time = clock();

    char temp[LENGTH];
    sprintf(temp, "huffman tree has been created successfully. exhaust %ld s.", (end_time-start_time)/CLOCKS_PER_SEC);
    logger.info(temp);
}


#endif //FASTAI_WORD2VEC_H
