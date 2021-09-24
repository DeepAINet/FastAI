//
// Created by mengqy on 2019/1/21.
//

#ifndef FASTAI_INPUT_H
#define FASTAI_INPUT_H

#include <cstdio>
#include <cmath>
#include "dictionary.h"
#include "common.h"
#include "tensor.h"
#include "util.h"


int __idx = 0;
unsigned long long next_random = 1;

bool can_discard(dictionary& d, LONG idx){
    double ran = (sqrt(d.count(idx) / d.total()) + 1) * d.total() / d.count(idx);
    next_random = next_random * (unsigned long long)25214903917 + 11;
    if (ran < (next_random & 0xFFFF) / (float)65536) return true;
    return false;
}

/**
 *
 * @param fid
 * @param d
 * @param x
 * @param y
 * @param context
 * @param context_size
 */
void skip_gram_input_fn(FILE *fid,
                        dictionary& d,
                        tensor<LONG>& x,
                        tensor<LONG>& y,
                        LONG *context=0,
                        int context_size=5){
    if (context_size < 1) {
        logger.error("context size < 1");
        exit(EXIT_FAILURE);
    }
    assert(x.get_shape()[0] % 2 == 0);

    string __word;
    int a = 0, i = 0;
    bool __new_allocated = false;
    if (0 == context) {
        context = new LONG[context_size];
        __new_allocated = true;
    }

    x.reset();
    y.reset();
    LONG __center_word;
    int __batch_idx = 0;
    while(!feof(fid)){
        a = read_word(__word, fid);
        if (0 == a){
            if (feof(fid)) break;
            __idx = 0;
            continue;
        }

        __center_word = d[__word];
        if (-1 == __center_word) continue;
        if (can_discard(d, __center_word)) continue;
        for (i = __idx-1; i >= (__idx - context_size > 0 ? (__idx - context_size) : 0); --i){
            x.set(context[i % context_size]);
            y.set(__center_word);
            x.set(__center_word);
            y.set(context[i % context_size]);
            __batch_idx += 2;
            if (__batch_idx == x.get_shape()[0]) break;
        }
        context[__idx % context_size] = __center_word;
        ++__idx;
        if (__batch_idx == x.get_shape()[0]) break;
    }
    if (__new_allocated) delete [] context;
}

/**
 *
 * @param fid
 * @param d
 * @param x
 * @param y
 * @param context
 * @param context_size
 */
void cbow_input_fn(FILE *fid,
                   dictionary& d,
                   tensor<LONG>& x,
                   tensor<LONG>& y,
                   LONG *context=0,
                   int context_size=2){
    string __word;
    int a = 0;
    LONG i = 0;
    bool __new_allocated = false;
    int length = (context_size << 1) + 1;
    if (0 == context) {
        context = new LONG[length];
        __new_allocated = true;
    }

    x.reset();
    y.reset();
    LONG __center_word;
    int __batch_idx = 0;
    while(!feof(fid)){
        a = read_word(__word, fid);
        if (0 == a){
            if (feof(fid)) break;
            __idx = 0;
            continue;
        }

        __center_word = d[__word];
        if (-1 == __center_word) continue;
        if (can_discard(d, __center_word)) continue;
        context[__idx % context_size] = __center_word;
        ++__idx;
        __center_word = __idx - context_size - 1;
        if (__center_word < 0) continue;
        for (i = __center_word - context_size; i <= __center_word + context_size; ++i){
            if (i < 0){
                x.set(-1);
                continue;
            }
            if (i == __center_word) continue;
            x.set(context[i % length]);
        }
        y.set(context[__center_word % length]);
        ++__batch_idx;
        if (__batch_idx == x.get_shape()[0]) break;
    }
    if (__new_allocated) delete [] context;
}

#endif //FASTAI_INPUT_H
