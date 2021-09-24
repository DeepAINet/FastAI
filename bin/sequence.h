//
// Created by mengqy on 2018/12/15.
//

#ifndef FASTAI_SEQUENCE_H
#define FASTAI_SEQUENCE_H

#include "common.h"

template <typename T>
class sequence{
    T *__s;
    size_t _size;
public:
    sequence(INT size){
        __s = new T[size];
        _size = size;
    }

    ~sequence(){
        if(__s != 0) delete [] __s;
    }

    size_t size(){
        return _size;
    }

    void set(INT i, T t){
        assert(i < _size);
        __s[i] = t;
    }

    void get(INT i){
        assert(i < _size);
        return __s[i];
    }
};

#endif //FASTAI_SEQUENCE_H
