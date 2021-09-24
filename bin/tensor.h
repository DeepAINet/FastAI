//
// Created by mengqy on 2019/1/7.
//

#ifndef FASTAI_TENSOR_H
#define FASTAI_TENSOR_H

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include "shape.h"
#include "common.h"

#define ALIGN_SIZE 128

using namespace std;

template <typename T>
class tensor {
public:
    string name;
private:
    shape __shape;
    T *__elements = 0;
    LONG __idx = 0;

    bool __to_deleted = true;
//    tensor<T> *__sub_tensor;

public:
    tensor(T* elements, const shape& shape)
    :__shape(shape),__elements(elements), __to_deleted(false){}

public:
    tensor(){}

    tensor(int len, T t){
        int a = posix_memalign((void **)&__elements, ALIGN_SIZE, len * sizeof(T));

        if (a || 0 == __elements){
            std::cerr << "Allocate memory failure.\n";
            exit(EXIT_FAILURE);
        }
        int temp[1] = {len};
        __shape.assign(temp, 1);
        memset(__elements, 0, len);
    }

    tensor(const shape& shape):__shape(shape){
        int a = posix_memalign((void **)&__elements, ALIGN_SIZE, __shape.size() * sizeof(T));
        if (a || 0 == __elements){
            std::cerr << "Allocate memory failure\n";
            exit(EXIT_FAILURE);
        }
        memset(__elements, 0, __shape.size());
    }

    tensor(const tensor<T>& copy){
        int a = posix_memalign((void **)&__elements, ALIGN_SIZE, copy.get_shape().size() * sizeof(T));
        if (a || 0 == __elements){
            std::cerr << "Allocate memory failure.\n";
            exit(EXIT_FAILURE);
        }
        memcpy(__elements, copy.__elements, sizeof(T) * copy.get_shape().size());
        __shape = copy.get_shape();
    }

    tensor(vector<int> s):__shape(s){
        int a = posix_memalign((void **)&__elements, ALIGN_SIZE, __shape.size() * sizeof(T));
        if (a || 0 == __elements){
            std::cerr << "Allocate memory failure\n";
            exit(EXIT_FAILURE);
        }
        memset(__elements, 0, __shape.size());
    }

    const shape& get_shape() const;

    tensor<T>& operator=(const tensor<T>& tensor1);

    tensor<T> operator[](int idx);

    ~tensor(){
        if (__to_deleted && __elements != 0) {
            free(__elements);
            __elements = 0;
        }
    }

    void init(T t);
    void reshape(const shape& s);
    void reshape(vector<int> v);
    size_t size() const;
    T& element();
    void set(T t);
    string to_str();
    string show();
    string type_name();
    T * data() const;
    bool is_scalar();
    bool empty() const;
    void reset(LONG idx=0);
    void flat();
    void save(ofstream &out);
    void load(ifstream &in);
};

template <typename T>
inline bool tensor<T>::empty() const{
    return 0 == __elements;
}
template <typename T>
inline const shape& tensor<T>::get_shape() const{
    return __shape;
}

template <typename T>
inline size_t tensor<T>::size() const{
    return __shape.size();
}

template <typename T>
inline T * tensor<T>::data() const{
    return __elements;
}

template <typename T>
inline bool tensor<T>::is_scalar(){
    return __shape.dims() == 1 && __shape[0] == 1;
}

template <typename T>
inline void tensor<T>::flat(){
    int *a = new int[1];
    a[0] = __shape.size();
    shape shape1(a, 1);
    __shape = shape1;
}

template <typename T>
void tensor<T>::set(T t){
    assert(__idx < __shape.size());
    __elements[__idx++] = t;
}

template <typename T>
string tensor<T>::type_name() {
    if (typeid(T) == typeid(float))
        return "float";
    if (typeid(int64_t) == typeid(T))
        return "int64_t";
    if (typeid(int32_t) == typeid(T))
        return "int32_t";
    if (typeid(int8_t) == typeid(T))
        return "int8_t";

    return "others";
}

template <typename T>
tensor<T> tensor<T>::operator[](int idx){
    assert(idx < __shape[0]);
    shape __new_shape(__shape.reduce());
    return tensor<T>(__elements + __new_shape.size() * idx, __new_shape);
}

template <typename T>
string tensor<T>::to_str(){
    string __s = __shape.to_str();
    __s = __s.substr(0, __s.size()-1) + ", dtype=" + type_name() + "]";
    return __s;
}

template <typename T>
string tensor<T>::show(){
    string __s = "[";
    if (__shape.dims() == 1){
        for (int i = 0; i < __shape[0] - 1; ++i){
            __s += std::to_string(__elements[i]) + ", ";
        }
        __s += std::to_string(__elements[__shape[0] - 1]) + "]";
        return __s;
    }
    int __first_dim = __shape[0];
    for (int i = 0; i < __first_dim - 1; ++i){
        __s += (*this)[i].show() + ", \n";
    }

    __s += (*this)[__first_dim - 1].show() + "]";
    return __s;
}

template <typename T>
tensor<T>& tensor<T>::operator=(const tensor<T>& assigned_tensor){
    if (__to_deleted) {
        if (__elements != 0){
            delete [] __elements;
            __elements = 0;
        }

        __elements = new T[assigned_tensor.get_shape().size()];
        if (0 == __elements){
            std::cerr << "Allocate memory failure.\n";
            exit(EXIT_FAILURE);
        }
        memcpy(__elements, assigned_tensor.__elements, assigned_tensor.get_shape().size() * sizeof(T));
        __shape = assigned_tensor.get_shape();
    }

    if (__elements != 0 && !__to_deleted){
        if (__shape != assigned_tensor.get_shape()){
            std::cerr << "__shape != assigned_tensor.__shape!";
            exit(EXIT_FAILURE);
        }
        memcpy(__elements, assigned_tensor.__elements, assigned_tensor.get_shape().size() * sizeof(T));
    }

    return *this;
}

//template <typename T>
//T& tensor<T>::get(...){
//    va_list ap;
//    __idx = 0;
//    int __coordinate_idx = 0;
//    va_start(ap, __coordinate_idx);
//    for (int i = 0; i < __shape.dims()-1; ++i) {
//        __coordinate_idx = va_arg(ap, int);
//        assert(__coordinate_idx < __shape[i]);
//        __idx += __coordinate_idx * __shape.bulks[i];
//    }
//    __idx += va_arg(ap, int);
//    va_end(ap);
//
//    return __elements[__idx];
//}

template <typename T>
T& tensor<T>::element(){
    assert(__idx < __shape.size());
    return __elements[__idx++];
}

template <typename T>
void tensor<T>::reshape(const shape& s){
    if (__shape == s) return;

    if (__shape.size() == s.size()){
        __shape = s;
        return;
    }

    LONG size = s.size();
    if (0 != __elements) {
        free(__elements);
        __elements = 0;
    }
    int a = posix_memalign((void **)&__elements, ALIGN_SIZE, s.size() * sizeof(T));
    if (0 == __elements || a){
        std::cerr << "Allocate memory failure\n";
        exit(EXIT_FAILURE);
    }
    for (LONG i = 0; i < size; ++i)
        __elements[i] = 0;
    __shape = s;
}

template <typename T>
void tensor<T>::reshape(vector<int> s){
    if (__shape == s) return;

    LONG size = 1;
    for (int n: s) size *= n;
    if (__shape.size() == size){
        __shape = s;
        return;
    }

    if (0 != __elements) {
        free(__elements);
        __elements = 0;
    }
    int a = posix_memalign((void **)&__elements, ALIGN_SIZE, size * sizeof(T));
    if (0 == __elements || a){
        std::cerr << "Allocate memory failure\n";
        exit(EXIT_FAILURE);
    }
    for (LONG i = 0; i < size; ++i)
        __elements[i] = 0;
    __shape = s;
}

template <typename T>
void tensor<T>::init(T t){
    for (LONG i = 0; i < size(); ++i)
        __elements[i] = t;
}

template<typename T>
inline void tensor<T>::reset(LONG idx) {
    __idx = idx;
}

template <typename T>
void tensor<T>::save(ofstream &out){
    __shape.save(out);
    for (LONG i = 0; i < __shape.size(); ++i)
        out.write((char *)&__elements[i], sizeof(T));
}

template <typename T>
void tensor<T>::load(ifstream &in){
    __shape.load(in);
    if (0 != __elements) delete [] __elements;
    __elements = new T[__shape.size()];

    for (LONG i = 0; i < __shape.size(); ++i)
        in.read((char *)&__elements[i], sizeof(T));
}

tensor<real> DEFAULT_FLOAT_TENSOR;
tensor<LONG> DEFAULT_LONG_TENSOR;

#endif //FASTAI_TENSOR_H
