//
// Created by mengqy on 2019/1/7.
//

#ifndef FASTAI_SHAPE_H
#define FASTAI_SHAPE_H

#include <iostream>
#include <vector>
#include <fstream>

using namespace std;
typedef long long LINT;

class shape {
public:
    LINT *bulks = 0;
private:
    int *__coordinates = 0;
    int __dims = 0;
    LINT __size = 0;

public:
    shape(){}
    shape(int *coordinates, int dims);
    shape(vector<int> coordinates);
    shape(const shape& c);

    ~shape(){
        if (__coordinates != 0) {
            delete [] __coordinates;
            __coordinates = 0;
        }

        if (bulks != 0) {
            delete [] bulks;
            bulks = 0;
        }
    }

    void __get_bulks(int *coordinates, int dims);
    shape& operator=(const shape& s);
    shape& operator=(vector<int> s);
    bool operator==(const shape& s);
    bool operator==(vector<int>& s);
    bool operator!=(const shape& s);
    bool operator!=(vector<int>& s);
    void assign(int *shapes, int len);
    int operator[](int coordinate_idx) const;
    int dims() const;
    LINT size() const;
    shape reduce();
    string to_str();
    bool empty() const;
    void save(ofstream& out);
    void load(ifstream& in);

    static bool same_prefix_shape(const shape& a, const shape& b){
        if (a.empty() || b.empty()) return false;
        int __a_idx = 0;
        int __b_idx = 0;

        while((__a_idx != a.dims() - 1) && (__b_idx != b.dims() - 1)){
            if (a[__a_idx++] != b[__b_idx++]) return false;
        }

        if (__a_idx == a.dims() - 1 && __b_idx == b.dims() - 1)
            return (a[__a_idx] == 1 || b[__b_idx] == 1 || a[__a_idx] == b[__b_idx]);

        if (__a_idx == a.dims() - 1) return a[__a_idx] == 1 || a[__a_idx] == b[__b_idx];

        return b[__b_idx] == 1 || b[__b_idx] == a[__a_idx];
    }

    static bool same_suffix_shape(const shape& a, const shape& b){
        if (a.empty() || b.empty()) return false;
        int __a_idx = a.dims();
        int __b_idx = b.dims();

        __a_idx -= 1;
        __b_idx -= 1;
        while((__a_idx != 0) && (__b_idx != 0)){
            if (a[__a_idx--] != b[__b_idx--]) return false;
        }

        if (__a_idx == 0 && __b_idx == 0)
            return (a[0] == 1 || b[0] == 1 || a[0] == b[0]);

        if (__a_idx == 0) return a[0] == 1 || a[0] == b[__b_idx];

        return b[0] == 1 || b[0] == a[__a_idx];
    }
};

inline int shape::operator[](int coordinate_idx) const{
    assert(coordinate_idx < __dims);
    return __coordinates[coordinate_idx];
}

inline int shape::dims() const{
    return __dims;
}

inline LINT shape::size() const{
    return __size;
}

inline bool shape::empty() const{
    return __dims == 0 && __size == 0;
}






#endif //FASTAI_SHAPE_H
