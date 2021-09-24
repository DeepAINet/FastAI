//
// Created by mengqy on 2019/1/7.
//


#include "shape.h"


shape::shape(const shape& c){
    __coordinates = new int[c.dims()];
    if(0 == __coordinates){
        std::cerr << "Allocate memory failure.\n";
        exit(EXIT_FAILURE);
    }
    memcpy(__coordinates, c.__coordinates, c.dims() * sizeof(int));
    __dims = c.dims();
    __size = c.size();
    if (__dims > 1) __get_bulks(__coordinates, __dims);
}

shape::shape(int *coordinates, int dims) {
    if (dims <= 0){
        std::cerr << "dims <= 0\n";
        exit(EXIT_FAILURE);
    }
    assign(coordinates, dims);
}

shape::shape(vector<int> coordinates){
    if (coordinates.size() <= 0){
        std::cerr << "dims <= 0\n";
        exit(EXIT_FAILURE);
    }
    assign(coordinates.data(), coordinates.size());
}

shape& shape::operator=(const shape& s){
    if (__coordinates != 0) {
        delete [] __coordinates;
        __coordinates = 0;
    }
    __coordinates = new int[s.__dims];
    if (0 == __coordinates){
        std::cerr << "Allocate memory failure.\n";
        exit(EXIT_FAILURE);
    }
    memcpy(__coordinates, s.__coordinates, sizeof(int) * s.dims());

    __dims = s.dims();
    __size = s.size();
    if (__dims > 1)
        __get_bulks(__coordinates, __dims);
    return *this;
}

shape& shape::operator=(vector<int> s){
    if (__coordinates != 0) {
        delete [] __coordinates;
        __coordinates = 0;
    }
    __coordinates = new int[s.size()];
    if (0 == __coordinates){
        std::cerr << "Allocate memory failure.\n";
        exit(EXIT_FAILURE);
    }
    memcpy(__coordinates, s.data(), sizeof(int) * s.size());

    __dims = s.size();
    __size = 1;
    for (int n: s) __size *= n;
    if (__dims > 1)
        __get_bulks(__coordinates, __dims);
    return *this;
}

bool shape::operator==(const shape& s){
    if (__dims != s.dims())
        return false;
    for (int i = 0; i < __dims; ++i){
        if (__coordinates[i] != s[i]) return false;
    }
    return true;
}

bool shape::operator==(vector<int>& s){
    if (__dims != s.size())
        return false;
    for (int i = 0; i < __dims; ++i){
        if (__coordinates[i] != s[i]) return false;
    }
    return true;
}


bool shape::operator!=(const shape& s){
    return !(*this == s);
}

bool shape::operator!=(vector<int>& s){
    return !(*this == s);
}

/**
 * In order to get (or set) each element effectively,
 * compute the size of each bulk from the second dimension.
 * Note that the dims must greater or equal 2.
 * @param coordinates: values of dimensions.
 * @param dims: the length of dimensions.
 */
void shape::__get_bulks(int *coordinates, int dims){
    if (0 != bulks){
        delete [] bulks;
        bulks = 0;
    }

    bulks = new LINT[dims];
    if (0 == bulks){
        std::cerr << "Allocate memory failure.\n";
        exit(EXIT_FAILURE);
    }

    bulks[dims-1] = 1;
    for (int i = dims-2; i >= 0; --i){
        bulks[i] = bulks[i+1] * coordinates[i+1];
    }
}

/**
 * Initialize values of dimensions.
 * @param coordinates: values of dimensions.
 * @param len: the length of dimensions.
 */
void shape::assign(int *coordinates, int len){
    if (len <= 0){
        std::cerr << "dims <= 0.\n";
        exit(EXIT_FAILURE);
    }
    if (0 != __coordinates){
        delete [] __coordinates;
        __coordinates = 0;
    }
    __coordinates = new int[len];
    if (0 == __coordinates){
        std::cerr << "Allocate memory failure.\n";
        exit(EXIT_FAILURE);
    }
    memcpy(__coordinates, coordinates, len * sizeof(int));

    for (int i = 0; i < len; ++i){
        if (coordinates[i] < 1){
            std::cerr << "coordinate should not be less than 1.\n";
            exit(EXIT_FAILURE);
        }
    }
    __dims = len;
    __size = 1;
    for (int i = 0; i < __dims; ++i)
        __size *= __coordinates[i];

    if (__dims >= 2) __get_bulks(coordinates, __dims);
}

/**
 * Get a shape which reduces the first dimension.
 * @return a shape which reduces the first dimension.
 */
shape shape::reduce(){
    if (__dims <= 1){
        std::cerr << "dims <= 1, can't reduce to a smaller dim tensor!\n";
        exit(EXIT_FAILURE);
    }
    shape reduced_shape(__coordinates + 1, __dims-1);
    return reduced_shape;
}

string shape::to_str(){
    string str="[shape=";
    if (__dims == 1)
        str += "(," + std::to_string(__coordinates[0]) + ")";
    else{
        str += "(";
        for (int i = 0; i < __dims - 1; ++i){
            str += std::to_string(__coordinates[i]) + ",";
        }
        str += std::to_string(__coordinates[__dims-1]);
        str += ")";
    }

    str += ", dims=" + std::to_string(__dims) + ", size=" + std::to_string(__size) + "]";

    return str;
}

void shape::save(ofstream& out){
    out.write((char *)&__dims, sizeof(__dims));
    for (int i = 0; i < __dims; ++i)
        out.write((char*)&(__coordinates[i]), sizeof(int));
}

void shape::load(ifstream& in){
    in.read((char*)&__dims, sizeof(int));
    if (__coordinates != 0) delete [] __coordinates;
    __coordinates = new int[__dims];
    for (int i = 0; i < __dims; ++i)
        in.read((char*)&__coordinates[i], sizeof(int));
    assign(__coordinates, __dims);
}



