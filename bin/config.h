//
// Created by mengqy on 2019/3/6.
//

#ifndef FASTAI_CONFIG_H
#define FASTAI_CONFIG_H

#include <map>
#include <string>
#include <sstream>

using namespace std;

template <typename T1, typename T2>
class config{
private:
    map<T1, T2> __config;

public:
    config(){}

    T2& operator[](T1 key) {
        return __config[key];
    }

    config& operator=(map<T1, T2> key_values){
        __config = key_values;
        return *this;
    }

    string to_str(){
        stringstream ss;
        for (auto iter = __config.begin(); iter != __config.end(); ++iter){
            ss << iter->first << "\t" << iter->second << "\n";
        }
        return ss.str();
    }
};

#endif //FASTAI_CONFIG_H
