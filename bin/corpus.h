//
// Created by mengqy on 2018/11/23.
//

#ifndef FASTAI_CORPUS_H
#define FASTAI_CORPUS_H

#include <fstream>
#include <vector>
#include <set>
#include <iostream>
#include <memory>
#include <map>
#include "common.h"
#include "util.h"
#include "global.h"
#include "rand.h"

#define SPLIT_RATE 0.2

using namespace std;

map<real, real> label_dict;
map<real, real> label_reverse_dict;

class corpus{
public:
    // 语料长度
    LONG size;
    // 特征维度
    int x_dim;
    // 类别总数
    int num_classes=1;

    // 是否为二分类
    bool binary_class = false;
    // 是否为多分类
//    bool multi_class = false;
    // 是否为回归
    bool regression = false;
    // one hot label.
    bool zero_one_label = false;

    ifstream in;
    // 特征名称列表
    vector<string> feature_names;
    // 连续或者离散的特征
    vector<bool> feature_types;

    map<LONG, string> idx_value_map;
    map<string, LONG> value_idx_map;

    std::shared_ptr<vector<real>> data_x;
    std::shared_ptr<vector<real>> data_y;

    std::shared_ptr<vector<real>> buffer;

    real label_idx = -1.0f;

    real *x=0;
    real *y=0;

    // 批量大小
    int32_t batch_size=128;
    int32_t current_batch_size=0;
    LONG __records_idx = 0;

    string filename;

public:
    corpus(const char *file, int batch_size=128, bool shuffled=true, bool zero_one_label=true)
    :in(file), batch_size(batch_size), filename(file), zero_one_label(zero_one_label){
        __get_basic_info();
        global_batch_size = batch_size;
        total_steps = (size / global_batch_size + 1) * epoch_num;
        read(shuffled);
    }

    corpus(string file, int batch_size=128, bool shuffled=true,  bool zero_one_label=true)
    :in(file), batch_size(batch_size), filename(file), zero_one_label(zero_one_label){
        __get_basic_info();
        global_batch_size = batch_size;
        total_steps = size / global_batch_size + 1;
        read(shuffled);
    }

    corpus(){}

    ~corpus(){
        if (in.is_open()) {
            in.close();
            if (0 == remove(filename.c_str())){
                logger.info("remove file named " + filename);
            }
        }
    }

    void close(){
        if (in.is_open()) in.close();
    }

    void open(const char *file){
        if (in.is_open()) in.close();
        in.open(file);
        if (!in.is_open()){
            std::cerr << "Not found file named " << file << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    void open(string file){
        open(file.c_str());
    }

    void read(bool shuffled){
        if (!in.is_open()) {
            std::cerr << "can not open file " << filename << std::endl;
            exit(EXIT_FAILURE);
        }
        vector<string> records;
        string line, temp;
        int i = 0;
        vector<set<string>> counts(feature_names.size());
        while(read_word(temp, in)){
            if (temp == EOS){
                int size = line.size();
                if (line != "") records.push_back(line);
                line = "";
                i = 0;
            }else{
                if (line == "") line = temp;
                else line += " " + temp;
                if (i < x_dim) counts[i++].insert(temp);
            }
        }
        if (line != "") records.push_back(line);

        assert(counts.size() == feature_names.size());

        for (int i = 0; i < counts.size(); ++i){
            if (counts[i].size() >= MAX_VALUES) feature_types.push_back(true);
            else feature_types.push_back(false);
        }

        if (shuffled) {
            shuffle(records);
        }
        in.close();

        if (DEBUG) {
            std::cout << "num_samples:" << records.size() << std::endl;
            for (string r: records)
                std::cout << r << std::endl;

            for (bool b: feature_types)
                if (b)
                    std::cout << "C\t";
                else
                    std::cout << "D\t";
            std::cout << std::endl;
        }

        filename += "-shuffled";
        ofstream out(filename);
        if (out.is_open()){
            out << size << "\t" << x_dim << '\t' << num_classes << std::endl;
        }else{
            std::cerr << "corpus-shuffled not opened!\n";
            exit(EXIT_FAILURE);
        }

        for (string s: records)
            out << s << std::endl;

        out.close();

        in.open(filename);
        reset();
    }

    void __get_basic_info(){
        if (!in.is_open()){
            std::cout << "File is not opened!\n";
            exit(EXIT_FAILURE);
        }
        // 获得样本数量、特征维度以及类别个数
        in >> size >> x_dim >> num_classes;

        if (num_classes == 2) {
            binary_class = true;
//            multi_class = false;
            if (zero_one_label)
                label_idx = 0.0f;
            else
                label_idx = -1.0f;
        }else if (num_classes > 2){
            binary_class = false;
//            multi_class = true;
            label_idx = 0.0f;
        }else{
            regression = true;
        }

        feature_names.clear();
        for (int i = 0; i < x_dim; ++i) feature_names.push_back(std::to_string(i+1));
    }

    bool generator(){
        if (0 == x || 0 == y){
            logger.error("x or y is null pointer!");
            exit(EXIT_FAILURE);
        }

        current_batch_size = 0;
        LONG __x_idx = 0, __y_idx = 0;
        while(in >> x[__x_idx++]){
            if (__x_idx % x_dim == 0){
                in >> y[__y_idx++];
                ++current_batch_size;
                ++__records_idx;
            }
            if (current_batch_size == batch_size)
                return true;
        }
        return false;
    }

    void get_data(){
        data_x = std::make_shared<vector<real>>(size * x_dim, 0.0);
        data_y = std::make_shared<vector<real>>(size, 0.0);

        LONG idx = 1;
        string temp;
        real num;
        for (LONG i = 0; i < size; ++i){
            for (LONG j = 0; j < x_dim; ++j){
                in >> temp;
                if(valid_number(temp)){
                    (*data_x)[i * x_dim + j] = (float)atof(temp.c_str());
                }else{
                    if (value_idx_map.count(temp) == 0){
                        idx_value_map[idx] = temp;
                        value_idx_map[temp] = idx;
                        ++idx;
                    }
                    (*data_x)[i * x_dim + j] = value_idx_map[temp];
                }
            }
            in >> temp;
            if(valid_number(temp)){
                num = (float)atof(temp.c_str());
                if (binary_class){
                    if (label_dict.count(num) == 0) {
                        label_dict[num] = label_idx;
                        label_reverse_dict[label_idx] = num;
                        if (zero_one_label)
                            label_idx += 1.0f;
                        else label_idx += 2.0f;
                    }
                }else{
                    if (label_dict.count(num) == 0) {
                        label_dict[num] = label_idx;
                        label_reverse_dict[label_idx] = num;
                        label_idx += 1.0f;
                    }
                }
                (*data_y)[i] = label_dict[num];
            }else{
                if (value_idx_map.count(temp) == 0){
                    idx_value_map[idx] = temp;
                    value_idx_map[temp] = idx;
                    ++idx;
                }
                (*data_y)[i] = value_idx_map[temp];
            }
        }
    }

    void test(){
        for (auto iter = value_idx_map.begin(); iter != value_idx_map.end(); ++iter){
            std::cout << iter->first << '\t' << iter->second << std::endl;
        }

        for (LONG i = 0; i < size; ++i){
            for (LONG j = 0; j < x_dim; ++j){
                std::cout << (*data_x)[i * x_dim + j] << ":" << idx_value_map[(*data_x)[i * x_dim + j]] << "\t";
            }

            std::cout << (*data_y)[i] << ":" << idx_value_map[(*data_y)[i]] << std::endl;
        }

    }

    inline real get_x(LONG idx, int32_t fid){
        assert(idx >= 0);
        assert(idx < size);
        assert(fid < x_dim);
        return (*data_x)[idx * x_dim + fid];
    }

    inline real* get_x(LONG idx){
        assert(idx >= 0);
        assert(idx < size);
        return &(*data_x)[idx * x_dim];
    }

    inline real get_label(LONG idx){
        assert(idx >= 0 && idx < size);
        return (*data_y)[idx];
    }

    inline pair<real, real> get_data(LONG idx){
        assert(idx >= 0 && idx < size);
        return {(*data_y)[idx], (*buffer)[idx]};
    }

    real most_common(const vector<LONG>& x){
        assert(!x.empty());
        map<real, LONG> label_count_map;
        for (LONG idx: x){
            ++label_count_map[get_label(idx)];
        }
        if (label_count_map.size() == 1)
            return label_count_map.begin()->first;

        int idx = 0;
        real res;
        LONG __max;
        for (auto iter = label_count_map.begin(); iter != label_count_map.end(); ++iter){
            if (0 == idx) {
                res = iter->first;
                __max = iter->second;
            }else if (__max < iter->second){
                __max = iter->second;
                res = iter->first;
            }
            ++idx;
        }
        return res;
    }

    real most_common(const vector<LONG>& x, LONG& most_count){
        assert(!x.empty());
        map<real, LONG> label_count_map;
        for (LONG idx: x){
            ++label_count_map[get_label(idx)];
        }
        if (label_count_map.size() == 1){
            most_count = label_count_map.begin()->second;
            return label_count_map.begin()->first;
        }

        int idx = 0;
        real res;
        for (auto iter = label_count_map.begin(); iter != label_count_map.end(); ++iter){
            if (0 == idx) {
                res = iter->first;
                most_count = iter->second;
            }else if (most_count < iter->second){
                most_count = iter->second;
                res = iter->first;
            }
            ++idx;
        }
        return res;
    }

    void reset(){
        in.clear();
        in.seekg(0, ios::beg);
        __get_basic_info();
    }

};



#endif //FASTAI_CORPUS_H
