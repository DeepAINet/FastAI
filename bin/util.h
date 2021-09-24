//
// Created by mengqy on 2019/1/15.
//

#ifndef FASTAI_UTIL_H
#define FASTAI_UTIL_H

#include <cstdio>
#include <string>
#include <cmath>
#include <map>

#include "common.h"
using namespace std;

#define MAX_STRING_LENGTH 1000
#define EOS "</s>"
#define ENTROPY_TABLE_SIZE 1000
#define QUERY false
vector<real> ENTROPY_TABLE;

void precompute(){
    ENTROPY_TABLE.push_back(0.0f);
    real t;
    for (int i = 1; i < ENTROPY_TABLE_SIZE; ++i){
        t = (real)i / (real)ENTROPY_TABLE_SIZE;
        ENTROPY_TABLE.push_back(-1 * t * std::log(t));
    }
    ENTROPY_TABLE.push_back(0.0f);
}

real query(real weight){
    if (weight == 0.0f || weight == 1.0f) return 0.0f;
    assert(weight > 0.0f);
    assert(weight < 1.0f);
    real idx = weight * ENTROPY_TABLE_SIZE;
    if (idx == (int)idx) return ENTROPY_TABLE[idx];
    return (ENTROPY_TABLE[idx] + ENTROPY_TABLE[idx+1]) / 2.0f;
}


int read_word(char *str, FILE *fin){
    int ch, a = 0;
    while(!feof(fin)){
        ch = fgetc(fin);
        if (feof(fin)) return 0;
        if (13 == ch) continue;
        if (ch == ' ' || ch == '\t' || ch == '\n'){
            if (0 == a){
                if (ch == '\n') {
                    str[a] = 0;
                    return 0;
                }
            }else{
                if (ch == '\n') {
                    ungetc(ch, fin);
                }
                break;
            }
        }
        if (a >= MAX_STRING_LENGTH-2) --a;
        str[a++] = ch;
    }
    str[a] = 0;
    return 1;
}

int read_word(string& word, ifstream& in){
    int c;
    std::streambuf& sb = *in.rdbuf();
    word.clear();

    while((c = sb.sbumpc()) != EOF){
        if (c == ' ' ||
            c == '\n' ||
            c == '\r' ||
            c == '\t' ||
            c == '\v' ||
            c == '\f' || c == '\0'){
            if (word.empty()){
                if (c == '\n'){
                    word += EOS;
                    return true;
                }
                continue;
            }else{
                if (c == '\n') sb.sungetc();
                return true;
            }
        }
        word.push_back(c);
    }
    in.get();

    return !word.empty();
}

int read_word(string& str, FILE *fin){
    str.clear();
    int ch;
    while(!feof(fin)){
        ch = fgetc(fin);
        if (feof(fin)) return 0;
        if (13 == ch) continue;
        if (ch == ' ' || ch == '\t' || ch == '\n'){
            if (0 == str.size()){
                if (ch == '\n') return 0;
            }else{
                if (ch == '\n') {
                    ungetc(ch, fin);
                }
                break;
            }
        }
        str.push_back(ch);
    }
    return 1;
}

/**
 * 判断字符串是否为合法数字.
 * @param c
 * @return
 */
bool valid_number(string c){
    if (c.empty()) return false;
    int i = 0;
    int size = c.size();
    while(i < size && c[i] == ' '){
        ++i;
    }

    if (i < size && (c[i] == '+' || c[i] == '-')) ++i;

    bool integer = false;
    while(i < size && isdigit(c[i])){
        integer = true;
        ++i;
    }

    bool point = false;
    if (i < size && c[i] == '.'){
        point = true;
        ++i;
    }

    bool small = false;
    while(i < size && isdigit(c[i])){
        ++i;
        small = true;
    }

    if (i < size && c[i] == 'e'){
        ++i;
        bool exp = false;
        if (i < size && (c[i] == '+' || c[i] == '-')) ++i;
        while(i < size && isdigit(c[i])){
            ++i;
            exp = true;
        }
        if (!exp) return false;
    }
    while(i < size && c[i] == ' '){
        ++i;
    }

    return i == size && (integer || (point && small));
}

template <typename T>
T most_common(T *ts, LONG len){
    map<T, LONG> __counts;

    for (LONG i = 0; i < len; ++i)
        __counts[ts[i]]++;

    T __res;
    LONG __max = -1;
    for (auto iter = __counts.begin(); iter != __counts.end(); ++iter)
        if (iter->second > __max){
            __max = iter->second;
            __res = iter->first;
        }
    return __res;
}

real compute_entropy(const vector<real> &nums){
    if (nums.empty()) return 0.0f;
    real entropy = 0.0;
    for (real num: nums){
        if (num == 0) continue;
        if (num < 0) {
            std::cerr << "num < 0, execute log(num) failure!\n";
            exit(EXIT_FAILURE);
        }
        if (QUERY)
            entropy += query(num);
        else entropy -= num * std::log(num);
    }
    return entropy;
}

real compute_entropy(real weight){
    real entropy = 0.0f;
    real left = weight, right = 1 - weight;
    if (QUERY){
        entropy += query(left);
        entropy += query(right);
    }else{
        entropy -= left * std::log(left);
        entropy -= right * std::log(right);
    }

    return entropy;
}

/**
 * 计算信息增益
 */
real compute_entropy(map<real, LONG>& count, LONG total_count){
    if (total_count <= 1) return 0.0f;

    real entropy = 0.0;
    real t;
    for (auto iter = count.begin(); iter != count.end(); ++iter){
        if (iter->second == 0) continue;
        t = (real)iter->second / (real)total_count;
        if (t < 0) {
            std::cerr << "num < 0, execute log(num) failure!\n";
            exit(EXIT_FAILURE);
        }
        if (QUERY)
            entropy += query(t);
        else entropy -= t * std::log(t);
    }
    return entropy;
}

real compute_gini_index(const map<real, LONG>& count, LONG total_count){
    if (total_count <= 1) return 0.0f;

    real entropy = 1.0;
    real t;
    for (auto iter = count.cbegin(); iter != count.end(); ++iter){
        if (iter->second == 0) continue;
        t = (real)iter->second / (real)total_count;
        if (t < 0) {
            std::cerr << "num < 0, execute log(num) failure!\n";
            exit(EXIT_FAILURE);
        }
        entropy -= t * t;
    }
    return entropy;
}

real compute_gini_index(const vector<real> &nums){
    if (nums.empty()) return 0.0f;

    real entropy = 1.0;
    for (real num: nums){
        if (num == 0) continue;
//        if (num < 0) {
//            std::cerr << "num < 0, execute log(num) failure!\n";
//            exit(EXIT_FAILURE);
//        }
        entropy -= num * num;
    }
    return entropy;
}

template<typename T>
pair<size_t, T> argmax(const vector<T>& nums){
    pair<size_t, T> res;
    if (nums.empty()) return res;

    T m = nums[0];
    res.first = 0;
    res.second = m;
    for (size_t i = 1; i < nums.size(); ++i){
        if (m < nums[i]){
            res.first = i;
            res.second = nums[i];
            m = nums[i];
        }
    }
    return res;
}

template<typename T>
pair<size_t, T> argmin(const vector<T>& nums){
    pair<size_t, T> res;
    if (nums.empty()) return res;

    T m = nums[0];
    res.first = 0;
    res.second = m;
    for (size_t i = 1; i < nums.size(); ++i){
        if (m > nums[i]){
            res.first = i;
            res.second = nums[i];
            m = nums[i];
        }
    }
    return res;
}

void convert2vector(const string& text, vector<string>& vec){
    string temp;
    vec.clear();
    for (int i = 0; i < text.size(); ++i){
        if (isspace(text[i])){
            if (temp.empty()) continue;
            vec.push_back(temp);
            temp.clear();
        }else temp.push_back(text[i]);
    }
    if (!temp.empty()) vec.push_back(temp);
}

vector<string> convert2vector(const string& text){
    string temp;
    vector<string> vec;
    for (int i = 0; i < text.size(); ++i){
//        if (isspace(text[i])) continue;
        if (text[i] == '\t'){
            if (temp.empty()) continue;
            vec.push_back(temp);
            temp.clear();
        }else {
            if (isspace(text[i])) continue;
            temp.push_back(text[i]);
        }
    }
    if (!temp.empty()) vec.push_back(temp);

    return vec;
}


vector<string> split(const string& txt, char c=' '){
    string temp;
    vector<string> res;
    if (c == ' '){
        for (int i = 0; i < txt.size(); ++i) {
            if (isspace(txt[i])) {
                if (temp.empty()) continue;
                res.push_back(temp);
                temp.clear();
            } else {
                if (txt[i] == '\r') continue;
                temp.push_back(txt[i]);
            }
        }
        if (!temp.empty()) res.push_back(temp);
    }else{
        for (int i = 0; i < txt.size(); ++i) {
            if (txt[i] == c) {
                if (temp.empty()) continue;
                res.push_back(temp);
                temp.clear();
            } else {
                if (txt[i] == '\r') continue;
                temp.push_back(txt[i]);
            }
        }
        if (!temp.empty()) res.push_back(temp);
    }
    return res;
}


#endif //FASTAI_UTIL_H
