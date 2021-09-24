//
// Created by mengqy on 2018/12/15.
//

#ifndef FASTAI_CORN_H
#define FASTAI_CORN_H

#include <vector>
#include <unordered_map>
#include <set>
#include "dictionary.h"
#include "util.h"
#include "global.h"

namespace lc{

typedef struct term{
    string word;
    string label;

    string to_str(){
        return word + "/" + label;
    }
} TERM;

typedef struct state{
    LONG idx;
    LONG init_count;
    LONG count;
} STATE;

set<string> END = {"。", "!", "?", ".", "？"};

class LabelCorpus {
public:
    ifstream in;
    vector<vector<TERM>> corpus;
    dictionary words;
    dictionary labels;

public:
    LabelCorpus(string filename)
    :in(filename){
        read(in);
    }

    ~LabelCorpus(){}

    term parse2term(string text){
        int i = 0;
        TERM res;
        while (i < text.size() && text[i] != '/'){
            res.word.push_back(text[i]);
            ++i;
        }

        while (i < text.size() && text[i] == '/') {
            ++i;
        }

        while (i < text.size() && text[i] != '/'){
            res.label.push_back(text[i]);
            ++i;
        }
        return res;
    }

    bool ending(string txt){
        if (END.count(txt)) return true;
        return false;
    }

    void parse(string text){
        string temp;
        vector<TERM> sentence;
        for (int i = 0; i < text.size(); ++i){
            if (text[i] == ' ' || text[i] == '\t'){
                if (!temp.empty()){
                    TERM term_ = parse2term(temp);
                    sentence.push_back(term_);
                    if (ending(term_.word)){
                        corpus.push_back(sentence);
                        sentence.clear();
                    }
                    temp.clear();
                }
                continue;
            }
            if (text[i] == '\r') continue;
            temp.push_back(text[i]);
        }
    }

    void read(ifstream& in){
        string line;
        while(getline(in, line)){
            if (line.length() == 1 && line[0] == '\n') continue;
            parse(line);
        }

        for (auto sen_ = corpus.begin(); sen_ != corpus.end(); ++sen_)
            for (auto term_ = sen_->begin(); term_ != sen_->end(); ++term_){
                words.add_word(term_->word);
                labels.add_word(term_->label);
            }
        words.rearrange();
        labels.rearrange();
        in.close();
    }

    size_t size(){
        return corpus.size();
    }

    size_t word_size(){
        return words.size();
    }

    size_t label_size(){
        return labels.size();
    }

    vector<TERM>& operator [] (LONG idx){
        return corpus[idx];
    }

    void print(){
        words.list();
        labels.list();
    }
};

class sentence{
public:
    int n;
    int m;
    vector<vector<string>> terms;
    vector<int> words;
    unordered_map<int, real> sparse_features;
    unordered_map<string, real> sparse_features_;

    unordered_map<string, real> state_transfer_features;
    unordered_map<string, real> state_features;
    real total=0.0f;
    vector<vector<LONG>> potential_label_series;
    real total_probability;

public:
    sentence(){};

    void construct(vector<string>& vec, LONG x_dim){
        for (LONG i = 0; i < vec.size(); ++i){
            terms.push_back(convert2vector(vec[i]));
        }
        m = vec.size();
        n = x_dim;
    }

    void active_feature(int fid){
        sparse_features[fid] += 1.0f;
        total += 1.0f;
    }

    void active_feature(string key){
        sparse_features_[key] += 1.0f;
        total += 1.0f;
    }

    void active_state_feature(string key){
        state_features[key] += 1.0f;
        total += 1.0f;
    }

    void active_state_transfer_feature(string key){
        state_transfer_features[key] += 1.0f;
        total += 1.0f;
    }

    string get_x(LONG i, LONG j){
        assert(i < m);
        assert(j < n - 1);
        return terms[i][j];
    }

    vector<string>& operator[] (LONG idx){
        assert(idx < m);
        return terms[idx];
    }

    string get_label(LONG i){
        assert(i < m);
        return terms[i][n-1];
    }

    string to_str(){
        string res;
        for (LONG i = 0; i < m; ++i){
            for(LONG j = 0; j < n; ++j)
                res += terms[i][j] + "\t";
            res += "\n";
        }
        return res;
    }

    void print_sparse_features(){
        for (auto sf: sparse_features){
            std::cout << sf.first << ':' << sf.second << ' ';
        }
        std::cout << std::endl;
    }
};

class corn {
public:
    LONG num_samples;
    INT x_dim;
    sentence *sentences;
    dictionary words;
    dictionary labels;
    unordered_map<int, set<int>> state_transfer_matrix;
    unordered_map<int, set<int>> observation_matrix;
    unordered_map<int, vector<int>> potential_states;
    int num_states;


public:
    corn(){}

    corn(string file){
        read(file.c_str());
    }

    corn(const char *file){
        read(file);
    }

    ~corn(){
        if (sentences != 0) delete [] sentences;
    }

    void read(const char *file){
        ifstream in(file);
        in >> num_samples >> x_dim;
        if (DEBUG) {
            std::cout << "num_samples:" << num_samples << "x_dim:" << x_dim << std::endl;
        }

        sentences = new sentence[num_samples];
        string temp;
        vector<string> terms;
        LONG idx = 0;
        while(getline(in, temp)){
            if (!temp.empty()){
                terms.push_back(temp);
            }else{
                if (!terms.empty()){
                    sentences[idx++].construct(terms, x_dim);
                    terms.clear();
                }
            }
        }
        if (!terms.empty()) sentences[idx++].construct(terms, x_dim);
        logger.info(std::to_string(idx) + " sentences.");

        if (DEBUG){
            for (LONG i = 0; i < num_samples; ++i)
                std::cout << sentences[i].to_str() << std::endl;
        }
        x_dim -= 1;

        for (LONG i = 0; i < num_samples; ++i){
            for (int j = 0; j < sentences[i].terms.size(); ++j){
                 words.add_word(sentences[i].terms[j][0]);
                 labels.add_word(sentences[i].terms[j].back());
            }
        }
        words.rearrange();
        labels.rearrange();
        stat();

        for (auto elem: potential_states){
            std::cout << words[elem.first] << ":\t";
            for (auto e: elem.second){
                std::cout << e << ":" << labels[e] << ",\t";
            }
            std::cout << std::endl;
        }

        get_words();
//        get_potential_label_series();
    }

    void stat(){
        for (LONG i = 0; i < num_samples; ++i){
            for (int j = 0; j < sentences[i].terms.size(); ++j){
                observation_matrix[labels[sentences[i].terms[j].back()]].insert(words[sentences[i].terms[j].front()]);
                if (j + 1 < sentences[i].terms.size()){
                    state_transfer_matrix[labels[sentences[i].terms[j].back()]].insert(labels[sentences[i].terms[j+1].back()]);
                }
            }
        }

        unordered_map<int, set<int>> ps;
        for (auto element: observation_matrix){
            for (auto word: element.second){
                ps[word].insert(element.first);
            }
        }

        for (auto element: ps){
            for (auto label: element.second){
                potential_states[element.first].push_back(label);
            }

        }

        num_states = state_transfer_matrix.size();
        std::cout << "num_states:" << num_states << std::endl;
    }

    LONG size(){
        return num_samples;
    }

    void get_potentials(vector<LONG>& sids, int i, vector<LONG>& potential, vector<vector<LONG>>& potentials){
        if (potential.size() == sids.size()){
            potentials.push_back(potential);
//            for (int num: potential)
//                std::cout << labels[num] << '\t';
//            std::cout << std::endl;
            return;
        }

        for (int j: potential_states[sids[i]]){
            if (i == 0){
                potential.push_back(j);
                get_potentials(sids, i+1, potential, potentials);
                potential.pop_back();
            }else{
                if (state_transfer_matrix[potential.back()].count(j) == 1){
                    potential.push_back(j);
                    get_potentials(sids, i+1, potential, potentials);
                    potential.pop_back();
                }
            }
        }
    }

    void get_words(){
        for (LONG i = 0; i < num_samples; ++i){
            for (int j = 0; j < sentences[i].terms.size(); ++j){
                sentences[i].words.push_back(words[sentences[i].terms[j][0]]);
            }
        }
    }
};

bool check_format(string word){
    if (word.substr(0, 3) != "%x[") return false;
    int i = 3;
    while(i < word.size() && isspace(word[i])) ++i;

    if (i < word.size() && (word[i] == '-'|| word[i] == '+')) ++i;
    while(i < word.size() && isdigit(word[i])) ++i;

    while(i < word.size() && isspace(word[i])) ++i;

    if (i < word.size() && word[i] == ',') ++i;

    while(i < word.size() && isspace(word[i])) ++i;

    if (i < word.size() && (word[i] == '-'|| word[i] == '+')) ++i;
    while(i < word.size() && isdigit(word[i])) ++i;

    while(i < word.size() && isspace(word[i])) ++i;
    if (i < word.size() && word[i] == ']') ++i;

    while(i < word.size() && isspace(word[i])) ++i;
    return i == word.size();
}

typedef vector<tuple<int, int>> TRANSFER;

typedef tuple<int, int> NODE;

class FeatureTemplate{
private:
    vector<TRANSFER> trans;
    vector<NODE> nodes;
public:
    FeatureTemplate(){}

    FeatureTemplate(const char *file){
        read_templates(file);
    }

    void read_templates(const char *file){
        nodes.clear();
        trans.clear();
        ifstream in(file);
        string temp;
        while(getline(in, temp)) {
            if (temp.size() == 0) continue;
            get_template(temp);
        }
        in.close();

        if (DEBUG){
            for (NODE n: nodes){
                std::cout << std::get<0>(n) << '\t' << std::get<1>(n) << std::endl;
            }

            for (TRANSFER t: trans) {
                for (tuple<int, int> &tt: t) {
                    std::cout << std::get<0>(tt) << ',' << std::get<1>(tt) << "->";
                }
                std::cout << std::endl;;
            }
        }
    }

    tuple<int, int> get_tuple(string tp){
        int first = 0, second = 0, i = 3;
        int c = 1;
        while(i < tp.size() && isspace(tp[i])) ++i;
        if (i < tp.size() && (tp[i] == '-' || tp[i] == '+')){
            if (tp[i] == '-') c *= -1;
            ++i;
        }
        while (i < tp.size() && isdigit(tp[i])){
            first = first * 10 + tp[i] - '0';
            ++i;
        }
        first *= c;
        c = 1;
        while(i < tp.size() && isspace(tp[i])) ++i;
        if (i < tp.size() && tp[i] == ',' ) ++i;
        while(i < tp.size() && isspace(tp[i])) ++i;
        if (i < tp.size() && (tp[i] == '-' || tp[i] == '+')){
            if (tp[i] == '-') c *= -1;
            ++i;
        }
        while (i < tp.size() && isdigit(tp[i])){
            second = second * 10 + tp[i] - '0';
            ++i;
        }
        second *= c;
        return std::make_tuple(first, second);
    }

    void get_template(const string& tp){
        vector<string> tps = split(tp, '/');
        if (tps.size() == 1){
            if (check_format(tps[0])){
                nodes.push_back(get_tuple(tp));
            }else{
                std::cerr << tps[0] << ":wrong format!\n";
            }
        }else{
            TRANSFER tran;
            bool valid = true;
            for (string a: tps){
                if (check_format(a)){
                    tran.push_back(get_tuple(a));
                }else{
                    valid = false;
                    std::cerr << a << ":wrong format!\n";
                }
            }
            if (valid) trans.push_back(tran);
        }
    }

    vector<NODE>& get_nodes(){
        return nodes;
    }

    vector<TRANSFER>& get_trans(){
        return trans;
    }
};

};

#endif //FASTAI_CORN_H
