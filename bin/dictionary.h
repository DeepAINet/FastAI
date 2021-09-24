//
// Created by mengqy on 2019/1/15.
//

#ifndef FASTAI_DICTIONARY_H
#define FASTAI_DICTIONARY_H

#include "constant.h"
#include "common.h"

const PLINT hash_size = 30000000;

typedef struct word{
    string name;
    PLINT count;
} WORD;

int word_cmp(const void *a, const void *b){
    return ((WORD *)b)->count - ((WORD *)a)->count;
}

struct search_result{
    PLINT hv;
    bool searched;
};

class dictionary {
private:
    LONG *hash_table = nullptr;
    WORD *words = nullptr;
    PLINT current_idx = 0;
    PLINT capacity = LENGTH;
    LONG __total = 0;

public:
    dictionary(){
        hash_table = (LONG *)malloc(hash_size * sizeof(LONG));
        words = (WORD *)malloc(sizeof(WORD) * LENGTH);
        init_hash_table();
    }

    dictionary(const char *file, int min_count=0){
        hash_table = (LONG *)malloc(hash_size * sizeof(LONG));
        words = (WORD *)malloc(sizeof(WORD) * LENGTH);
        init_hash_table();
        construct_dict(file, min_count);
    }

    void init_hash_table();


    ~dictionary(){
        if (words != nullptr) free(words);
        if (hash_table != nullptr) free(hash_table);
    }

    void construct_dict(const char *file, int min_count=0);

    void add_word(const string& word);

    PLINT get_hash_value(const string& word);

    struct search_result find(const string& word);

    void rearrange(int min_count=0);

    void qsort0(LONG start, LONG end);

    PLINT count(LONG idx);

    PLINT count(const string& word);

    string operator[](LONG idx);

    LONG operator[](const string& word);

    size_t size();

    LONG total();

    bool empty();

    void clear();

    void list();

    void save(const char *file);
};


inline PLINT dictionary::count(LONG idx){
    if (idx >= current_idx || idx < 0) return 0;
    return words[idx].count;
}

inline PLINT dictionary::count(const string& word){
    struct search_result  res = find(word);
    if (!res.searched) return 0;
    return words[hash_table[res.hv]].count;
}

inline string dictionary::operator[](LONG idx) {
    if (idx >= current_idx || idx < 0) return "";
    return words[idx].name;
}

inline LONG dictionary::operator[](const string& word){
    struct search_result  res = find(word);
    return hash_table[res.hv];
}

inline size_t dictionary::size(){
    return current_idx;
}

inline LONG dictionary::total(){
    return __total;
}

inline bool dictionary::empty(){
    return current_idx == 0;
}


void dictionary::init_hash_table(){
    for (LONG i = 0; i < hash_size; ++i)
        hash_table[i] = -1;
}

void dictionary::construct_dict(const char *file, int min_count){
    string temp;
    ifstream in(file);
    while(in >> temp){
        add_word(temp);
        ++__total;
    }
    rearrange(min_count);
    in.close();
}

void dictionary::add_word(const string& word){
    if (current_idx == hash_size) return;
    struct search_result res = find(word);

    if (!res.searched){
        hash_table[res.hv] = current_idx;
        words[current_idx].name = word;
        words[current_idx].count = 0;
        ++current_idx;
    }

    words[hash_table[res.hv]].count += 1;

    if (current_idx == hash_size){
        std::cerr << "Capacity is too small to use!\n";
        return;
    }
    if (current_idx == capacity){
        capacity += LENGTH;
        words = (WORD *)realloc(words, capacity * sizeof(WORD));
        if (words == nullptr){
            std::cerr << "Memory allocated failure!\n";
            exit(EXIT_FAILURE);
        }
    }
}

PLINT dictionary::get_hash_value(const string& word){
    PLINT result = 0;
    for (int i = 0; i < word.length(); ++i)
        result = result * 257 + word[i];
    return result % hash_size;
}

struct search_result dictionary::find(const string& word){
    PLINT __hv = get_hash_value(word);
    PLINT count = 0;
    struct search_result res;
    while(count < hash_size){
        if(hash_table[__hv] == -1) {
            res.searched = false;
            res.hv = __hv;
            return res;
        }
        if (strcmp(words[hash_table[__hv]].name.c_str(), word.c_str()) != 0)
            __hv = (__hv + 1) % hash_size;
        else {
            res.searched = true;
            res.hv = __hv;
            return res;
        }
        ++count;
    }
    std::cerr << "Have accessed all the hash table, but didn't find it.\n";
    exit(EXIT_FAILURE);
}

void dictionary::rearrange(int min_count){
    qsort(words, current_idx, sizeof(WORD), word_cmp);
//        qsort0(0, current_idx-1);
    init_hash_table();

    LONG size = current_idx;
    LONG i = size-1;
    for (; i >= 0; --i)
        if (words[i].count >= min_count) break;

    current_idx = (PLINT)(i+1);
    struct search_result res;
    for (i = 0; i < current_idx; ++i){
        res = find(words[i].name);
        hash_table[res.hv] = i;
    }
}

void dictionary::qsort0(LONG start, LONG end){
    if (start >= end) return;
    struct word __word;
    __word.name = words[start].name;
    __word.count = words[start].count;
    LONG _start = start;
    LONG _end = end;

    while(start < end){
        while (start < end && words[end].count < __word.count)
            --end;
        words[start] = words[end];
        while(start < end && words[start].count >= __word.count)
            ++start;
        words[end] = words[start];
    }
    words[start] = __word;
    qsort0(_start, start-1);
    qsort0(start+1, _end);
}

void dictionary::clear(){
    init_hash_table();
    current_idx = 0;
}

void dictionary::list(){
    for (LONG i = 0; i < current_idx; ++i)
        std::cout << words[i].name << "\t"
                  << words[i].count << "\t"
                  << hash_table[find(words[i].name).hv] << std::endl;
}

void dictionary::save(const char *file){
    ofstream out(file);
    if (!out.is_open())
        logger.error("File not opened!");

    for (LONG i = 0; i < current_idx; ++i)
        out << words[i].name << '\t' << words[i].count << std::endl;

    out.close();
}


#endif //FASTAI_DICTIONARY_H
