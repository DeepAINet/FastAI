//
// Created by mengqy on 2019/1/13.
//

#ifndef FASTAI_GLOVE_H
#define FASTAI_GLOVE_H

#include <cstdlib>
#include <thread>
#include "dictionary.h"
#include "tensor.h"
#include "common.h"
#include "log.h"
#include "util.h"
#include "rand.h"
#include "global.h"

#define MAX_STRING_SIZE 1000

typedef float real;
typedef long long LONG;

const long long max_product = 100000;
const long long overflow = 1000000;
int verbose = 2;

LONG shuffle_size = 1000000;

LONG num_lines, *lines_per_thread;

int write_header = 1;

typedef struct cooccurrence{
    LONG word1;
    LONG word2;
    real count;
} CREC;

typedef struct cooccurrence_id{
    LONG word1;
    LONG word2;
    real count;
    int id;
} CRECID;

int comp_crec(const void *a, const void *b){
    int c;
    if ((c = ((CREC *)b)->word1 - ((CREC *)a)->word1) != 0) return c > 0 ? 1 : -1;
    else {
        c = ((CREC *)b)->word2 - ((CREC *)a)->word2;
        if (0 == c) return 0;
        return c > 0 ? 1 : -1;
    }
}

int comp_crecid(const void *a, const void *b){
    int c;
    if ((c = ((CRECID *)b)->word1 - ((CRECID *)a)->word1) != 0) return c > 0 ? 1 : -1;
    else {
        c = ((CRECID *)b)->word2 - ((CRECID *)a)->word2;
        if (0 == c) return 0;
        return c > 0 ? 1 : -1;
    }
}

void swap_crecid(CRECID *crecids, int i, int j){
    CRECID tmp = crecids[i];
    crecids[i] = crecids[j];
    crecids[j] = tmp;
}

void insert(CRECID *prior_queue, const CRECID& crecid, int idx){
    if (idx < 0) return;
    prior_queue[idx] = crecid;

    int __p = 0;
    while(idx != 0){
        __p = (idx-1)>>1;
        if (comp_crecid(&prior_queue[__p], &prior_queue[idx]) > 0){
            swap_crecid(prior_queue, __p, idx);
            idx = __p;
        }else break;
    }
}

void delete_crecid(CRECID *prior_queue, int idx){
    int i = 0;
    prior_queue[i] = prior_queue[idx];

    int __p = i;
    while((i = ((__p << 1) + 1)) < idx){
        if ((i+1 < idx) && (comp_crecid(&prior_queue[i], &prior_queue[i+1]) > 0))
            ++i;
        if (comp_crecid(&prior_queue[__p], &prior_queue[i]) > 0){
            swap_crecid(prior_queue, i, __p);
            __p = i;
        }else break;
    }
}

real check_nan(real update){
    if (isnan(update) || isinf(update)){
        fprintf(stderr, "\ncaught NaN in update.");
        return 0.;
    }else{
        return update;
    }
}

void test(int id){
    std::cout << id << std::endl;
}

void glove_train_thread(LONG id
        , real *cost, real *W, int embedding_dim, int thread_num,
        int x_max, real alpha,
        real eta, real *grad_sq, LONG dict_size
        ){
    LONG a, b, l1, l2;
    CREC cr;
    real diff, fdiff, temp1, temp2;
    FILE *fid;
    fid = fopen("shuffle.bin", "rb");
    if (0 == fid){
        std::cout << "can not open file shuffle.bin\n";
    }
    fseeko(fid, (num_lines / thread_num * id) * sizeof(CREC), SEEK_SET);
    cost[id] = 0;

    real *W_updates1 = (real*)malloc(embedding_dim * sizeof(real));
    real *W_updates2 = (real*)malloc(embedding_dim * sizeof(real));

    for (a = 0; a < lines_per_thread[id]; ++a){
        fread(&cr, sizeof(CREC), 1, fid);

        if (feof(fid)) break;
        if (cr.word1 < 1 || cr.word2 < 1) continue;

        l1 = (cr.word1 - 1LL) * (embedding_dim+1);
        l2 = ((cr.word2- 1LL) + dict_size) * (embedding_dim+1);

        diff = 0;
        for (b = 0; b < embedding_dim; ++b)
            diff += W[b + l1] * W[b + l2];
        diff += W[embedding_dim + l1] + W[embedding_dim + l2] - std::log(cr.count);

        fdiff = (cr.count > x_max) ? diff : pow(cr.count / x_max, alpha) * diff;

        if (isnan(diff) || isnan(fdiff) || isinf(diff) || isinf(fdiff)){
            fprintf(stderr, "Caught NaN in diff for kdiff for thread. Skipping update");
            continue;
        }

        cost[id] += 0.5 * fdiff * diff;

        /** 自适应梯度更新 **/
        fdiff *= eta;

        real W_updates1_sum = 0;
        real W_updates2_sum = 0;
        for (b = 0; b < embedding_dim; ++b){
            temp1 = fdiff * W[b + l2];
            temp2 = fdiff * W[b + l1];

            W_updates1[b] = temp1 / sqrt(grad_sq[b+l1]);
            W_updates2[b] = temp2 / sqrt(grad_sq[b+l2]);

            W_updates1_sum += W_updates1[b];
            W_updates2_sum += W_updates2[b];
            grad_sq[b+l1] += temp1 * temp1;
            grad_sq[b+l2] += temp2 * temp2;
        }

        if (!isnan(W_updates1_sum) && !isinf(W_updates1_sum) && !isnan(W_updates2_sum) && !isinf(W_updates2_sum)){
            for (b = 0; b < embedding_dim; ++b){
                W[b+l1] -= W_updates1[b];
                W[b+l2] -= W_updates2[b];
            }
        }

        W[embedding_dim + l1] -= check_nan(fdiff/sqrt(grad_sq[embedding_dim + l1]));
        W[embedding_dim + l2] -= check_nan(fdiff / sqrt(grad_sq[embedding_dim + l2]));
        fdiff *= fdiff;
        grad_sq[embedding_dim+l1] += fdiff;
        grad_sq[embedding_dim+l2] += fdiff;
    }

    free(W_updates1);
    free(W_updates2);

    fclose(fid);
}



class glove {
private:
    dictionary __dict;
    int __context_size=15;
    int __embedding_dim=128;
    int __train_steps=10000;
    int __every_steps=1000;
    int __epoch_num=1;
    int __min_count=0;
    bool __weighted = true;
    int __symmetric = true;

    int __iter_num = 25;
    int __use_unk_vec = 1;
    int __save_grad_sq = 0;
    int __use_binary = 0;
    int __model = 2;
    int checkpoint_every = 0;
    int __thread_num = 8;


    real eta = 0.05;
    real alpha = 0.75, x_max = 100.0;


    LONG total = 0;

    tensor<float> __embeddings;
    tensor<float> __coefficients;

    real *W, *grad_sq, *cost;

    char* vocab_file;
    char* input_file;
    char* save_W_file;
    char* save_grad_sq_file;
public:
    glove(){}

    void get_cooccurrences(const char *corpus_filename);

    void shuffle();

    int optimize();

    void train(const char *corpus_filename,
               const char *model_dir,
               int embedding_dim=128,
               int window_size=15,
               int epoch_num=1,
               int min_count=0,
               int train_steps=10000,
               int every_steps=10000,
               bool weighted=true,
               int symmetric=0);

    void merge_chunk(const CRECID& current_crec, CRECID& old_crec, FILE *output_file);

    void merge_chunk(CREC *crecs, long long size, FILE *output_file);

    void merge_files(int merge_num);

    void shuffle(CREC *array, LONG size);

    void shuffle_chunks();

    void save_crecs(CREC *crecs, LONG size, FILE *output_file);

    void shuffle_merge(int merge_num);

    void init_params();

    int save_params(int nb_iter);

};


void glove::train(const char *corpus_filename,
        const char *model_dir,
        int embedding_dim,
        int window_size,
        int epoch_num,
        int min_count,
        int train_steps,
        int every_steps,
        bool weighted,
        int symmetric){

    __context_size = window_size;
    __embedding_dim = embedding_dim;
    __epoch_num = epoch_num;
    __min_count = min_count;
    __train_steps = train_steps;
    __every_steps = every_steps;
    __weighted = weighted;
    __symmetric = symmetric;

    __dict.construct_dict(corpus_filename, __min_count);

//    __dict.list();

    get_cooccurrences(corpus_filename);

    shuffle_chunks();

    optimize();

}

void glove::merge_chunk(const CRECID& current_crec, CRECID& old_crec, FILE *output_file){
    if ((current_crec.word1 == old_crec.word1) && (current_crec.word2 == old_crec.word2)){
        old_crec.count += current_crec.count;
        return;
    }

    ++total;
    fwrite(&old_crec, sizeof(CREC), 1, output_file);
    old_crec = current_crec;
}

void glove::merge_chunk(CREC *crecs, long long size, FILE *output_file){
    CREC old = crecs[0];
    for (long long i = 1; i < size; ++i) {
        if (crecs[i].word1 == old.word1 && crecs[i].word2 == old.word2){
            old.count += crecs[i].count;
        }else{
            fwrite(&old, sizeof(CREC), 1, output_file);
            old = crecs[i];
        }
    }
    fwrite(&old, sizeof(CREC), 1, output_file);
}

void glove::get_cooccurrences(const char *corpus_filename) {
    clock_t start_time = clock();
    size_t __dict_size = __dict.size();
    if (0 == __dict_size){
        logger.error("corpus file is empty!");
        exit(EXIT_FAILURE);
    }

    // look up start and end point.
    LONG *__lookup = (LONG *)malloc(sizeof(LONG) * (__dict_size+1));
    if (0 == __lookup){
        logger.error("lookup allocate memory failure.");
        exit(EXIT_FAILURE);
    }
    int __tmp;
    __lookup[0] = 0;
    for (int i = 1; i < __dict_size+1; ++i){
        __tmp = (int)(max_product / i);
        if (__tmp < __dict_size) __lookup[i] = __lookup[i-1] + __tmp;
        else __lookup[i] = __lookup[i-1] + __dict_size;
    }

    // co occurrences table.
    real *__co_occur_table = (real *)malloc(sizeof(real) * __lookup[__dict_size]);
    if (0 == __co_occur_table){
        logger.error("co_occur_table allocate memory failure.");
        exit(EXIT_FAILURE);
    }
    for (LONG i = 0; i < __lookup[__dict_size]; ++i)
        __co_occur_table[i] = 0.0;

    // context.
    LONG *__context = (LONG *)malloc(sizeof(LONG) * __context_size);
    if (0 == __context){
        logger.error("context allocate memory failure.");
        exit(EXIT_FAILURE);
    }

    CREC *__crecs = (CREC *)malloc(sizeof(CREC) * overflow);
    if (0 == __crecs){
        logger.error("crecs allocate memory failure.");
        exit(EXIT_FAILURE);
    }

    char tmp[1000];
    FILE *fid = fopen(corpus_filename, "r");
    if (0 == fid){
        sprintf(tmp, "can not open file %s.", corpus_filename);
        logger.error(tmp);
    }

    int __merge_idx = 1;
    sprintf(tmp, "%s_%04d.bin", "co_occurrence", __merge_idx);
    FILE *output_file = fopen(tmp, "wb");
    if (0 == output_file){
        logger.error("can't open file output_file");
        exit(EXIT_FAILURE);
    }

    int a = 0, k = 0, __term_idx = 0;
    LONG __center_word, __context_word;
    string word;
    long long __overflow_idx = 0;
    while(!feof(fid)){
        if (__overflow_idx >= (overflow - 2 * __context_size)){
            qsort(__crecs, __overflow_idx, sizeof(CREC), comp_crec);
            merge_chunk(__crecs, __overflow_idx, output_file);
            fclose(output_file);

            __overflow_idx = 0;
            ++__merge_idx;
            sprintf(tmp, "%s_%04d.bin", "co_occurrence", __merge_idx);
            output_file = fopen(tmp, "wb");
            if (0 == output_file){
                logger.error("can't open file output_file");
                exit(EXIT_FAILURE);
            }
        }

        a = read_word(word, fid);

        if (0 == a){
            if (feof(fid)) {
                logger.info("corpus file end reading.");
                break;
            }
            __term_idx = 0;
            continue;
        }

        __center_word = __dict[word];
        for (k = __term_idx-1; k >= (__term_idx>__context_size?(__term_idx-__context_size):0); --k){
            __context_word = __context[k % __context_size];

            if (((__context_word+1) * (__center_word+1)) < max_product){
                if (__symmetric == 0) __co_occur_table[__lookup[__center_word] + __context_word] += __weighted?1/(real)(__term_idx-k):(real)1;
                else if (__symmetric == 1) __co_occur_table[__lookup[__context_word] + __center_word] += __weighted?1/(real)(__term_idx-k):(real)1;
                else{
                    __co_occur_table[__lookup[__center_word] + __context_word] += __weighted?(1/(real)(__term_idx-k)):(real)1;
                    __co_occur_table[__lookup[__context_word] + __center_word] += __weighted?(1/(real)(__term_idx-k)):(real)1;
                }
            }else{
                if (__symmetric == 0){
                    __crecs[__overflow_idx].word1 = __center_word;
                    __crecs[__overflow_idx].word2 = __context_word;
                    __crecs[__overflow_idx].count = __weighted?(1/(real)(__term_idx-k)):(real)1;
                    ++__overflow_idx;
                }else if (__symmetric == 1){
                    __crecs[__overflow_idx].word1 = __context_word;
                    __crecs[__overflow_idx].word2 = __center_word;
                    __crecs[__overflow_idx].count = __weighted?(1/(real)(__term_idx-k)):(real)1;
                    ++__overflow_idx;
                }else{
                    __crecs[__overflow_idx].word1 = __center_word;
                    __crecs[__overflow_idx].word2 = __context_word;
                    __crecs[__overflow_idx].count = __weighted?(1/(real)(__term_idx-k)):(real)1;
                    ++__overflow_idx;

                    __crecs[__overflow_idx].word1 = __context_word;
                    __crecs[__overflow_idx].word2 = __center_word;
                    __crecs[__overflow_idx].count = __weighted?(1/(real)(__term_idx-k)):(real)1;
                    ++__overflow_idx;
                }
            }
        }

        __context[__term_idx % __context_size] = __center_word;
        ++__term_idx;
    }

    qsort(__crecs, (size_t)__overflow_idx, sizeof(CREC), comp_crec);
    merge_chunk(__crecs, __overflow_idx, output_file);
    fclose(output_file);

    sprintf(tmp, "%s_0000.bin", "co_occurrence");
    output_file = fopen(tmp, "wb");
    if (0 == output_file){
        logger.error("can't open file output_file");
        exit(EXIT_FAILURE);
    }

    for (LONG i = 0; i < __dict_size; ++i)
        for (LONG j = __lookup[i]; j < __lookup[i+1]; ++j){
            CREC crec;
            crec.word1 = i;
            crec.word2 = j-__lookup[i];
            crec.count = __co_occur_table[j];
            if (crec.count > 0) fwrite(&crec, sizeof(CREC), 1, output_file);
        }
    fclose(output_file);

    free(__lookup);
    free(__co_occur_table);
    free(__context);
    free(__crecs);
    fclose(fid);

    merge_files(__merge_idx+1);
    clock_t end_time = clock();
    sprintf(tmp, "co-occurrences have been merged, %lld co-occurrences, exhaust %ld s", total, (end_time-start_time)/CLOCKS_PER_SEC);
    logger.info(tmp);
}

void glove::merge_files(int merge_num){
    FILE **input_files=0, *output_file=0;

    input_files = (FILE **)malloc(sizeof(FILE) * merge_num);
    if (0 == input_files){
        logger.error("input_files allocate memory failure.\n");
        exit(EXIT_FAILURE);
    }

    char temp[1000];
    output_file = fopen("co-occurrence.bin", "wb");
    if (0 == output_file){
        logger.error("can not open file output_file.\n");
        exit(EXIT_FAILURE);
    }

    CRECID *prior_queue = (CRECID *)malloc(sizeof(CRECID) * merge_num);
    if (0 == prior_queue){
        logger.error("prior_queue allocate memory failure.\n");
        exit(EXIT_FAILURE);
    }
    CRECID __new, __old;
    for (int i = 0; i < merge_num; ++i){
        sprintf(temp, "co_occurrence_%04d.bin", i);
        input_files[i] = fopen(temp, "rb");
        if (0 == input_files[i]){
            logger.error("can not open file input_files[i].\n");
            exit(EXIT_FAILURE);
        }
        fread(&__new, sizeof(CREC), 1, input_files[i]);
        __new.id = i;
        insert(prior_queue, __new, i);
    }

    int size = merge_num, __current_input_file = 0;
    __old = prior_queue[0];
    __current_input_file = __old.id;
    delete_crecid(prior_queue, size-1);
    fread(&__new, sizeof(CREC), 1, input_files[__current_input_file]);
    if (feof(input_files[__current_input_file]))
        --size;
    else {
        __new.id = __current_input_file;
        insert(prior_queue, __new, size-1);
    }

    while(size > 0){
        merge_chunk(prior_queue[0], __old, output_file);
        __current_input_file = prior_queue[0].id;
        delete_crecid(prior_queue, size-1);

        fread(&__new, sizeof(CREC), 1, input_files[__current_input_file]);
        if (feof(input_files[__current_input_file])) --size;
        else{
            __new.id = __current_input_file;
            insert(prior_queue, __new, size-1);
        }
    }
    fwrite(&__old, sizeof(CREC), 1, output_file);

    for (int i = 0; i < merge_num; ++i){
        fclose(input_files[i]);
        sprintf(temp, "co_occurrence_%04d.bin", i);
        remove(temp);
    }

    free(prior_queue);
}

void glove::shuffle(CREC *array, LONG size){
    CREC __temp;
    LONG j;
    for (LONG i = size - 1; i >= 0; --i){
        j = get_rand(i+1);
        __temp = array[j];
        array[j] = array[i];
        array[i] = __temp;
    }
}

void glove::save_crecs(CREC *crecs, LONG size, FILE *output_file){
    for (LONG i = 0; i < size; ++i){
        fwrite(&crecs[i], sizeof(CREC), 1, output_file);
    }
}

void glove::shuffle_chunks(){
    CREC *crecs = (CREC *)malloc(sizeof(CREC) * shuffle_size);
    if (0 == crecs){
        logger.error("shuffle array allocate memory failure.\n");
        exit(EXIT_FAILURE);
    }

    FILE *fid = fopen("co-occurrence.bin", "rb");
    if (0 == fid){
        logger.error("can not open file co-occurrence.bin");
        exit(EXIT_FAILURE);
    }

    int __merge_idx = 0;
    LONG __idx = 0;
    char temp[1000];
    sprintf(temp, "shuffle.%04d.bin", __merge_idx);
    FILE *__chunk_file = fopen(temp, "wb");
    if (0 == __chunk_file){
        logger.error("can not open file __chunk_file");
        exit(EXIT_FAILURE);
    }

    while(1){
        if (__idx >= shuffle_size){
            shuffle(crecs, __idx);
            save_crecs(crecs, __idx, __chunk_file);
            fclose(__chunk_file);

            ++__merge_idx;
            sprintf(temp, "shuffle.%04d.bin", __merge_idx);
            __chunk_file = fopen(temp, "wb");
            if (0 == __chunk_file){
                logger.error("can not open file __chunk_file");
                exit(EXIT_FAILURE);
            }
            __idx = 0;
        }
        fread(&crecs[__idx], sizeof(CREC), 1, fid);
        ++__idx;
        if (feof(fid)) break;
    }
    shuffle(crecs, __idx);
    save_crecs(crecs, __idx, __chunk_file);
    fclose(__chunk_file);

    fclose(fid);
    free(crecs);
    remove("co-occurrence.bin");

    shuffle_merge(__merge_idx+1);
}

void glove::shuffle_merge(int merge_num){
    FILE **merges = (FILE **)malloc(sizeof(FILE) * merge_num);
    if (0 == merges){
        logger.error("merges allocate memory failure.\n");
        exit(EXIT_FAILURE);
    }

    FILE *shuffle_file = fopen("shuffle.bin", "wb");
    if (0 == shuffle_file){
        logger.error("can not open shuffle file.");
        exit(EXIT_FAILURE);
    }

    char temp[1000];
    for (int i = 0; i < merge_num; ++i){
        sprintf(temp, "shuffle.%04d.bin", i);
        merges[i] = fopen(temp, "rb");
        if (0 == merges[i]){
            logger.error("can not open merges[i].\n");
            exit(EXIT_FAILURE);
        }
    }

    CREC *crecs = (CREC *)malloc(sizeof(CREC) * shuffle_size);
    if (0 == crecs){
        logger.error("crecs allocate memory failure.\n");
        exit(EXIT_FAILURE);
    }

    clock_t start_time = clock();
    LONG __idx = 0;
    while(1){
        __idx = 0;
        for (int i = 0; i < merge_num; ++i){
            if (feof(merges[i])) continue;
            for (int k = 0; k < shuffle_size / merge_num; ++k){
                fread(&crecs[__idx], sizeof(CREC), 1, merges[i]);
                ++__idx;
                if (feof(merges[i])) break;
            }
        }
        if (__idx == 0) break;

        shuffle(crecs, __idx);
        save_crecs(crecs, __idx, shuffle_file);
    }
    for (int i = 0; i < merge_num; ++i){
        fclose(merges[i]);
        sprintf(temp, "shuffle.%04d.bin", i);
        remove(temp);
    }

    fclose(shuffle_file);
    free(crecs);
    free(merges);
    clock_t end_time = clock();

    sprintf(temp, "shuffle merged exhaust %ld s.", (end_time - start_time)/CLOCKS_PER_SEC);
    logger.info(temp);
}

int glove::optimize() {
    LONG a, file_size;
    int save_params_return_code;

    FILE *fin;
    real total_cost = 0;

    printf("========================>TRAINING MODEL ....\n");

    fin = fopen("shuffle.bin", "rb");
    if (0 == fin) {
        fprintf(stderr, "Can not open cooccurence file shuffle.bin.\n");
        return 1;
    }

    fseeko(fin, 0, SEEK_END);
    file_size = ftello(fin);

    num_lines = file_size / (sizeof(CREC));
    fclose(fin);

    printf("Read %lld lines.\n", num_lines);

    init_params();
    logger.info("Initializing parameters .....");

    if (verbose > 0) printf("embedding dim: %d\n", __embedding_dim);
    if (verbose > 0) printf("dict size: %lld\n", (LONG)__dict.size());
    if (verbose > 0) printf("x_max: %lf\n", x_max);
    if (verbose > 0) printf("alpha: %lf\n", alpha);

    vector<thread*> threads(__thread_num, 0);
    lines_per_thread = (LONG *)malloc(__thread_num * sizeof(LONG));

    time_t raw_time;
//    struct tm* info;

    char time_buffer[80];
    //Lock-free asynchronous SGD


    for (current_epoch = 0; current_epoch < __iter_num; ++current_epoch){
        total_cost = 0;
        for (a = 0; a < __thread_num - 1; ++a)
            lines_per_thread[a] = num_lines / __thread_num;
        lines_per_thread[a] = num_lines / __thread_num + num_lines % __thread_num;

        for (a = 0; a < __thread_num; ++a){
            threads[a] = new thread(glove_train_thread, a, cost, W, __embedding_dim, __thread_num, x_max, alpha,
                    eta, grad_sq, (LONG)__dict.size());
        }

        for (a = 0; a < __thread_num; ++a) threads[a]->join();

        for (a = 0; a < __thread_num; ++a) total_cost += cost[a];

        for (a = 0; a < __thread_num; ++a) delete threads[a];

        time(&raw_time);

        logger.info(std::to_string(current_epoch) + " epoch - total loss - " + std::to_string(total_cost / num_lines) + " - " + std::to_string(logger.get_diff_time()) + " s");

        if (checkpoint_every > 0 && (current_epoch + 1) % checkpoint_every == 0) {
            fprintf(stderr,"    saving intermediate parameters for iter %03d...", current_epoch + 1);
            save_params_return_code = save_params(current_epoch+1);
            if (save_params_return_code != 0)
                return save_params_return_code;
            fprintf(stderr,"done.\n");
        }

    }
    free(lines_per_thread);
    return 0;
//    return save_params(0);
}

void glove::init_params() {
    LONG a, b;
    ++__embedding_dim;

    a = posix_memalign((void **)&W, 128, 2 * __dict.size() * __embedding_dim * sizeof(real));
    if (W == 0){
        fprintf(stderr, "Error allocating memory for W\n");
        exit(1);
    }
    a = posix_memalign((void **)&grad_sq, 128, 2 * __dict.size() * __embedding_dim * sizeof(real));
    if (grad_sq == 0){
        fprintf(stderr, "Error allocating memory for grad_sq\n");
        exit(1);
    }

    for (b = 0; b < __embedding_dim; ++b){
        for (a = 0; a < 2 * __dict.size(); ++a){
            W[a * __embedding_dim + b] = (rand() / (real)RAND_MAX - 0.5) / __embedding_dim;
        }
    }

    for (b = 0; b < __embedding_dim; ++b){
        for (a = 0; a < 2 * __dict.size(); ++a){
            grad_sq[a * __embedding_dim + b] = 1.0;
        }
    }

    cost = (real *)malloc(sizeof(real) * __thread_num);
    if (0 == cost){
        exit(1);
    }

    --__embedding_dim;
}

int glove::save_params(int nb_iter){
    LONG a, b;
    char format[20];
    char output_file[MAX_STRING_SIZE], output_file_gsq[MAX_STRING_SIZE];
    char *word = (char *)malloc(sizeof(char) * MAX_STRING_SIZE + 1);
    FILE *fid, *fout, *fgs;

    if (__use_binary > 0){
        if (nb_iter <= 0) sprintf(output_file, "%s.bin", save_W_file);
        else sprintf(output_file, "%s.%03d.bin", save_W_file, nb_iter);

        fout = fopen(output_file, "wb");

        if (0 == fout){
            fprintf(stderr, "Unable to open file %s.\n", save_W_file);
            return 1;
        }

        for (a = 0; a < 2 * (LONG)__dict.size() * (__embedding_dim + 1); ++a)
            fwrite(&W[a], sizeof(real), 1, fout);
        fclose(fout);

        if (__save_grad_sq){
            if (nb_iter <= 0) sprintf(output_file_gsq, "%s.bin", save_grad_sq_file);
            else sprintf(output_file_gsq, "%s.%03d.bin", save_grad_sq_file, nb_iter);

            fgs = fopen(output_file_gsq, "wb");
            if (0 == fgs) {
                fprintf(stderr, "Unable to open file %s.\n", save_grad_sq_file);
                return 1;
            }

            for (a = 0; a < 2 * (LONG)__dict.size() * (__embedding_dim+1); ++a)
                fwrite(&grad_sq[a], sizeof(real), 1, fgs);
            fclose(fgs);
        }
    }

    if (__use_binary != 1){
        if (nb_iter <= 0) sprintf(output_file, "%s.txt", save_W_file);
        else sprintf(output_file, "%s.%03d.txt", save_W_file, nb_iter);

        if (__save_grad_sq > 0){
            if (nb_iter <= 0) sprintf(output_file_gsq, "%s.txt", save_grad_sq_file);
            else sprintf(output_file_gsq, "%s.%03d.txt", save_grad_sq_file, nb_iter);

            fgs = fopen(output_file_gsq, "wb");
            if (0 == fgs) {
                fprintf(stderr, "Unable to open file %s.\n", save_grad_sq_file);
                return 1;
            }
        }
        fout = fopen(output_file, "wb");
        if (fout == 0){
            fprintf(stderr, "Unable to open file %s.\n", save_W_file);
            return 1;
        }

        fid = fopen(vocab_file, "r");
        sprintf(format, "%%%ds", MAX_STRING_SIZE);
        if (fid == 0){
            fprintf(stderr, "Unable to open file %s.\n", vocab_file);
            return 1;
        }

        if (write_header) fprintf(fout, "%lld %d\n", (LONG)__dict.size(), __embedding_dim);

        for (a = 0; a < __dict.size(); ++a) {
            if (fscanf(fid, format, word) == 0) return 1;

            if (strcmp(word, "<unk>") == 0) return 1;
            fprintf(fout, "%s", word);

            if (__model == 0) {
                for (b = 0; b < (__embedding_dim + 1); b++) fprintf(fout, " %lf", W[a * (__embedding_dim + 1) + b]);
                for (b = 0; b < (__embedding_dim + 1); b++)
                    fprintf(fout, " %lf", W[(__dict.size() + a) * (__embedding_dim + 1) + b]);
            }
            if (__model == 1) {
                for (b = 0; b < (__embedding_dim + 1); b++) fprintf(fout, " %lf", W[a * (__embedding_dim + 1) + b]);
            }

            if (__model == 2) // Save "word + context word" vectors (without bias)
                for (b = 0; b < __embedding_dim; b++)
                    fprintf(fout, " %lf", W[a * (__embedding_dim + 1) + b] + W[(__dict.size() + a) *
                                                                               (__embedding_dim + 1) + b]);
            fprintf(fout, "\n");

            if (__save_grad_sq > 0) {
                fprintf(fgs, "%s", word);
                for (b = 0; b < (__embedding_dim + 1); ++b)
                    fprintf(fgs, " %lf", grad_sq[a * (__embedding_dim + 1) + b]);
                for (b = 0; b < (__embedding_dim + 1); ++b)
                    fprintf(fgs, " %lf", grad_sq[(__dict.size() + a) * (__embedding_dim + 1) + b]);
                fprintf(fgs, "\n");
            }
            if (fscanf(fid, format, word) == 0) return 1;
        }

            if (__use_unk_vec){
                real *unk_vec = (real*)calloc((__embedding_dim+1), sizeof(real));
                real *unk_context = (real *)calloc((__embedding_dim+1), sizeof(real));
                word = (char *)"<unk>";

                int num_rare_words = __dict.size() < 100 ? __dict.size(): 100;

                for ( a = __dict.size() - num_rare_words; a < __dict.size(); ++a){
                    for (b = 0; b < __embedding_dim+1; ++b){
                        unk_vec[b] += W[a * (__embedding_dim+1) + b] / num_rare_words;
                        unk_context[b] += W[(__dict.size() + a) * (__embedding_dim+1) + b] / num_rare_words;
                    }
                }

                fprintf(fout, "%s", word);
                if (__model == 0){
                    for (b = 0; b < (__embedding_dim + 1); ++b) fprintf(fout, " %lf", unk_vec[b]);
                    for (b = 0; b < (__embedding_dim + 1); ++b) fprintf(fout, " %lf", unk_context[b]);
                }

                if (__model == 1){
                    for (b = 0; b < __embedding_dim; ++b) fprintf(fout, " %lf", unk_vec[b]);
                }

                if (__model == 2){
                    for (b = 0; b < __embedding_dim; ++b) fprintf(fout, " %lf", unk_vec[b] + unk_context[b]);
                }
                fprintf(fout, "\n");

                free(unk_context);
                free(unk_vec);
            }

            fclose(fid);
            fclose(fout);

            if (__save_grad_sq > 0) fclose(fgs);

    }
    return 0;
}

#endif //FASTAI_GLOVE_H
