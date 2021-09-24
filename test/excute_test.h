//
// Created by mengqy on 2019/1/8.
//
#include "../bin/shape.h"
#include "../bin/rand.h"
#include "../bin/node.h"
#include "../bin/global_ops.h"
#include "../bin/word2vec.h"
#include "../bin/glove.h"
#include "../bin/sigmoid_table.h"
#include "../bin/tensor_ops.h"
#include "../bin/DNN.h"
#include "../bin/corpus.h"
#include "../bin/RNN.h"
#include "../bin/LSTM.h"
#include "../bin/config.h"
#include "../bin/ID3.h"
#include "../bin/C4.5.h"
#include "../bin/cart.h"
#include "../bin/base_trees.h"
#include "../bin/GBDT.h"
#include "../bin/xgboost.h"
#include "../bin/corn.h"
#include "../bin/hmm.h"
#include "../bin/CRF.h"
#include "../bin/SVM.h"
#include <numeric>


using namespace glop;
using namespace tops;

void shape_test() {
    int r[5] = {1, 2, 3, 4, 5};
    shape s(r, 5);
    std::cout << s.to_str() << std::endl;

    ofstream out("shape.bin");
    s.save(out);
    out.close();

    ifstream in("shape.bin");
    shape ss;
    ss.load(in);
    in.close();
    std::cout << "ss:" << ss.to_str() << std::endl;

    shape s0({2, 3, 4, 6, 8});
    std::cout << s0.to_str() << std::endl;

    vector<int> as = {2, 3, 4, 6, 8};
    if (s0 == as){
        std::cout << "相等\n";
    }

    if (s0 != as){
        std::cout << "不相等\n";
    } else std::cout << "相等\n";

    shape shape4 = as;

    std::cout << "shape4:" << shape4.to_str() << std::endl;

    int a[1] = {3, };
    shape s1(a, 1);
    std::cout << s1.to_str() << std::endl;

    shape shape1;
    shape1 = s.reduce();
    std::cout << shape1.to_str() << std::endl;

    shape1.assign(r+2, 3);
    std::cout << shape1.to_str() << std::endl;

    shape1.reduce().reduce();

    shape1.assign(r, 5);

    if (shape1 == s){
        std::cout << "相等\n";
    }
    std::cout << shape1.to_str() << std::endl;

    shape1.assign(r+2, 3);
    if (shape1 != s){
        std::cout << "不相等\n";
    }
    std::cout << shape1.to_str() << std::endl;

    shape c;
    c = shape1;
    std::cout << c.to_str() << std::endl;

    std::cout << c[0];

//    int b[5] = {};
//    shape1.assign(b, 5);

    int d[3] = {12, 5, 4};
    int e[2] = {5, 4};

    shape shape2(d, 3);
    shape shape3(e, 2);
    std::cout << shape3.to_str();
    if (shape::same_suffix_shape(shape2, shape3)) std::cout << "suffix shape\n";
    else std::cout << "not suffix shape\n";

    if (shape::same_prefix_shape(shape2, shape3)) std::cout << "prefix shape\n";
    else std::cout << "not prefix shape\n";
}

void tensor_test(){
    int a = 5;
    tensor<int> t(100, a);
    shape s = t.get_shape();
    std::cout << s.to_str() << std::endl;

    int b[3] = {1, 2, 3};
    shape shape1(b, 3);
    tensor<float> tensor1(shape1);
    shape s2 = tensor1.get_shape();
    std::cout << s2.to_str() << std::endl;

    tensor<float> tensor2(tensor1);
    std::cout << tensor1.to_str() << std::endl;

    tensor<float> tensor3;
    tensor3 = tensor1[0];
    std::cout << tensor3.to_str() << std::endl;
    std::cout << tensor3.show() << std::endl;

    shape shape2(b+1, 2);
    tensor<float> tensor4(shape2);
    uniform(tensor4, -1, 1);
    std::cout << tensor4.to_str() << std::endl;
    std::cout << tensor4.show() << std::endl;

    ofstream out("ten.bin");
    tensor4.save(out);
    out.close();

    ifstream in("ten.bin");
    tensor<real> tensor8;
    tensor8.load(in);
    in.close();
    std::cout << "tensor8:";
    std::cout << tensor8.show() << std::endl;
    std::cout << tensor8.to_str() << std::endl;

    tensor1[0] = tensor4;
    std::cout << tensor1[0].to_str() << std::endl;
    std::cout << tensor1[0].show() << std::endl;

//    uniform(tensor1[0], -2, 1);
    std::cout << tensor1[0].to_str() << std::endl;
    std::cout << tensor1[0].show() << std::endl;

//    std::cout << tensor1[0].get(1, 2) << std::endl;
//
//    tensor1[0].get(1, 2) = 33.0;

//    uniform(tensor1[0], -2, 1);
    std::cout << tensor1[0].to_str() << std::endl;
    std::cout << tensor1[0].show() << std::endl;

    tensor<float> tensor5;
    tensor5 = tensor1[0][1];

//    std::cout << tensor5.get(2) << std::endl;
//    tensor5.get(2) = 9.0;
//    std::cout << tensor5.get(2) << std::endl;
    std::cout << tensor5.show() << std::endl;
    std::cout << tensor5.to_str() << std::endl;

    int d[1] = {1};
    shape shape3(d, 1);
    tensor<float> tensor6(shape3);

//    std::cout << tensor6.get(0) << std::endl;
    std::cout << tensor6.to_str() << std::endl;
    std::cout << tensor6.show() << std::endl;

    tensor<float> tensor7({1, 2, 3, 5});

    std::cout << tensor7.to_str();
}

void variable_test(){

//    int a[2] = {10, 20};
//    shape shape1(a, 2);
//    variable variable1(shape1, "weights");
//    uniform(variable1.get(), -1, 2);
//
//    std::cout << variable1.get().to_str() << std::endl;
//    std::cout << variable1.get().show() << std::endl;
//
//    std::cout << variable1.to_str() << std::endl;
//
//    variable variable2(shape1, "weights1");
//    uniform(variable2.get(), -1, 2);
//
//    std::cout << variable2.get().to_str() << std::endl;
//    std::cout << variable2.get().show() << std::endl;
//
//    std::cout << variable2.to_str() << std::endl;
//
//    variable variable3(shape1, "weights2");
//    uniform(variable3.get(), -1, 2);

//    std::cout << variable3.get().to_str() << std::endl;
//    std::cout << variable3.get().show() << std::endl;
//
//    std::cout << variable3.to_str() << std::endl;
//
//    glop::add(variable3, variable2, variable1, true);
////    variable1.add();
//    std::cout << variable1.to_str() << std::endl;
//    std::cout << variable2.to_str() << std::endl;
//    std::cout << variable3.to_str() << std::endl;

};

void word2vec_test(){
    word2vec w;

    w.train("/Users/mengqy/git/GloVe/predict.txt", "embedding-model");
}

void glove_test(){
    glove g;
    g.train("/Users/mengqy/git/GloVe/predict.txt", "");
}

void sigmoid_table_test(){
    sigmoid_table sigmoidTable;

    for (int i = 0; i < 1000; ++i)
        std::cout << sigmoidTable[i] << '\n';
}

void transpose_test(){
    std::cout << "TRANSPOSE TEST:\n";
    int a[2] = {5, 6};
    shape s(a, 2);
    tensor<float> tensor1(s);

    clock_t start0 = clock();
    uniform(tensor1, -1, 1);
    clock_t end0 = clock();
    std::cout << "uniform(tensor1, -1, 1) exhaust time = "
              << (end0 - start0)/CLOCKS_PER_SEC << "s" << std::endl;

    clock_t start = clock();
    tensor<float> tensor2;
    transpose(tensor1, tensor2);

    std::cout << "tensor1:\n" << tensor1.show() << std::endl
              << "tensor2:\n" << tensor2.show() << std::endl;
    clock_t end = clock();
    std::cout << "transpose(tensor1, tensor2) exhaust time = "
              << (end - start)/CLOCKS_PER_SEC  << "s" << std::endl;


    std::cout << "--------------------\n";

    clock_t start1 = clock();
    tensor<float> tensor3 = transpose(tensor1);
    clock_t end1 = clock();
    std::cout << "tensor1:\n" << tensor1.show() << std::endl
              << "tensor3:\n" << tensor3.show();
    std::cout << "\ntensor3 = transpose(tensor1) exhaust time = "
              << (end1 - start1)/CLOCKS_PER_SEC << "s" << std::endl;

}

void dot_mul_test(){
    std::cout << "DOT MUL TEST:\n";
    float a[12];
    for (int i = 0; i < 12; ++i)
        a[i] = i;

    int b[2] = {3, 4};
    shape s(b, 2);
    tensor<float> tensor1(s);
    assign(tensor1, a, 12);

    shape shape1(b, 2);
    tensor<float> tensor2(shape1);
    assign(tensor2, a, 12);

    tensor<float> res;
    dot_mul(tensor1, tensor2, res, true);

    string d = tensor1.show();
    string e = tensor2.show();
    string f = res.show();

    std::cout << "dot_mul(tensor1, tensor2, res):\n";
    std::cout << "tensor1:\n"<< d << "\ntensor2:\n" << e << "\nres:\n" << f << "\n\n";

    dot_mul(tensor1, tensor2, res, false, true);
    d = tensor1.show();
    e = tensor2.show();
    f = res.show();
    std::cout << "dot_mul(tensor1, tensor2, res, false, true):\n";
    std::cout << "tensor1:\n"<< d << "\ntensor2:\n" << e << "\nres:\n" << f << "\n\n";

    tensor1 = transpose(tensor1);

    dot_mul(tensor1, tensor2, res, true, true);
    d = tensor1.show();
    e = tensor2.show();
    f = res.show();
    std::cout << "dot_mul(tensor1, tensor2, res, true, true):\n";
    std::cout << "tensor1:\n"<< d << "\ntensor2:\n" << e << "\nres:\n" << f << "\n\n";

    tensor1 = transpose(tensor1);

    int c[2] = {4, 3};
    shape shape2(c, 2);
    tensor<float> tensor3(shape2);
    assign(tensor3, a, 12);
    tensor<float> res0;
    dot_mul(tensor1, tensor3, res0, false);

    string g = tensor1.show();
    string h = tensor3.show();
    string i = res0.show();

    std::cout << "dot_mul(tensor1, tensor3, res0, false):\n";
    std::cout << "tensor1:\n" << g << "\ntensor3:\n" << h << "\nres0:\n" << i << "\n\n";

    dot_mul(tensor1, tensor3, res0, false, false);

    g = tensor1.show();
    h = tensor3.show();
    i = res0.show();

    std::cout << "dot_mul(tensor1, tensor3, res0, false, false):\n";
    std::cout << "tensor1:\n" << g << "\ntensor3:\n" << h << "\nres0:\n" << i << "\n\n";

    tensor1 = transpose(tensor1);
    dot_mul(tensor1, tensor3, res0, true, false);

    g = tensor1.show();
    h = tensor3.show();
    i = res0.show();

    std::cout << "dot_mul(tensor1, tensor3, res0, true, false):\n";
    std::cout << "tensor1:\n" << g << "\ntensor3:\n" << h << "\nres0:\n" << i << "\n\n";


    tensor1 = transpose(tensor1);

    int C[2] = {2, 1};
    shape shape3(C, 2);
    tensor<float> ids(shape3);
    float D[2] = {0, 1};
    assign(ids, D, 2);
    dot_mul(tensor1, tensor3, res0, false, ids);

    g = tensor1.show();
    h = tensor3.show();
    i = res0.show();

    std::cout << "dot_mul(tensor1, tensor3, res0, false, ids), ids={0, 1}:\n";
    std::cout << "tensor1:\n"<< g << "\ntensor3:\n" << h << "\nres0:\n" << i << std::endl;

    dot_mul(tensor1, tensor2, res, true, ids);
    g = tensor1.show();
    h = tensor2.show();
    i = res.show();
    std::cout << "\ndot_mul(tensor1, tensor2, res, true, ids), ids={0, 1}:\n";
    std::cout << "tensor1:\n"<< g << "\ntensor2:\n" << h << "\nres:\n" << i << std::endl;
}

void embeddings_lookup_test(){
//    std::cout << "EMBEDDINGS LOOKUP TEST:\n";
//    int a[2] = {10000, 128};
//    shape shape1(a, 2);
//    tensor<float> embeddings(shape1);
//    uniform(embeddings, -1.0, 1.0);
//
//    int b[2] = {100, 20};
//    shape shape2(b, 2);
//    tensor<LONG> ids(shape2);
//    LONG c[2000];
//    for (int i = 0; i < 2000; ++i)
//        c[i] = i;
//    assign(ids, c, 2000);
//
//    tensor<float> lookups;
//    embeddings_lookup(embeddings, ids, lookups);
//
//    std::cout << "embeddings: "<< embeddings.to_str() << "\n";
//    std::cout << "lookup: "<< lookups.to_str() << "\n";
//
//    std::cout << "embeddings[0:20, :]:\n";
//    for (int i = 0; i < 20; ++i)
//        std::cout << embeddings[i].show() << std::endl;
//    std::cout << "lookups[0]:\n";
//    std::cout << lookups[0].to_str() << std::endl << lookups[0].show() << std::endl;
//
//    std::cout << "lookups[0][10]\n";
//    std::cout << lookups[0][10].to_str() << std::endl;
//    std::cout << lookups[0][10].show();


    int cc[2] = {10, 12};
    shape shape4(cc, 2);
    tensor<float> embeddings00(shape4);
    uniform(embeddings00, -1.0, 1.0);

    vector<LONG> ides = {3, 6, 7};
    tensor<float> res;
    embeddings_lookup(embeddings00, ides, res);
    std::cout << embeddings00.show() << std::endl;
    std::cout << res.show() << std::endl;
}

void reduce_mean_test(){
    std::cout << "REDUCE MEAN TEST:\n";

    int a[3] = {100, 256, 128};
    shape shape1(a, 3);
    tensor<float> embeddings(shape1);
    uniform(embeddings, -1.0, 1.0);

    clock_t start = clock();
    tensor<float> res;
    reduce_mean(embeddings, res, 1);
    clock_t end = clock();
    std::cout << res.to_str() << std::endl
              << "exhaust time:" << (end - start)
              << "us, " << (end - start)/CLOCKS_PER_SEC << "s\n";

    std::cout << "------------------------------\n";
    int b[2] = {10, 5};

    shape shape2(b, 2);
    tensor<float> tensor1(shape2);
    uniform(tensor1, -1.0, 1.0);

    tensor<float> res0;
    std::cout << "reduce_mean(tensor1, res0, 0):\n";
    reduce_mean(tensor1, res0, 0);

    std::cout << "tensor1:\n" << tensor1.to_str() << std::endl
              << tensor1.show() << std::endl << "res0:\n" << res0.to_str() << std::endl
              << res0.show() << std::endl;

    std::cout << "------------------------------\n";
    reduce_mean(tensor1, res0, 1);

    std::cout << "reduce_mean(tensor1, res0, 1):\n";
    std::cout << "tensor1:\n" << tensor1.to_str() << std::endl
              << tensor1.show() << std::endl << "res0:\n" << res0.to_str() << std::endl
              << res0.show() << std::endl;

    std::cout << "------------------------------\n";

    int d[3] = {3, 4, 5};
    shape shape3(d, 3);
    tensor<float> tensor2(shape3);

    float t[60];
    for (int i = 0; i < 60; ++i)
        t[i] = i;
    assign(tensor2, t, 60);

    tensor<float> res2;
    reduce_mean(tensor2, res2, 1);
    std::cout << "reduce_mean(tensor2, res2, 1):\n"
              << "tensor2:\n" << tensor2.show() << std::endl
              << "tensor2[0]:\n" << tensor2[0].to_str() << std::endl
              << tensor2[0].show() << std::endl
              << "res2[0]:\n" << res2[0].to_str() << std::endl << res2[0].show() << std::endl;

    std::cout << "------------------------------\n";

    reduce_mean(tensor2, res2, 0);
    std::cout << "reduce_mean(tensor2, res2, 0):\n"
              << "tensor2:\n" << tensor2.show() << std::endl
              << "tensor2[0]:\n" << tensor2[0].to_str() << std::endl << tensor2[0].show() << std::endl
              << "tensor2[1]:\n" << tensor2[1].to_str() << std::endl << tensor2[1].show() << std::endl
              << "tensor2[2]:\n" << tensor2[2].to_str() << std::endl << tensor2[2].show() << std::endl
              << "res2:\n" << res2.to_str() << std::endl << res2.show() << std::endl;

    std::cout << "------------------------------\n";

    reduce_mean(tensor2, res2, 2);
    std::cout << "reduce_mean(tensor2, res2, 2):\n"
              << "tensor2:\n" << tensor2.show() << std::endl
              << "tensor2[0]:\n" << tensor2[0].to_str() << std::endl << tensor2[0].show() << std::endl
              << "tensor2[1]:\n" << tensor2[1].to_str() << std::endl << tensor2[1].show() << std::endl
              << "tensor2[2]:\n" << tensor2[2].to_str() << std::endl << tensor2[2].show() << std::endl
              << "res2:\n" << res2.to_str() << std::endl << res2.show();
}

void reduce_sum_test(){
    std::cout << "REDUCE SUM TEST:\n";

    int a[3] = {100, 256, 128};
    shape shape1(a, 3);
    tensor<float> embeddings(shape1);
    uniform(embeddings, -1.0, 1.0);

    clock_t start = clock();
    tensor<float> res;
    reduce_sum(embeddings, res, 1);
    clock_t end = clock();
    std::cout << res.to_str() << std::endl
              << "exhaust time:" << (end - start)
              << "us, " << (end - start)/CLOCKS_PER_SEC << "s\n";

    std::cout << "------------------------------\n";
    int b[2] = {10, 5};

    shape shape2(b, 2);
    tensor<float> tensor1(shape2);
    uniform(tensor1, -1.0, 1.0);

    tensor<float> res0;
    std::cout << "reduce_sum(tensor1, res0, 0):\n";
    reduce_sum(tensor1, res0, 0);

    std::cout << "tensor1:\n" << tensor1.to_str() << std::endl
              << tensor1.show() << std::endl << "res0:\n" << res0.to_str() << std::endl
              << res0.show() << std::endl;

    std::cout << "------------------------------\n";
    reduce_sum(tensor1, res0, 1);

    std::cout << "reduce_sum(tensor1, res0, 1):\n";
    std::cout << "tensor1:\n" << tensor1.to_str() << std::endl
              << tensor1.show() << std::endl << "res0:\n" << res0.to_str() << std::endl
              << res0.show() << std::endl;

    std::cout << "------------------------------\n";

    int d[3] = {3, 4, 5};
    shape shape3(d, 3);
    tensor<float> tensor2(shape3);

    float t[60];
    for (int i = 0; i < 60; ++i)
        t[i] = i;
    assign(tensor2, t, 60);

    tensor<float> res2;
    reduce_sum(tensor2, res2, 1);
    std::cout << "reduce_sum(tensor2, res2, 1):\n"
              << "tensor2:\n" << tensor2.show() << std::endl
              << "tensor2[0]:\n" << tensor2[0].to_str() << std::endl
              << tensor2[0].show() << std::endl
              << "res2[0]:\n" << res2[0].to_str() << std::endl << res2[0].show() << std::endl;

    std::cout << "------------------------------\n";

    reduce_sum(tensor2, res2, 0);
    std::cout << "reduce_sum(tensor2, res2, 0):\n"
              << "tensor2:\n" << tensor2.show() << std::endl
              << "tensor2[0]:\n" << tensor2[0].to_str() << std::endl << tensor2[0].show() << std::endl
              << "tensor2[1]:\n" << tensor2[1].to_str() << std::endl << tensor2[1].show() << std::endl
              << "tensor2[2]:\n" << tensor2[2].to_str() << std::endl << tensor2[2].show() << std::endl
              << "res2:\n" << res2.to_str() << std::endl << res2.show() << std::endl;

    std::cout << "------------------------------\n";

    reduce_sum(tensor2, res2, 2);
    std::cout << "reduce_sum(tensor2, res2, 2):\n"
              << "tensor2:\n" << tensor2.show() << std::endl
              << "tensor2[0]:\n" << tensor2[0].to_str() << std::endl << tensor2[0].show() << std::endl
              << "tensor2[1]:\n" << tensor2[1].to_str() << std::endl << tensor2[1].show() << std::endl
              << "tensor2[2]:\n" << tensor2[2].to_str() << std::endl << tensor2[2].show() << std::endl
              << "res2:\n" << res2.to_str() << std::endl << res2.show();
}

void same_shape_test(){
    std::cout << "SAME SHAPE TEST:\n";
    int a[3] = {3, 4, 5};
    shape __a_shape(a, 3);

    int b[2] = {3, 4};
    shape __b_shape(b, 2);

    if (shape::same_prefix_shape(__a_shape, __b_shape)){
        std::cout << "a_shape{3, 4, 5}, b_shape{3, 4} same_prefix_shape(a_shape, b_shape)\n";
    }

    int c[2] = {3, 1};
    shape __c_shape(c, 2);

    if (shape::same_prefix_shape(__a_shape, __c_shape)){
        std::cout << "a_shape{3, 4, 5}, c_shape{3, 1} same_prefix_shape(a_shape, c_shape)\n";
    }

    int e[3] = {3, 4, 1};
    shape __e_shape(e, 3);

    if (shape::same_prefix_shape(__a_shape, __e_shape)){
        std::cout << "a_shape{3, 4, 5}, e_shape{3, 4, 1} same_prefix_shape(a_shape, e_shape)\n";
    }

    int d[3] = {1, 4, 5};
    shape __d_shape(d, 3);

    if (shape::same_suffix_shape(__a_shape, __d_shape)){
        std::cout << "a_shape{3, 4, 5}, d_shape{1, 4, 5} same_suffix_shape(a_shape, d_shape)\n";
    }

    int g[2] = {4, 5};
    shape __g_shape(g, 2);

    if (shape::same_suffix_shape(__a_shape, __g_shape)){
        std::cout << "a_shape{3, 4, 5}, g_shape{4, 5} same_suffix_shape(a_shape, g_shape)\n";
    }

    int h[2] = {1, 5};
    shape __h_shape(h, 2);

    if (shape::same_suffix_shape(__a_shape, __h_shape)){
        std::cout << "a_shape{3, 4, 5}, h_shape{1, 5} same_suffix_shape(a_shape, h_shape)\n";
    }

    int i[1] = {5};
    shape __i_shape(i, 1);
    if (shape::same_suffix_shape(__a_shape, __i_shape)){
        std::cout << "a_shape{3, 4, 5}, i_shape{5} same_suffix_shape(a_shape, i_shape)\n";
    }

    if (shape::same_prefix_shape(__a_shape, __i_shape)){
        std::cout << "a_shape{3, 4, 5}, i_shape{5} same_prefix_shape(a_shape, i_shape)\n";
    }

    int j[1] = {1};
    shape __j_shape(j, 1);
    if (shape::same_suffix_shape(__a_shape, __j_shape)){
        std::cout << "a_shape{3, 4, 5}, j_shape{1} same_suffix_shape(a_shape, j_shape)\n";
    }

    if (shape::same_prefix_shape(__a_shape, __j_shape)){
        std::cout << "a_shape{3, 4, 5}, j_shape{1} same_prefix_shape(a_shape, j_shape)\n";
    }
}

void add_test(){
    std::cout << "ADD TEST:\n";
    int a[2] = {3, 4};
    shape shape1(a, 2);
    tensor<float> tensor1(shape1);

    int e[1] = {4};
    shape __e_shape(e, 1);
    tensor<float> tensor2(__e_shape);

    float b[12];
    for (int i = 0; i < 12; ++i)
        b[i] = i;
    assign(tensor1, b, 12);

    assign(tensor2, b, 4);

    tensor<float> tensor3;
    add_suffix(tensor1, tensor2, tensor3);
    std::cout << "\n-----------------------------\n";
    std::cout << "add_suffix(tensor1, tensor2, tensor3):\n";
    std::cout << "tensor1:"  << tensor1.to_str() << std::endl << tensor1.show() << std::endl;
    std::cout << "tensor2:"  << tensor2.to_str() << std::endl << tensor2.show() << std::endl;
    std::cout << "tensor3:"  << tensor3.to_str() << std::endl << tensor3.show() << std::endl;

    std::cout << "\n-----------------------------\n";

    int h[1] = {1};
    shape __h_shape(h, 1);
    tensor<float> tensor4(__h_shape);

    float hh[1] = {3.0};
    assign(tensor4, hh, 1);
    tensor<float> tensor5;
    add_suffix(tensor1, tensor4, tensor5);
    std::cout << "add_suffix(tensor1, tensor4, tensor5):\n";
    std::cout << "tensor1:"  << tensor1.to_str() << std::endl << tensor1.show() << std::endl;
    std::cout << "tensor4:"  << tensor4.to_str() << std::endl << tensor4.show() << std::endl;
    std::cout << "tensor5:"  << tensor5.to_str() << std::endl << tensor5.show() << std::endl;


    int d[1] = {3};
    shape __d_shape(d, 1);
    tensor<float> tensor6(__d_shape);
    assign(tensor6, b, 3);

    add_prefix(tensor1, tensor6, tensor3);
    std::cout << "\n-----------------------------\n";
    std::cout << "add_prefix(tensor1, tensor6, tensor3):\n";
    std::cout << "tensor1:"  << tensor1.to_str() << std::endl << tensor1.show() << std::endl;
    std::cout << "tensor6:"  << tensor6.to_str() << std::endl << tensor6.show() << std::endl;
    std::cout << "tensor3:"  << tensor3.to_str() << std::endl << tensor3.show() << std::endl;

    std::cout << "\n-----------------------------\n";

    add_prefix(tensor1, tensor4, tensor5);
    std::cout << "add_prefix(tensor1, tensor4, tensor5):\n";
    std::cout << "tensor1:"  << tensor1.to_str() << std::endl << tensor1.show() << std::endl;
    std::cout << "tensor4:"  << tensor4.to_str() << std::endl << tensor4.show() << std::endl;
    std::cout << "tensor5:"  << tensor5.to_str() << std::endl << tensor5.show() << std::endl;


    std::cout << "\n-----------------------------\n";

    tops::add(tensor1, tensor1, tensor5);
    std::cout << "add(tensor1, tensor1, tensor5):\n";
    std::cout << "tensor1:"  << tensor1.to_str() << std::endl << tensor1.show() << std::endl;
    std::cout << "tensor5:"  << tensor5.to_str() << std::endl << tensor5.show() << std::endl;
}

void subtract_test(){
    std::cout << "SUBTRACT TEST:\n";
    int a[2] = {3, 4};
    shape shape1(a, 2);
    tensor<float> tensor1(shape1);

    int e[1] = {4};
    shape __e_shape(e, 1);
    tensor<float> tensor2(__e_shape);

    float b[12];
    for (int i = 0; i < 12; ++i)
        b[i] = i;
    assign(tensor1, b, 12);

    assign(tensor2, b, 4);

    tensor<float> tensor3;
    subtract_suffix(tensor1, tensor2, tensor3);
    std::cout << "\n-----------------------------\n";
    std::cout << "subtract_suffix(tensor1, tensor2, tensor3):\n";
    std::cout << "tensor1:"  << tensor1.to_str() << std::endl << tensor1.show() << std::endl;
    std::cout << "tensor2:"  << tensor2.to_str() << std::endl << tensor2.show() << std::endl;
    std::cout << "tensor3:"  << tensor3.to_str() << std::endl << tensor3.show() << std::endl;

    std::cout << "\n-----------------------------\n";

    int h[1] = {1};
    shape __h_shape(h, 1);
    tensor<float> tensor4(__h_shape);

    float hh[1] = {3.0};
    assign(tensor4, hh, 1);
    tensor<float> tensor5;
    subtract_suffix(tensor1, tensor4, tensor5);
    std::cout << "subtract_suffix(tensor1, tensor4, tensor5):\n";
    std::cout << "tensor1:"  << tensor1.to_str() << std::endl << tensor1.show() << std::endl;
    std::cout << "tensor4:"  << tensor4.to_str() << std::endl << tensor4.show() << std::endl;
    std::cout << "tensor5:"  << tensor5.to_str() << std::endl << tensor5.show() << std::endl;


    int d[1] = {3};
    shape __d_shape(d, 1);
    tensor<float> tensor6(__d_shape);

//    int bb[12];
//    for (int i = 0; i < 12; ++i)
//        bb[i] = i;
    assign(tensor6, b, 3);

    subtract_prefix(tensor1, tensor6, tensor3);
    std::cout << "\n-----------------------------\n";
    std::cout << "subtract_prefix(tensor1, tensor6, tensor3):\n";
    std::cout << "tensor1:"  << tensor1.to_str() << std::endl << tensor1.show() << std::endl;
    std::cout << "tensor6:"  << tensor6.to_str() << std::endl << tensor6.show() << std::endl;
    std::cout << "tensor3:"  << tensor3.to_str() << std::endl << tensor3.show() << std::endl;

    std::cout << "\n-----------------------------\n";

    subtract_prefix(tensor1, tensor4, tensor5);
    std::cout << "subtract_prefix(tensor1, tensor4, tensor5):\n";
    std::cout << "tensor1:"  << tensor1.to_str() << std::endl << tensor1.show() << std::endl;
    std::cout << "tensor4:"  << tensor4.to_str() << std::endl << tensor4.show() << std::endl;
    std::cout << "tensor5:"  << tensor5.to_str() << std::endl << tensor5.show() << std::endl;


    std::cout << "\n-----------------------------\n";

    tops::subtract(tensor1, tensor1, tensor5);
    std::cout << "subtract(tensor1, tensor1, tensor5):\n";
    std::cout << "tensor1:"  << tensor1.to_str() << std::endl << tensor1.show() << std::endl;
    std::cout << "tensor5:"  << tensor5.to_str() << std::endl << tensor5.show() << std::endl;
}

void subtract_max_test(){
    std::cout << "SUBTRACT MAX TEST:\n";
    int a[2] = {3, 4};
    shape shape1(a, 2);
    tensor<float> tensor1(shape1);

    float b[12];
    for (int i = 0; i < 12; ++i)
        b[i] = i;
    assign(tensor1, b, 12);

    std::cout << tensor1.show() << std::endl;

    subtract_max(tensor1, tensor1);

    std::cout << tensor1.show() << std::endl;
}

void multiply_test(){
    std::cout << "MULTIPLY TEST:\n";
    int a[2] = {3, 4};
    shape shape1(a, 2);
    tensor<float> tensor1(shape1);

    int e[1] = {4};
    shape __e_shape(e, 1);
    tensor<float> tensor2(__e_shape);

    float b[12];
    for (int i = 0; i < 12; ++i)
        b[i] = i;
    assign(tensor1, b, 12);

    assign(tensor2, b, 4);

    tensor<float> tensor3;
    multiply_suffix(tensor1, tensor2, tensor3);
    std::cout << "\n-----------------------------\n";
    std::cout << "multiply_suffix(tensor1, tensor2, tensor3):\n";
    std::cout << "tensor1:"  << tensor1.to_str() << std::endl << tensor1.show() << std::endl;
    std::cout << "tensor2:"  << tensor2.to_str() << std::endl << tensor2.show() << std::endl;
    std::cout << "tensor3:"  << tensor3.to_str() << std::endl << tensor3.show() << std::endl;

    std::cout << "\n-----------------------------\n";

    int h[1] = {1};
    shape __h_shape(h, 1);
    tensor<float> tensor4(__h_shape);

    float hh[1] = {3.0};
    assign(tensor4, hh, 1);
    tensor<float> tensor5;
    multiply_suffix(tensor1, tensor4, tensor5);
    std::cout << "multiply_suffix(tensor1, tensor4, tensor5):\n";
    std::cout << "tensor1:"  << tensor1.to_str() << std::endl << tensor1.show() << std::endl;
    std::cout << "tensor4:"  << tensor4.to_str() << std::endl << tensor4.show() << std::endl;
    std::cout << "tensor5:"  << tensor5.to_str() << std::endl << tensor5.show() << std::endl;


    int d[1] = {3};
    shape __d_shape(d, 1);
    tensor<float> tensor6(__d_shape);
    assign(tensor6, b, 3);

    multiply_prefix(tensor1, tensor6, tensor3);
    std::cout << "\n-----------------------------\n";
    std::cout << "multiply_prefix(tensor1, tensor6, tensor3):\n";
    std::cout << "tensor1:"  << tensor1.to_str() << std::endl << tensor1.show() << std::endl;
    std::cout << "tensor6:"  << tensor6.to_str() << std::endl << tensor6.show() << std::endl;
    std::cout << "tensor3:"  << tensor3.to_str() << std::endl << tensor3.show() << std::endl;

    std::cout << "\n-----------------------------\n";

    multiply_prefix(tensor1, tensor4, tensor5);
    std::cout << "multiply_prefix(tensor1, tensor4, tensor5):\n";
    std::cout << "tensor1:"  << tensor1.to_str() << std::endl << tensor1.show() << std::endl;
    std::cout << "tensor4:"  << tensor4.to_str() << std::endl << tensor4.show() << std::endl;
    std::cout << "tensor5:"  << tensor5.to_str() << std::endl << tensor5.show() << std::endl;


    std::cout << "\n-----------------------------\n";

    tops::multiply(tensor1, tensor1, tensor5);
    std::cout << "multiply(tensor1, tensor1, tensor5):\n";
    std::cout << "tensor1:"  << tensor1.to_str() << std::endl << tensor1.show() << std::endl;
    std::cout << "tensor5:"  << tensor5.to_str() << std::endl << tensor5.show() << std::endl;


    vector<int> v = {12800, 6400};
    tensor<real> tensor7(v);

    uniform(tensor7, -1, 1);

    clock_t start = clock();
    tops::multiply(tensor7, tensor7, tensor5);
//    std::cout << "multiply(tensor7, tensor7, tensor5):\n";
//    std::cout << "tensor7:"  << tensor1.to_str() << std::endl << tensor1.show() << std::endl;
//    std::cout << "tensor5:"  << tensor5.to_str() << std::endl << tensor5.show() << std::endl;
    clock_t end = clock();

    std::cout << (end - start) << std::endl;

}


void divide_test(){
    std::cout << "DIVIDE TEST:\n";
    int a[2] = {3, 4};
    shape shape1(a, 2);
    tensor<float> tensor1(shape1);

    int e[1] = {4};
    shape __e_shape(e, 1);
    tensor<float> tensor2(__e_shape);

    float b[12];
    for (int i = 0; i < 12; ++i)
        b[i] = i + 1;
    assign(tensor1, b, 12);

    assign(tensor2, b, 4);

    tensor<float> tensor3;
    divide_suffix(tensor1, tensor2, tensor3);
    std::cout << "\n-----------------------------\n";
    std::cout << "divide_suffix(tensor1, tensor2, tensor3):\n";
    std::cout << "tensor1:"  << tensor1.to_str() << std::endl << tensor1.show() << std::endl;
    std::cout << "tensor2:"  << tensor2.to_str() << std::endl << tensor2.show() << std::endl;
    std::cout << "tensor3:"  << tensor3.to_str() << std::endl << tensor3.show() << std::endl;

    std::cout << "\n-----------------------------\n";

    int h[1] = {1};
    shape __h_shape(h, 1);
    tensor<float> tensor4(__h_shape);

    float hh[1] = {3.0};
    assign(tensor4, hh, 1);
    tensor<float> tensor5;
    divide_suffix(tensor1, tensor4, tensor5);
    std::cout << "divide_suffix(tensor1, tensor4, tensor5):\n";
    std::cout << "tensor1:"  << tensor1.to_str() << std::endl << tensor1.show() << std::endl;
    std::cout << "tensor4:"  << tensor4.to_str() << std::endl << tensor4.show() << std::endl;
    std::cout << "tensor5:"  << tensor5.to_str() << std::endl << tensor5.show() << std::endl;


    int d[1] = {3};
    shape __d_shape(d, 1);
    tensor<float> tensor6(__d_shape);
    assign(tensor6, b, 3);

    divide_prefix(tensor1, tensor6, tensor3);
    std::cout << "\n-----------------------------\n";
    std::cout << "divide_prefix(tensor1, tensor6, tensor3):\n";
    std::cout << "tensor1:"  << tensor1.to_str() << std::endl << tensor1.show() << std::endl;
    std::cout << "tensor6:"  << tensor6.to_str() << std::endl << tensor6.show() << std::endl;
    std::cout << "tensor3:"  << tensor3.to_str() << std::endl << tensor3.show() << std::endl;

    std::cout << "\n-----------------------------\n";

    divide_prefix(tensor1, tensor4, tensor5);
    std::cout << "divide_prefix(tensor1, tensor4, tensor5):\n";
    std::cout << "tensor1:"  << tensor1.to_str() << std::endl << tensor1.show() << std::endl;
    std::cout << "tensor4:"  << tensor4.to_str() << std::endl << tensor4.show() << std::endl;
    std::cout << "tensor5:"  << tensor5.to_str() << std::endl << tensor5.show() << std::endl;

    std::cout << "\n-----------------------------\n";

    tops::divide(tensor1, tensor1, tensor5);
    std::cout << "divide(tensor1, tensor1, tensor5):\n";
    std::cout << "tensor1:"  << tensor1.to_str() << std::endl << tensor1.show() << std::endl;
    std::cout << "tensor5:"  << tensor5.to_str() << std::endl << tensor5.show() << std::endl;
}

void sigmoid_test(){
    std::cout << "SIGMOID TEST:\n";
    clock_t start = clock();
    int a[2] = {30000, 40000};
    shape shape1(a, 2);
    tensor<float> tensor1(shape1);
    tensor<float> tensor2(shape1);

    float *b = new float[1200000000];
    for (int i = 0; i < 1200000000; ++i)
        b[i] = i;
    assign(tensor1, b, 1200000000);
    tops::sigmoid(tensor1, tensor2);
    clock_t end = clock();
    std::cout << "耗时:" << (end - start) / CLOCKS_PER_SEC << "s\n";

    int d[2] = {3, 4};
    shape __d_shape(d, 2);
    tensor<float> tensor3(__d_shape);
    assign(tensor3, b, 12);
    tensor<float> tensor4;
    tops::sigmoid(tensor3, tensor4);
    std::cout << "sigmoid(tensor3, tensor4):\n";
    std::cout << "tensor3:" << tensor3.to_str() << std::endl << tensor3.show()
              << "\ntensor4:" << tensor4.to_str() << std::endl << tensor4.show();

    if (0 != b) delete [] b;
}

void softmax_test() {
    std::cout << "SOFTMAX TEST:\n";
    int a[2] = {3, 4};
    shape shape1(a, 2);
    tensor<float> tensor1(shape1);
    tensor<float> tensor2(shape1);

    float *b = new float[12];
    for (int i = 0; i < 12; ++i)
        b[i] = i;
    assign(tensor1, b, 12);
    clock_t start = clock();
    tops::softmax(tensor1, tensor2);
    clock_t end = clock();

    std::cout << "softmax(tensor1, tensor2):\n";
    std::cout << "tensor1:\n" << tensor1.to_str()
              << std::endl << tensor1.show() << std::endl
              << "tensor2:\n" << tensor2.to_str()
              << std::endl << tensor2.show() << std::endl
              << (end - start) / CLOCKS_PER_SEC << "s\n";

    delete [] b;
}

void log_test(){
    std::cout << "LOG TEST:\n";
    int a[2] = {3, 4};
    shape __a_shape(a, 2);

    float b[12];
    for (int i = 0; i < 12; ++i)
        b[i] = 1 << i;

    tensor<float> tensor1(__a_shape);
    assign(tensor1, b, 12);

    tensor<float> tensor2;
    tops::ln(tensor1, tensor2);

    std::cout << "log(tensor1, tensor2):\n";
    std::cout << "tensor1:" << tensor1.to_str() << std::endl
              << tensor1.show() << std::endl
              << "tensor2:" << tensor2.to_str() << std::endl
              << tensor2.show() << std::endl;
}

void log_softmax_test(){
    std::cout << "LOG SOFTMAX TEST:\n";
    int a[2] = {3, 4};
    shape __a_shape(a, 2);

    float b[12];
    for (int i = 0; i < 12; ++i)
        b[i] = i;

    tensor<float> tensor1(__a_shape);
    assign(tensor1, b, 12);

    tensor<float> tensor2;
    tensor<float> tensor3;
    tops::log_softmax(tensor1, tensor3, tensor2);

    std::cout << "log_softmax(tensor1, tensor2):\n";
    std::cout << "tensor1:" << tensor1.to_str() << std::endl
              << tensor1.show() << std::endl
              << "tensor2:" << tensor2.to_str() << std::endl
              << tensor2.show() << std::endl;
}

void compute_graph_test() {
//    int a[2] = {64, 8};
//    shape shape1(a, 2);
//    variable batch_x(shape1, "batch_x");
//    uniform(batch_x.get(), -1, 1);
//
//    int b[2] = {5, 8};
//    shape shape2(b, 2);
//    variable weight(shape2, "weight");
//    uniform(weight.get(), -1, 1);
//
//
//    int c[2] = {64, 5};
//    shape shape3(c, 2);
//    variable wx(shape3, "wx");
//
//    int e[2] = {1, 5};
//    shape shape5(e, 2);
//    variable bias(shape5, "bias");
//
//    variable wx_b(shape3, "wx_b");
//
//    variable wx_b_sigmoid(shape3, "sigmoid");
//    dot_multiply(batch_x, weight, wx, true);
//
//    glop::add(wx, bias, wx_b, false);

//    glop::sigmoid(wx_b, wx_b_sigmoid);
//
//    int d[2] = {3, 5};
//    shape shape4(d, 2);
//    variable weight2(shape4, "weight2");
//    uniform(weight2.get(), -1, 1);
//
//
//    int f[2] = {64, 3};
//    shape shape6(f, 2);
//    int g[2] = {1, 3};
//    variable wx2(shape6, "wx2");
//    shape shape7(g, 2);
//    variable bias2(shape7, "bias2");
//
//    variable wx_bias_2(shape6, "wx_bias_2");
//
//    dot_multiply(wx_b_sigmoid, weight2, wx2, true);
//    glop::add(wx2, bias2, wx_bias_2, false);


//    int gg[1] = {1};
//    shape shape8(gg, 1);
//    variable loss(shape8, "loss");
//
//    variable label(shape6, "label");
//
//    variable probability(shape6, "probability");
//    glop::softmax_cross_entropy_with_logits(wx_bias_2, probability, label, loss);
//    vector<variable*>::iterator iter;
//    for (iter = variable::forward_graph.begin(); iter != variable::forward_graph.end(); ++iter){
//        std::cout << (*iter)->to_str() << std::endl;
    //        std::cout << (*iter)->name << " ** consumers:{";
//        for (node* np: (*iter)->consumers){
//            std::cout << np->name << ",";
//        }
//
//        std::cout << "}, inputs:{";
//        for (node* np: (*iter)->inputs){
//            std::cout << np->name << ",";
//        }
//
//        (*iter)->forward();
//
//        std::cout << "}" << std::endl;
//    }

//    for (iter = variable::forward_graph.end()-1; iter != variable::forward_graph.begin() - 1; --iter){
//        std::cout << (*iter)->to_str() << std::endl;
//    }
}

void convert_to_one_hot_test(){
    std::cout << "CONVERT TO ONE HOT TEST:\n";
    real a[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    int b[2] = {5, 2};
    shape shape1(b, 2);

    tensor<real>  y(shape1);
    assign(y, a, 10);

    tensor<real> label;
    convert_to_one_hot(y, label, 11);

    std::cout << "convert_to_one_hot(y, label, 11):\n";
    std::cout << "y:" << y.to_str() << std::endl << y.show() << std::endl
              << "label:" << label.to_str() << std::endl
              << label.show();

    std::cout << label.show();
}

void mul_add_test(){
    clock_t start = clock();
    float a = 555556666.0 * 66666.0;
    clock_t end = clock();
    std::cout << end - start << std::endl;

    clock_t start0 = clock();
    float b = 555556666 + 66666;
    clock_t end0 = clock();
    std::cout << end0-start0 << std::endl;
}

void batch_norm_test(){
    vector<int> s = {3, 4};
    tensor<real> a(s);

    real b[12];
    for (int i = 0; i < 12; ++i)
        b[i] = (real)i;

    assign(a, b, 12);

    tensor<real> c;
    tensor<real> d, e;
    batch_norm(a, d, e, c);

    std::cout << c.show() << std::endl;

    std::cout << d.show() << std::endl;
    std::cout << d.to_str() << std::endl;

    std::cout << e.show() << std::endl;
    std::cout << e.to_str() << std::endl;
}

void config_test(){
    config<string, int> config1;

    config1 = {
            {"is_train", true},
            {"batch_size", 128},
            {"epoch_num", 10}
    };

    std::cout << config1.to_str();


}

void dnn_test(){
    int a[3] = {3, 5, 6};

    DNN dnn(a, 3, 60, 10, 64);

    variable *batch_x = variable::global_variables["batch_x"];

    uniform(batch_x->get(), 1, 2);

    float f[64];
    int p = 0;
    for (int i = 0; i < 64; ++i)
            f[p++] = i % 10;

    vector<int> vector1 = {64};
    tensor<real> t(vector1);
    assign(t, f, 64);
    variable *batch_y = variable::global_variables["batch_y"];


    tops::convert_to_one_hot(t, batch_y->get(), 10);

    dnn.iteration(batch_x->get(), batch_y->get());


//    vector<variable*>::iterator iter;
//    for (iter = variable::forward_graph.begin(); iter != variable::forward_graph.end(); ++iter){
//        std::cout << (*iter)->to_str() << std::endl;
//    }
//
////    DNN dnn2({1, 2, 3, 4});
////
//    for (iter = variable::backward_graph.end()-1; iter != variable::backward_graph.begin() - 1; --iter){
//        std::cout << (*iter)->to_str() << std::endl;
//    }
//
//     std::cout << grad_tables.size();
//    for (auto iter0 = grad_tables.begin(); iter0 != grad_tables.end(); ++iter0){
//        if (iter0->second != 0)
//        std::cout << iter0->first << ' ' << iter0->second << std::endl;
//    }
}

void corpus_test(){
    corpus corpus1("../binary_class.txt", 64);

    corpus1.get_data();
    for (int i = 0; i < corpus1.size; ++i){
        std::cout << corpus1.get_label(i) << std::endl;
    }

    corpus1.reset();
    vector<int> sh = {64, 784};
    tensor<real> batch_x(sh);

    vector<int> s = {64};
    tensor<real> batch_y(s);
    corpus1.x = batch_x.data();
    corpus1.y = batch_y.data();

    bool res = true;

    while(res){
        res = corpus1.generator();

        if (res){
            std::cout << batch_x.show() << std::endl;
            std::cout << batch_y.show() << std::endl;
        }else {
            std::cout << batch_x.show() << std::endl;
            std::cout << batch_y.show() << std::endl;
        }
    }
}

void rnn_test(){
    RNN rnn(64, 28, 28, 10, 64, "sigmoid");


    corpus corpus1("../train.txt", 64);

    vector<int> sh = {64, 784};
    tensor<real> batch_x(sh);

    vector<int> s = {64};
    tensor<real> batch_y(s);
    corpus1.x = batch_x.data();
    corpus1.y = batch_y.data();

    bool res = true;


    for (int i = 0; i < 100; ++i) {
        while(res){
            res = corpus1.generator();

            batch_x.reshape({64*28, 28});
            tensor<LONG> ids({64});
            LONG a[64];
            for (int j = 0; j < 28; ++j){
                for (int i = 0; i < 64; ++i){
                    a[i] = j + 28 * i;
                }
                variable *v = get_variable("X" + std::to_string(j), {64, 28}, false);
                assign(ids, a, 64);
                embeddings_lookup(batch_x, ids, v->get());
            }

            variable *y = get_variable("Y", {64, 10}, false);
            convert_to_one_hot(batch_y, y->get(), 10);
            rnn.iteration();
        }
        corpus1.reset();
        res = true;
    }
}

void lstm_test(){
    LSTM lstm(64, 64, 28, 28, 10, 8, true);

    epoch_num = 100;
    corpus corpus1("../train_small.txt", 64);

    vector<int> sh = {64, 784};
    tensor<real> batch_x(sh);

    vector<int> s = {64};
    tensor<real> batch_y(s);
    corpus1.x = batch_x.data();
    corpus1.y = batch_y.data();

    bool res = true;


    for (int i = 0; i < 100; ++i) {
        while(res){

            for (int k = 0; k < 8; ++k){
                res = corpus1.generator();
                batch_x.reshape({64*28, 28});
                tensor<LONG> ids({64});
                LONG a[64];
                for (int j = 0; j < 28; ++j){
                    for (int i = 0; i < 64; ++i){
                        a[i] = j + 28 * i;
                    }
                    variable *v = get_variable(get_name("X", 0, k), {64, 28}, false);
                    assign(ids, a, 64);
                    embeddings_lookup(batch_x, ids, v->get());
                }

                variable *y = get_variable(get_name("Y", 0, k), {64, 10}, false);
                convert_to_one_hot(batch_y, y->get(), 10);
            }
            lstm.iteration();
        }

        current_epoch++;
        corpus1.reset();
        res = true;
    }
}

void ID3_test(){
    corpus corpus1("../ID3_test.txt", 64, false);

    corpus1.get_data();
    corpus1.test();

    ID3 id3 = ID3();

    id3.generate(corpus1, 1);

    id3.predict(&corpus1);
}

void C45_test(){

    precompute();
    corpus corpus1("../train_small.txt", true);

    corpus1.get_data();
    fill(corpus1.feature_types.begin(), corpus1.feature_types.end(), true);
    C45 c45 = C45();

    c45.generate(corpus1, 8);
    corpus corpus2("../test_small.txt", true);
    fill(corpus2.feature_types.begin(), corpus2.feature_types.end(), true);
    corpus2.get_data();
    c45.predict(&corpus1, true);
    c45.predict(&corpus2, false);
    c45.prune();
    std::cout << "剪枝后:\n";
    c45.predict(&corpus1, true);
    c45.predict(&corpus2, false);

}

void cart_test(){
    corpus corpus1("../train_small.txt", 64, false);

    corpus1.get_data();
//    corpus1.test();
    fill(corpus1.feature_types.begin(), corpus1.feature_types.end(), true);
    cart __cart;

    __cart.generate(corpus1);

    corpus corpus2("../test_small.txt", true);
    fill(corpus2.feature_types.begin(), corpus2.feature_types.end(), true);
    corpus2.get_data();
    __cart.predict(&corpus2, false);

}

void gbdt_test(){
    corpus corpus1("../train_small.txt", 64, true);
    corpus1.get_data();
    fill(corpus1.feature_types.begin(), corpus1.feature_types.end(), true);
    corpus corpus2("../test_small.txt", 64, true);
    fill(corpus2.feature_types.begin(), corpus2.feature_types.end(), true);
    corpus2.get_data();

//    cart __cart;
//    __cart.generate(corpus1);
//    __cart.predict(&corpus1, true);
//    __cart.predict(&corpus2, false);

    GBDT gbdt_;
    gbdt_.train(&corpus1);
    corpus1.reset();
    corpus1.get_data();
    gbdt_.predict(&corpus1);
    gbdt_.predict(&corpus2);
}

void xgboost_test(){
    corpus corpus1("../train_small.txt", 64, true);
    corpus1.zero_one_label = true;
    corpus1.get_data();
    fill(corpus1.feature_types.begin(), corpus1.feature_types.end(), true);
    corpus corpus2("../test_small.txt", 64, true);
    corpus2.zero_one_label = true;
    fill(corpus2.feature_types.begin(), corpus2.feature_types.end(), true);

    corpus2.get_data();
    xgboost::XGBoost xgb;
    xgb.train(&corpus1);
//    corpus1.reset();
//    corpus1.get_data();

    xgb.predict(&corpus2);
}

void shuffle_test(){
    vector<int> v(100, 0);
    std::iota(v.begin(), v.end(), 0);
    shuffle(v);

    for (int num: v)
        std::cout << num << '\t';
    std::cout << std::endl;
}

void label_corpus_test(){
    lc::LabelCorpus labelCorpus("../word_label_corpus.txt");

    labelCorpus.print();

    lc::corn corn_("../label_train.txt");
}

void HMM_test(){
    lc::LabelCorpus labelCorpus("../word_label_corpus.txt");
    HMM hmm;
    hmm.supervise_learn(&labelCorpus);

    vector<string> sample;
    sample.push_back("孟");
    sample.push_back("小平");
    sample.push_back("带领");
    sample.push_back("我们");
    sample.push_back("前进");
    sample.push_back("孟庆阳");
    sample.push_back("。");
    sample.push_back("主航道");
    sample.push_back("水平");
    sample.push_back("提高");
    sample.push_back("考察");
    hmm.predict(sample);

    for (string s: sample)
        std::cout << s << '\t';

    hmm.predict("../pre.txt", "../hmm_predict.txt");
}

void check_format_test(){
    string a = "%x[-1, -2]   ";
    std::cout << a << ":" << lc::check_format(a) << std::endl;
    a = "%x[  -01, , +2.9 ]             ";
    std::cout << a << ":" <<  lc::check_format(a) << std::endl;
}

void split_test(){
    string a = "%x[-1, -2]/// %x[-1, -2]/%x[-1, -2]///";
    vector<string> records = split(a, '/');
    for (string str: records)
        std::cout << str << '\t';
    std::cout << std::endl;
}

void feature_templates_test(){
    lc::FeatureTemplate ft("../templates.txt");
}

void crf_test(){
    crf::CRF crf_;
    crf_.train("../train.txt", "../templates.txt");
}

void svm_test(){
    SVM svm;
    svm.train("../train01.txt");
}

