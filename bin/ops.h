//
// Created by mengqy on 2019/1/3.
//

#ifndef FASTAI_OPS_H
#define FASTAI_OPS_H

#include "tensor_ops.h"
#include "global.h"

using namespace tops;

namespace ops{
    class op {
    public:
        string name = "op";
    public:
        virtual ~op() {};
        virtual void forward() = 0;
        virtual void backward() = 0;

    };

    template <typename T>
    class DotMultiply:public op{
    private:
        tensor<T> *a;
        tensor<T> *b;
        tensor<T> *des;
        bool transposed=false;

    public:
        DotMultiply(tensor<T> *a, tensor<T> *b, tensor<T> *des, bool transposed)
        :a(a), b(b), des(des), transposed(transposed){
            op::name = "dot_mul";
        }

        void forward(){
            if (DEBUG) std::cout << a->name << op::name << b->name << " = "<< des->name << std::endl;
            tops::dot_mul(*a, *b, *des, transposed);
        }

        void backward(){
            if (transposed){
                tensor<real> *__p_des_grad = 0;
                if (grad_tables.count(partial_derivative_name(des->name)) == 1)
                    __p_des_grad = grad_tables[partial_derivative_name(des->name)];


                if (grad_tables.count(partial_derivative_name(b->name)) == 1){
                    tensor<real> _b;
                    tops::dot_mul(*a, *__p_des_grad, _b, true, false);
                    _b = transpose(_b);

                    tensor<real> *__p_b_grad = grad_tables[partial_derivative_name(b->name)];
                    __p_b_grad->reshape(b->get_shape());
                    add(*__p_b_grad, _b, *__p_b_grad);
                }


                if (grad_tables.count(partial_derivative_name(a->name)) == 1){
                    tensor<real> *__p_a_grad = grad_tables[partial_derivative_name(a->name)];
                    tensor<real> _a;
                    tops::dot_mul(*__p_des_grad, *b, _a, false, false);
                    add(*__p_a_grad, _a, *__p_a_grad);
                }
            } else {

            }
        }
    };

    template <typename T>
    class multiply:public op{
        tensor<T> *a;
        tensor<T> *b;
        tensor<T> *des;
    public:
        multiply(tensor<T> *a, tensor<T> *b, tensor<T> *des)
        :a(a), b(b), des(des){
            op::name = "multiply";
        }

        void forward(){
            tops::multiply(*a, *b, *des);
        }

        void backward(){
            tensor<real> *__p_des_grad = 0;
            if (grad_tables.count(partial_derivative_name(des->name)) == 1)
                __p_des_grad = grad_tables[partial_derivative_name(des->name)];

            if (grad_tables.count(partial_derivative_name(a->name)) == 1){
                tensor<T> __a_grad(a->get_shape());
                tops::multiply(*__p_des_grad, *a, __a_grad);
                update_grad_table(partial_derivative_name(a->name), __a_grad);
            }

            if (grad_tables.count(partial_derivative_name(b->name)) == 1){
                tensor<T> __b_grad(b->get_shape());
                tops::multiply(*__p_des_grad, *b, __b_grad);
                update_grad_table(partial_derivative_name(b->name), __b_grad);
            }
        }
    };

    template <typename T>
    class add:public op{
    private:
        tensor<T> *a;
        tensor<T> *b;
        tensor<T> *des;
    public:
        add(tensor<T> *a, tensor<T> *b, tensor<T> *des)
        :a(a), b(b), des(des){
            op::name = "add";
        }

        void forward(){
            tops::add(*a, *b, *des);
        }

        void backward(){
            tensor<real> *__p_des_grad = 0;
            if (grad_tables.count(partial_derivative_name(des->name)) == 1)
                __p_des_grad = grad_tables[partial_derivative_name(des->name)];

            if (grad_tables.count(partial_derivative_name(a->name)) == 1){
                update_grad_table(partial_derivative_name(a->name), *__p_des_grad);
            }

            if (grad_tables.count(partial_derivative_name(b->name)) == 1){
                update_grad_table(partial_derivative_name(b->name), *__p_des_grad);
            }
        }
    };

    template <typename T>
    class AddSuffix:public op{
    private:
        tensor<T> *a;
        tensor<T> *b;
        tensor<T> *des;
    public:
        AddSuffix(tensor<T> *a, tensor<T> *b, tensor<T> *des)
        :a(a), b(b), des(des){
            op::name = "add_suffix";
        }

        void forward(){
            if (DEBUG) std::cout << a->name << op::name << b->name << " = " << des->name << std::endl;
            tops::add_suffix(*a, *b, *des);
        }

        void backward(){
            tensor<real> *__p_b_grad = grad_tables[partial_derivative_name(b->name)];
            __p_b_grad->reshape(b->get_shape());

            tensor<real> *__p_des_grad = grad_tables[partial_derivative_name(des->name)];

            tensor<real> _b;
            reduce_sum(*__p_des_grad, _b, 0);

            tops::add(*__p_b_grad, _b, *__p_b_grad);

            tensor<real> *__p_a_grad = grad_tables[partial_derivative_name(a->name)];
            __p_a_grad->reshape(a->get_shape());

            tops::add(*__p_a_grad, *__p_des_grad, *__p_a_grad);
        }
    };

    template <typename T>
    class subtract:public op{

        void forward(){

        }

        void backward(){

        }
    };

    template <typename T>
    class sigmoid:public op{
    public:
        tensor<T> *a;
        tensor<real> *des;
    public:
        sigmoid(tensor<T> *a, tensor<real> *des):a(a), des(des){
            op::name = "sigmoid";
        }

        void forward(){
            if (DEBUG) std::cout << a->name << op::name << des->name << std::endl;
            tops::sigmoid(*a, *des);
        }

        void backward(){
            if (grad_tables.count(partial_derivative_name(des->name)) == 1){
                tensor<real> *__partial_sigmoid = grad_tables[partial_derivative_name(des->name)];
                tensor<real> __derivative_sigmoid(des->get_shape());

                real *__pds = __derivative_sigmoid.data();
                real *__pd = des->data();

                for (LONG i = 0; i < des->size(); ++i){
                    *__pds++ = (*__pd) * (1 - *__pd);
                    ++__pd;
                }

                tops::multiply(__derivative_sigmoid, *__partial_sigmoid, __derivative_sigmoid);

                update_grad_table(partial_derivative_name(a->name), __derivative_sigmoid);
            }
        }
    };

    template <typename T>
    class tanh: public op{
    public:
        tensor<T> *a;
        tensor<real> *des;
    public:
        tanh(tensor<T> *a, tensor<real> *des):a(a), des(des){
            op::name = "tanh";
        }

        void forward(){
            if (DEBUG) std::cout << a->name << op::name << des->name << std::endl;
            tops::tanh(*a, *des);
        }

        void backward(){
            if (grad_tables.count(partial_derivative_name(des->name)) == 1){
                tensor<real> __derivative_tanh(des->get_shape());

                real *__pdt = __derivative_tanh.data();
                real *__pd = des->data();

                for (LONG i = 0; i < des->size(); ++i){
                    *__pdt++ = 1 - (*__pd) * (*__pd);
                    ++__pd;
                }
//
//                tops::pow0(__derivative_tanh, 2, __derivative_tanh);
//                tops::subtract(1.0f, __derivative_tanh, __derivative_tanh);

                tensor<real> *__partial_des = grad_tables[partial_derivative_name(des->name)];
                tops::multiply(*__partial_des, __derivative_tanh, __derivative_tanh);

                update_grad_table(partial_derivative_name(a->name), __derivative_tanh);
            }
        }
    };

//    template <typename T>
//    class softmax:public op{
//    public:
//        tensor<T> *a;
//        tensor<real> *res;
//    public:
//        softmax(tensor<T> *a, tensor<real> *res):a(a), res(res){}
//
//        void forward(){
//            tops::softmax(*a, *res);
//        }
//
//        void backward(){
//
//        }
//    };

    template <typename T>
    class log:public op{
    public:
        tensor<T> *a;
        tensor<real> *des;
    public:
        log(tensor<T> *a, tensor<real> *res):a(a), des(des){}

        void forward(){
            tops::ln(*a, *des);
        }

        void backward(){

        }

    };

    class SoftmaxCrossEntropy: public op{
    public:
        tensor<real> *predict;
        tensor<real> *label;
        tensor<real> *loss;
    public:
        SoftmaxCrossEntropy(tensor<real> *p, tensor<real> *label, tensor<real> *loss=0)
                :predict(p), label(label), loss(loss){
            op::name = "softmax_cross_entropy";
        }

        void forward(){
            if (DEBUG) std::cout << predict->name << op::name << '\t' << label->name << std::endl;
            real total_loss = softmax_cross_entropy_with_logits(*predict, *label);
            if (loss == 0) update_total_loss(total_loss);
            else{
                real *__pl = loss->data();
                *__pl = total_loss;
            }
        }

        void backward(){
            tensor<real> __grad;
            softmax(*predict, __grad);
            tops::subtract(*label, __grad, __grad);
            string name = partial_derivative_name(predict->name);
            update_grad_table(name, __grad);
        }
    };

    class BatchNorm: public op{
    public:
        tensor<real> *a;
        tensor<real> *m;
        tensor<real> *v;
        tensor<real> *des;

    public:
        BatchNorm(tensor<real> *a, tensor<real> *m, tensor<real> *v, tensor<real> *des)
        :a(a), m(m), v(v), des(des){
            op::name = "batch_norm";
        }

        void forward(){
            tops::batch_norm(*a, *m, *v, *des);
        }

        void backward(){
            if (grad_tables.count(partial_derivative_name(des->name)) == 1){

                int batch_size = a->get_shape()[0];

                tensor<real> *__partial_norm_x = grad_tables[partial_derivative_name(des->name)];

                tensor<real> __partial_u;
                tops::reduce_sum(*__partial_norm_x, __partial_u, 0);
                tops::multiply(*v, __partial_u, __partial_u);
                tops::multiply(-1.0f / batch_size, __partial_u, __partial_u);

                tensor<real> __partial_delta2;
                tensor<real> w;
                tops::multiply(-1.0f, *v, w);

                tensor<real> tmp;
                tops::multiply(*des, *__partial_norm_x, tmp);
                tops::reduce_sum(tmp, __partial_delta2, 0);
                tops::multiply(__partial_delta2, w, __partial_delta2);

                tops::multiply_suffix(*__partial_norm_x, __partial_delta2, tmp);

                tops::multiply(tmp, 1.0f / batch_size, tmp);

                if (grad_tables.count(partial_derivative_name(a->name)) == 1){
                    tensor<real> *__partial_x = grad_tables[partial_derivative_name(a->name)];
                    tops::multiply_suffix(*__partial_x, *v, *__partial_x);
                    tops::add(*__partial_x, tmp, *__partial_x);
                    tops::add_suffix(*__partial_x, __partial_u, *__partial_x);
                }
            }
        }
    };

    class Redirect: public op{
    public:
        tensor<real> *a;
        tensor<real> *scale;
        tensor<real> *bias;
        tensor<real> *des;

    public:
        Redirect(tensor<real> *a, tensor<real> *scale, tensor<real> *bias, tensor<real> *des)
        :a(a), scale(scale), bias(bias), des(des){
            op::name = "redirect";
        }

    public:

        void forward(){
            tops::redirect(*a, *scale, *bias, *des);
        }

        void backward(){
            if (grad_tables.count(partial_derivative_name(des->name)) == 1){
                tensor<real> *__partial_y = grad_tables[partial_derivative_name(des->name)];

                tensor<real> tmp;
                tops::multiply(*a, *__partial_y, tmp);
                tensor<real> __partial_scale;
                tops::reduce_sum(tmp, __partial_scale, 0);
                update_grad_table(partial_derivative_name(scale->name), __partial_scale);

                tops::multiply_suffix(*__partial_y, *scale, tmp);
                update_grad_table(partial_derivative_name(a->name), tmp);

                tops::reduce_sum(*__partial_y, __partial_scale, 0);
                update_grad_table(partial_derivative_name(bias->name), __partial_scale);
            }
        }
    };
}










#endif //FASTAI_OPS_H
