//
// Created by mengqy on 2018/12/11.
//

#ifndef FASTAI_LOG_H
#define FASTAI_LOG_H

#include <iostream>
#include <fstream>
using namespace std;

char tmp[1000];
time_t __now = time(nullptr);
time_t __last = time(nullptr);

class Logger{
private:
    ofstream out;
    bool console = true;

    int last_second = 0;

public:
    Logger(const char *name){
        out.open(name);
        if (!out.is_open()){
            std::cerr << "Can't open file named " << name << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    Logger(){}

    ~Logger(){
        if (out.is_open()) out.close();
    }

    void info(const char *text){
        get_time();
        if (console) std::cout << tmp << "- Beyond - INFO - " << text << std::endl;
        if (out.is_open()) out << tmp << "- Beyond - INFO - " << text << std::endl;
    }


    void info(string text, bool flush=false){
        get_time();
        if (!flush){
            if (console) std::cout << tmp << "- Beyond - INFO - " << text << std::endl;
            if (out.is_open()) out << tmp << "- Beyond - INFO - " << text << std::endl;
        }else{
            std::cout << '\r' << tmp << "- Beyond - INFO - " << text << std::flush;
        }
    }


    void debug(const char *text){
        get_time();
        if (console) std::cout << tmp << "Beyond - DEBUG - " << text << std::endl;
        if (out.is_open()) out << tmp << "Beyond - DEBUG - " << text << std::endl;
    }

    void debug(string& text){
        get_time();
        if (console) std::cout << tmp << "Beyond - DEBUG - " << text << std::endl;
        if (out.is_open()) out << tmp << "Beyond - DEBUG - " << text << std::endl;
    }

    void error(const char *text){
        get_time();
        if (console) std::cout << tmp << "Beyond - ERROR - " << text << std::endl;
        if (out.is_open()) out << tmp << "Beyond - ERROR - " << text << std::endl;
    }

    void error(string& text){
        get_time();
        if (console) std::cout << tmp << "Beyond - ERROR - " << text << std::endl;
        if (out.is_open()) out << tmp << "Beyond - ERROR - " << text << std::endl;
    }

    void get_time(){
        __now = time(nullptr);
        tm* t= localtime(&__now);
        sprintf(tmp, "%d-%02d-%02d %02d:%02d:%02d ",
                t->tm_year + 1900,
                t->tm_mon + 1,
                t->tm_mday,
                t->tm_hour,
                t->tm_min,
                t->tm_sec);
    }

    float get_diff_time(){
        __now = time(nullptr);
        float res = difftime(__now, __last);
        __last = __now;
        return res;
    }
};

Logger logger;

#endif //FASTAI_LOG_H
