#!/usr/bin/env python
# coding: utf-8
import sys
import pickle
def set_name():
    with open('this_test_model_name','wb') as outfile:
        pickle.dump(name,outfile)
    with open(assign_name,'wb') as outfile:
        pickle.dump(name,outfile)
def main():
    global name
    name="test_model_2020-04-12_0738"
    global assign_name
    assign_name="name_test_estim_"
    set_name()
if __name__=='__main__':
    main()


