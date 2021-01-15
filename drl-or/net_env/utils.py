#coding=utf-8
import random

'''
@param: 
    weight: an int list
'''
def weight_choice(weight):
    t = random.randint(0, sum(weight) - 1)
    for i, val in enumerate(weight):
        t -= val
        if t < 0:
            return i

