'''
Author: LetMeFly
Date: 2024-07-11 10:23:56
LastEditors: LetMeFly
LastEditTime: 2024-07-11 10:32:05
'''
class A:
    def __init__(self) -> None:
        self.a = 1

a = A() 
a.a = 2
print(a.a)
a.b = 3
print(a.b)
print(a.c)