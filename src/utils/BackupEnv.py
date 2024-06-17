'''
Author: LetMeFly
Date: 2024-05-24 15:34:22
LastEditors: LetMeFly
LastEditTime: 2024-05-24 15:46:09
'''
from shutil import copyfile
from os.path import join


"""直接将本次运行时的.env复制到本次的输出文件夹中"""
def backupEnv(dirname):
    copyfile('.env', join('result', dirname, '.env'))
