'''
Author: LetMeFly
Date: 2024-05-17 16:32:09
LastEditors: LetMeFly
LastEditTime: 2024-05-24 15:45:15
'''
import datetime
now = datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S')
del datetime

from src.utils import initPrint, backupEnv
from src import VisionTransformersRobustness


if __name__ == "__main__":
    initPrint(now)
    print(now)
    backupEnv(now)
    VisionTransformersRobustness.main()