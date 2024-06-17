'''
Author: LetMeFly
Date: 2024-05-17 16:32:09
LastEditors: LetMeFly
LastEditTime: 2024-06-17 15:48:02
'''
import datetime
now = datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S')
del datetime

from dotenv import load_dotenv
load_dotenv('.env')
from src.utils import initPrint, backupEnv
from src import VisionTransformersRobustness


if __name__ == "__main__":
    initPrint(now)
    print(now)
    backupEnv(now)
    VisionTransformersRobustness.main()