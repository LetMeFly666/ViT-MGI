'''
Author: LetMeFly
Date: 2024-05-17 16:32:09
LastEditors: LetMeFly
LastEditTime: 2024-07-02 17:27:48
'''
import datetime
now = datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S')
del datetime

from src.utils import initPrint
from src import server


if __name__ == "__main__":
    initPrint(now)
    print(now)
    server.main()