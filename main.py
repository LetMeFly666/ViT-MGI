'''
Author: LetMeFly
Date: 2024-07-03 10:37:25
LastEditors: LetMeFly
LastEditTime: 2024-07-03 14:35:15
'''
import datetime
getNow = lambda: datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S')
now = getNow()
# del datetime
from src.utils import initPrint, TimeRecorder
from typing import List, Optional

initPrint(now)
print(now)



    



import time

timeRecoder = TimeRecorder()
timeRecoder.addRecord('Start', getNow())
time.sleep(3)
timeRecoder.addRecord('End', getNow())
timeRecoder.print()
