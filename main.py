'''
Author: LetMeFly
Date: 2024-07-03 10:37:25
LastEditors: LetMeFly
LastEditTime: 2024-07-03 14:29:00
'''
import datetime
getNow = lambda: datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S')
now = getNow()
# del datetime
from src.utils import initPrint
from typing import List, Optional

initPrint(now)
print(now)


class TimeRecorder():
    def __init__(self, eventName: str, eventTime: str):
        self.eventName = eventName
        self.eventTime = eventTime
    
    def __str__(self):
        return f'{self.eventName} - {self.eventTime}'
    

timeList = []

def printTimeList():
    toPrint = 'TimeList:'
    for i in range(len(timeList)):
        toPrint += f'\n{i:02d}: {timeList[i]}'
    print(toPrint)

import time

timeList.append(TimeRecorder('Start', getNow()))
time.sleep(5)
timeList.append(TimeRecorder('End', getNow()))

printTimeList()