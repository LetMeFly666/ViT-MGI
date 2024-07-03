'''
Author: LetMeFly
Date: 2024-07-03 14:30:48
LastEditors: LetMeFly
LastEditTime: 2024-07-03 16:57:45
'''
import datetime
from typing import List

getNow = lambda: datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S')


class TimeStruct():
    def __init__(self, eventName: str, eventTime: str):
        self.eventName = eventName
        self.eventTime = eventTime
    
    def __str__(self):
        return f'{self.eventName} | {self.eventTime}'

class TimeRecorder():
    def __init__(self) -> None:
        self.timeList: List[TimeStruct] = []
        self.hadPrint = False

    def printAll(self):
        toPrint = 'TimeList:'
        for i in range(len(self.timeList)):
            toPrint += f'\n{i:02d}: {self.timeList[i]}'
        print(toPrint)
        self.hadPrint = True
    
    def addRecord(self, eventName: str, eventTime: str=None, ifPrint=True):
        if not eventTime:
            eventTime = getNow()
        thisRecord = TimeStruct(eventName, eventTime)
        self.timeList.append(thisRecord)
        if ifPrint:
            print(thisRecord)
    
    def __del__(self):
        if not self.hadPrint:
            self.printAll()
    