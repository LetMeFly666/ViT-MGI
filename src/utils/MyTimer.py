'''
Author: LetMeFly
Date: 2024-07-03 14:30:48
LastEditors: LetMeFly
LastEditTime: 2024-07-03 14:35:01
'''
class TimeStruct():
    def __init__(self, eventName: str, eventTime: str):
        self.eventName = eventName
        self.eventTime = eventTime
    
    def __str__(self):
        return f'{self.eventName} - {self.eventTime}'

class TimeRecorder():
    def __init__(self) -> None:
        self.timeList = []

    def print(self):
        toPrint = 'TimeList:'
        for i in range(len(self.timeList)):
            toPrint += f'\n{i:02d}: {self.timeList[i]}'
        print(toPrint)
    
    def addRecord(self, eventName: str, eventTime: str):
        self.timeList.append(TimeStruct(eventName, eventTime))
    