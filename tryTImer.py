import datetime
getNow = lambda: datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S')
now = getNow()
# del datetime
from src.utils import initPrint, TimeRecorder
from typing import List, Optional, Tuple

initPrint(now)
print(now)

timeRecorder = TimeRecorder()
timeRecorder.addRecord('Start', getNow())

import time
time.sleep(1)
timeRecorder.addRecord('End', getNow())

timeRecorder.printAll()