import datetime
now = datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S')
del datetime

from src.utils import initPrint
from src import VisionTransformersRobustness


if __name__ == "__main__":
    initPrint(now)
    print(now)
    VisionTransformersRobustness.main()