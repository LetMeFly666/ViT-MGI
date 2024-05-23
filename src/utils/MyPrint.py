from typing import Any
from os import makedirs
import builtins
import inspect


builtins.stdout = print


class MyPrint:
    def __init__(self) -> None:
        self.fileout = None
    
    def myInit(self, dirname) -> None:
        dirname = f'result/{dirname}'
        makedirs(dirname, exist_ok=True)
        self.fileout = open(f'{dirname}/stdout.txt', 'w')
        
    def __call__(self, *args: Any, **kwds: Any) -> None:
        builtins.stdout(*args, **kwds)
        kwds['file'] = self.fileout
        stack = inspect.stack()
        frame = stack[2]
        filename = frame.filename
        lineno = frame.lineno
        builtins.stdout(f'---------------------- {filename}:{lineno} [Begin] --------------------', file=self.fileout)
        builtins.stdout(*args, **kwds)
        builtins.stdout(f'---------------------- {filename}:{lineno} [End] ----------------------', file=self.fileout)
        self.fileout.flush()

    def __del__(self) -> None:
        if self.fileout:
            self.fileout.close()


printer = MyPrint()


def print(*args: Any, **kwds: Any) -> None:
    printer(*args, **kwds)


def initPrint(dirname: str) -> None:
    printer.myInit(dirname)
    builtins.print = print
