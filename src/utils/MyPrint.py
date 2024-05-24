from typing import Any
from os import makedirs
import builtins
import inspect
import tempfile


builtins.stdout = print


class MyPrint:
    def __init__(self) -> None:
        self.fileout = None
    
    def myInit(self, dirname) -> None:
        dirname = f'result/{dirname}'
        makedirs(dirname, exist_ok=True)
        self.fileout = open(f'{dirname}/stdout.txt', 'w')
        
    def __call__(self, *args: Any, **kwds: Any) -> None:
        # get filename and lineNum
        stack = inspect.stack()
        frame = stack[2]
        filename = frame.filename
        lineno = frame.lineno
        # write into tempfile
        with tempfile.TemporaryFile(mode='w+', encoding='utf-8') as f:
            originalFile = kwds.get('file', None)
            kwds['file'] = f
            builtins.stdout(*args, **kwds)
            f.seek(0)
            content = f.read()
        # get data shape by tempfile
        lines = content.splitlines()
        if len(lines) > 1 and lines[-1] == '\n':
            lines.pop()
        length = max(len(line) + 4 for line in lines)
        title = f'{filename}:{lineno}'
        length = max(length, len(title) + 6)  # 第一行最短为 +- /path/to/file:2 -+
        if (length - len(title)) % 2:  # 第一行左右要对称
            length += 1
        hyphen = '-' * ((length - len(title) - 4) // 2)
        toShow = f'+{hyphen} {title} {hyphen}+\n'
        for line in lines:
            toShow += f'| {line}' + (' ' * (length - 3 - len(line))) + '|\n'
        toShow += '+' + ('-' * (length - 2)) + '+\n'
        # original print
        if originalFile:
            kwds['file'] = originalFile
        else:
            del kwds['file']
        builtins.stdout(toShow, **kwds)
        # write into file
        kwds['file'] = self.fileout
        builtins.stdout(toShow, **kwds)
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
