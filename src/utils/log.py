from atexit import register
from builtins import print
from functools import cache
from logging import Formatter, LogRecord, StreamHandler, getLogger
from time import process_time_ns

from utils.frame import get_caller_name

get_logger = cache(getLogger)


class RichConsoleHandler(StreamHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.first_time = True
        self.setFormatter(IntervalFormatter("%(interval)s  %(name)s  %(message)s"))

    def emit(self, record: LogRecord):
        if self.first_time:
            self.first_time = False
            print(file=self.stream)
            register(lambda: print(file=self.stream))

        try:
            msg = self.format(record)
            if "\n" in msg:
                msg = msg.replace(
                    "\n",
                    "\n" + " " * len(f"{record.interval}  {record.name}  "),  # type: ignore
                )
            print(msg, file=self.stream)

        except Exception:
            self.handleError(record)


class IntervalFormatter(Formatter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.last = 0

    def format(self, record: LogRecord):
        current = process_time_ns()
        interval = current - self.last
        self.last = current

        record.interval = f"{interval // 1_000_000:>5d} ms"
        record.name = f"{record.name:<20}"

        return super().format(record)


logger = getLogger()
logger.setLevel("DEBUG")
logger.addHandler(RichConsoleHandler())


def log(*message: str):
    logger = get_logger(get_caller_name())
    logger.info(message[0] if len(message) == 1 else message)


def patch_print():
    import builtins

    builtins.print = log
