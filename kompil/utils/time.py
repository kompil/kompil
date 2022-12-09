import time

from datetime import datetime

__START_TIME = None

__YMD_STR_FORMAT = "%Y%m%d"
__HMS_STR_FORMAT = "%H%M%S"


def to_str(date: datetime, reduction: str = "none"):
    if date:
        if reduction == "none":
            frmt = f"{__YMD_STR_FORMAT}-{__HMS_STR_FORMAT}"
        elif reduction == "ymd":
            frmt = __YMD_STR_FORMAT
        elif reduction == "hms":
            frmt = __HMS_STR_FORMAT

        return date.strftime(frmt)

    return None


def from_str(val) -> datetime:
    return datetime.strptime(val, f"{__YMD_STR_FORMAT}-{__HMS_STR_FORMAT}")


def now():
    return datetime.now()


def now_str():
    return to_str(now())


def start_time():
    global __START_TIME

    if __START_TIME is None:
        raise Exception("Start time is not defined !")

    return __START_TIME


def start_time_str():
    return to_str(start_time())


def setup_start_time(force: bool = False):
    global __START_TIME

    if __START_TIME is None or force:
        __START_TIME = datetime.now()


class MeanTimer:
    def __init__(self):
        self.__count = 0
        self.__start_time = 0.0
        self.__total_time = 0.0

    def __enter__(self):
        self.__start_time = time.time()

    def __exit__(self, type, value, traceback):
        end_time = time.time()
        spent = float(end_time - self.__start_time)
        self.__total_time += spent
        self.__count += 1

    @property
    def mean_time(self) -> float:
        return self.__total_time / self.__count

    @property
    def total_time(self) -> float:
        return self.__total_time

    @property
    def count(self) -> int:
        return self.__count
