from inspect import currentframe


def get_last_frame(depth=0):
    frame = currentframe()
    assert frame is not None
    for _ in range(depth + 2):
        frame = frame.f_back
        assert frame is not None
    return frame


def get_caller_name():
    frame = get_last_frame(1)
    return frame.f_globals["__name__"] + "." + frame.f_code.co_name
