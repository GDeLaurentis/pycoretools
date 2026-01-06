class Singleton:

    _instances = {}

    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__new__(cls)
        return cls._instances[cls]


class NotAnInteger(Singleton):

    def __init__(self):
        pass

    def __repr__(self):
        return "NaI"

    def __str__(self):
        return "NaI"

    def __add__(self, _):
        return self

    def __sub__(self, _):
        return self

    def __mul__(self, _):
        return self

    def __truediv__(self, _):
        return self

    def __floordiv__(self, _):
        return self

    def __mod__(self, _):
        return self

    def __pow__(self, _):
        return self

    def __radd__(self, _):
        return self

    def __rsub__(self, _):
        return self

    def __rmul__(self, _):
        return self

    def __rtruediv__(self, _):
        return self

    def __rfloordiv__(self, _):
        return self

    def __rmod__(self, _):
        return self

    def __rpow__(self, _):
        return self


NaI = NotAnInteger()
