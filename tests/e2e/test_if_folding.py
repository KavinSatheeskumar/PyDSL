from pydsl.frontend import compile
from pydsl.type import Bool


def test_fold():
    @compile(dump_mlir=True)
    def func1() -> Bool:
        a: Bool = False
        if True:
            a = True
        else:
            a = False
        return a

    @compile(dump_mlir=True)
    def func2() -> Bool:
        a: Bool = False
        if True:
            a = False
        else:
            a = True
        return a

    @compile(dump_mlir=True)
    def func3() -> Bool:
        return 3 if False else False

    @compile(dump_mlir=True)
    def func4() -> Bool:
        return True if True else False


if __name__ == "__main__":
    test_fold()
