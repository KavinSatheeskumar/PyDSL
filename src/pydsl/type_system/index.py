from pydsl.type_temp import (
    Int,
    Float,
    Sign,
    PyDSLType,
    TargetType,
    lower_single
)
import sys
from math import log2
import typing
import ctypes
from mlir.dialects import arith
from mlir.ir import (
    IndexType,
    IntegerType
)
import mlir.dialects.index as mlir_index
import mlir.ir as mlir
from typing import Type, Self


def get_index_width() -> int:
    s = log2(sys.maxsize + 1) + 1
    assert (
        s.is_integer()
    ), "the compiler cannot determine the index size of the current "
    f"system. sys.maxsize yielded {sys.maxsize}"

    return int(s)


# TODO: for now, you can only do limited math on Index
# division requires knowledge of whether Index is signed or unsigned
# everything will be assumed to be unsigned for now...
class Index(
    Int,
    width=get_index_width(),
    sign=Sign.UNSIGNED,
    ctype=ctypes.c_size_t
):
    @classmethod
    def lower_class(cls) -> tuple[mlir.Type]:
        return (IndexType.get(),)

    AnyInt = typing.TypeVar("I", bound="Int")

    def to(self, target_type: Type[TargetType]) -> TargetType:
        if issubclass(target_type, Index):
            return self._to_Index(target_type)
        elif issubclass(target_type, Int):
            return self._to_Int(target_type)
        elif issubclass(target_type, Float):
            return self._to_Float(target_type)
        else:
            raise TypeError(
                f"Cannot cast Index to non numeric type {
                    target_type.__name__}"
            )

    def _to_Int(self, cls: type[TargetType]) -> TargetType:
        if self.sign != cls.sign:
            raise TypeError(
                "attempt to cast Index to an Int of different sign"
            )

        op = {
            Sign.SIGNED: arith.index_cast,
            Sign.UNSIGNED: arith.index_castui,
        }

        return cls(
            op[cls.sign](IntegerType.get_signless(cls.width), self.value)
        )

    def _to_Index(self, _target_type: Type[TargetType]) -> TargetType:
        return self

    F = typing.TypeVar("F", bound="Float")

    def _to_Float(self, target_type: type[F]) -> F:
        # There does not seem to exist any operation that takes
        # Index -> target Float.

        # We instead do Index -> widest UInt -> target Float.
        return target_type(
            arith.UIToFPOp(
                lower_single(target_type),
                mlir_index.CastUOp(
                    IntegerType.get_signless(64), lower_single(self)
                ),
            )
        )

    def op_add(self, rhs: PyDSLType) -> Self:
        if (rhs_casted := rhs.try_to(type(self))):
            return type(self)(mlir_index.AddOp(self.value, rhs_casted.value))
        else:
            return rhs.op_radd(self)

    def op_radd(self, lhs: PyDSLType) -> Self:
        lhs_casted = lhs.to(type(self))
        return type(self)(mlir_index.AddOp(lhs_casted.value, self.value))

    def op_sub(self, rhs: PyDSLType) -> Self:
        if (rhs_casted := rhs.try_to(type(self))):
            return (type(self))(mlir_index.SubOp(self.value, rhs_casted.value))
        else:
            return rhs.op_rsub(self)

    def op_rsub(self, lhs: PyDSLType) -> Self:
        lhs_casted = lhs.to(type(self))
        return type(self)(mlir_index.SubOp(lhs_casted.value, self.value))

    def op_mul(self, rhs: PyDSLType) -> Self:
        if (rhs_casted := rhs.try_to(type(self))):
            return type(self)(mlir_index.MulOp(self.value, rhs_casted.value))
        else:
            return rhs.op_rmul(self)

    def op_rmul(self, lhs: PyDSLType) -> Self:
        lhs_casted = lhs.to(type(self))
        return type(self)(mlir_index.MulOp(lhs_casted.value, self.value))

    def op_floordiv(self, rhs: PyDSLType) -> Self:
        if (rhs_casted := rhs.try_to(self)):
            return type(self)(
                mlir_index.DivUOp(self.value, rhs_casted.value)
            )
        else:
            return rhs.op_rfloordiv(self)

    def op_rfloordiv(self, lhs: PyDSLType) -> Self:
        lhs_casted = lhs.to(type(self))
        return type(self)(mlir_index.DivUOp(lhs_casted.value, self.value))

    @classmethod
    def CType(cls) -> tuple[type]:
        # TODO: this needs to be different depending on the platform.
        # On Python, you use sys.maxsize. However, we should let the
        # user choose in CompilationSetting

        return (ctypes.c_size_t,)

    @classmethod
    def PolyCType(cls) -> tuple[type]:
        return (ctypes.c_int,)
