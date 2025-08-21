from typing import (
    Type,
    TypeVar,
    Self,
    Protocol,
    runtime_checkable
)

from mlir.dialects import arith
from pydsl.type_temp import Bool

from enum import Enum, auto

from mlir.dialects import arith, func, scf
import mlir.ir as mlir
from mlir.ir import (
    Context,
    InsertionPoint,
    Location,
    UnitAttr,
    Value,
    OpView
)


class CompType(Enum):
    eq = auto()
    ne = auto()
    lt = auto()
    gt = auto()
    le = auto()
    ge = auto()


class PyDSLType():
    """
    A common super class so we can
    do isinstance(my_obj, PyDSLType)

    This also gives us consisten default operation
    behaviour for the following operations.
    """
    # Casting
    # TODO: Maybe try_to should be the one people
    # define and to just raises an exception when try_to
    # return None. Might improve performance a bit by
    # reducing the number of exceptions thrown

    def to(self, target_type: Type[Self]) -> Self:
        raise TypeError(
            f"casting from {type(self).__name__} to {
                target_type.__name__} not implemented"
        )

    def try_to(self, target_type: Type[Self]) -> Self | None:
        try:
            return self.to(target_type)
        except TypeError:
            return None

    # Unary Ops

    def op_neg(self) -> Self:
        raise ValueError("op_neg not implemented on this type")

    def op_pos(self) -> Self:
        raise ValueError("op_pos not implemented on this type")

    def op_invert(self) -> Self:
        raise ValueError("op_invert not implemented on this type")

    def op_not(self) -> Self:
        raise ValueError("op_not not implemented on this type")

    def op_abs(self) -> Self:
        raise ValueError("op_abs not implemented on this type")

    # Boolean Ops

    def op_and(self, rhs: Self) -> Bool:
        return rhs.op_rand(self)

    def op_rand(self, lhs: Self) -> Bool:
        raise ValueError("op_rand not implemented on this type")

    def op_or(self, rhs: Self) -> Bool:
        return rhs.op_ror(self)

    def op_ror(self, lhs: Self) -> Bool:
        raise ValueError("op_ror not implemented on this type")

    # Comparision Ops

    def op_spaceship(self, rhs: Self, pred: CompType) -> Bool:
        return rhs.op_rspaceship(self, pred)

    def op_rspaceship(self, lhs: Self, pred: CompType) -> Bool:
        raise ValueError("op_rspaceship not implemented on this type")

    # No right hand equivalent for these since we have
    # op spaceship and op rspaceship
    def op_eq(self, rhs: Self) -> Bool:
        return self.op_spaceship(rhs, CompType.eq)

    def op_ne(self, rhs: Self) -> Bool:
        return self.op_spaceship(rhs, CompType.ne)

    def op_le(self, rhs: Self) -> Bool:
        return self.op_spaceship(rhs, CompType.le)

    def op_lt(self, rhs: Self) -> Bool:
        return self.op_spaceship(rhs, CompType.lt)

    def op_ge(self, rhs: Self) -> Bool:
        return self.op_spaceship(rhs, CompType.ge)

    def op_gt(self, rhs: Self) -> Bool:
        return self.op_spaceship(rhs, CompType.gt)

    def op_is(self, rhs: Self) -> Bool:
        return rhs.op_ris(self)

    def op_ris(self, lhs: Self) -> Bool:
        raise ValueError("op_ris is not implemented on this type")

    def op_isnot(self, rhs: Self) -> Bool:
        return rhs.op_risnot(self)

    def op_risnot(self, lhs: Self) -> Bool:
        raise ValueError("op_risnot is not implemented on this type")

    # Binary operation

    def op_add(self, rhs: Self) -> Self:
        return rhs.op_radd(self)

    def op_radd(self, lhs: Self) -> Self:
        raise ValueError("op_radd not implemented on this type")

    def op_sub(self, rhs: Self) -> Self:
        return rhs.op_sub(self)

    def op_rsub(self, lhs: Self) -> Self:
        raise ValueError("op_rsub not implemented on this type")

    def op_mul(self, rhs: Self) -> Self:
        return rhs.op_rmul(self)

    def op_rmul(self, lhs: Self) -> Self:
        raise ValueError("op_rmul not implemented on this type")

    def op_floordiv(self, rhs: Self) -> Self:
        return rhs.op_rfloordiv(self)

    def op_rfloordiv(self, lhs: Self) -> Self:
        raise ValueError("op_rfloordiv not implemented on this type")

    def op_truediv(self, rhs: Self) -> Self:
        return rhs.op_rtruediv(self)

    def op_rtruediv(self, lhs: Self) -> Self:
        raise ValueError("op_rtruediv not implemented on this type")

    def op_mod(self, rhs: Self) -> Self:
        return rhs.op_rmod(self)

    def op_rmod(self, lhs: Self) -> Self:
        raise ValueError("op_rmod not implemented on this type")

    def op_pow(self, rhs: Self) -> Self:
        return rhs.op_rpow(self)

    def op_rpow(self, lhs: Self) -> Self:
        raise ValueError("op_rpow not implemented on this type")

    def op_lshift(self, rhs: Self) -> Self:
        return rhs.op_rlshift(self)

    def op_rlshift(self, lhs: Self) -> Self:
        raise ValueError("op_rlshift not implemented on this type")

    def op_rshift(self, rhs: Self) -> Self:
        return rhs.op_rrshift(self)

    def op_rrshift(self, lhs: Self) -> Self:
        raise ValueError("op_rrshift not implemented on this type")

    def op_bitand(self, rhs: Self) -> Self:
        return rhs.op_rbitand(self)

    def op_rbitand(self, lhs: Self) -> Self:
        raise ValueError("op_rbitand not implemented on this type")

    def op_bitor(self, rhs: Self) -> Self:
        return rhs.op_rbitor(self)

    def op_rbitor(self, lhs: Self) -> Self:
        raise ValueError("op_rbitor not implemented on this type")

    def op_bitxor(self, rhs: Self) -> Self:
        return rhs.op_rbitxor(self)

    def op_rbitxor(self, lhs: Self) -> Self:
        raise ValueError("op_rbitxor not implemented on this type")


class PyDSLTypeMetaclass(type):
    def __new__(cls, name, bases, attrs):
        return super().__new__(
            name, bases + [cls.PyDSLType], attrs
        )

    def on_Call(self, val: PyDSLType):
        return val.to(self)

    def op_is(self, val: Type[PyDSLType]):
        if (self == val):
            return Bool(arith.ConstantOp(Bool.lower_class(), 1))
        else:
            return Bool(arith.ConstantOp(Bool.lower_class(), 0))

    op_ris = op_is

    def op_isnot(self, val: Type[PyDSLType]):
        if (self != val):
            return Bool(arith.ConstantOp(Bool.lower_class(), 1))
        else:
            return Bool(arith.ConstantOp(Bool.lower_class(), 0))

    op_risnot = op_isnot


# A useful alias
TargetType = TypeVar("TargetType", bound="PyDSLType")


@runtime_checkable
class Lowerable(Protocol):
    def lower(self) -> tuple[Value]: ...

    @classmethod
    def lower_class(cls) -> tuple[mlir.Type]: ...


def lower(
    v: Lowerable | type | OpView | Value | mlir.Type,
) -> tuple[Value] | tuple[mlir.Type]:
    """
    Convert a `Lowerable` type, type instance, and other MLIR objects into its
    lowest MLIR representation, as a tuple.

    This function is *not* idempotent.

    Specific behavior:
    - If `v` is a `Lowerable` type, a `mlir.ir.Type` is returned.
    - If `v` is a `Lowerable` type instance, a `mlir.ir.Value` is returned.
    - If `v` is an `mlir.ir.OpView` type instance, then its results (of type
      `mlir.ir.Value`) are returned.
    - If `v` is already an `mlir.ir.Value` or `mlir.ir.Type`, `v` is returned
      enclosed in a tuple.
    - If `v` is not any of the types above, `TypeError` will be raised.

    For example:
    - `lower(Index)` should be equivalent to `(IndexType.get(),)`.
    - `lower(Index(5))` should be equivalent to
      `(ConstantOp(IndexType.get(), 5).results,)`.
    - ```lower(UInt8(4).op_add(UInt8(5)))``` should be equivalent to ::

        tuple(AddIOp(
            ConstantOp(IntegerType.get_signless(8), 4), )
            ConstantOp(IntegerType.get_signless(8), 5)).results)
    """
    match v:
        case OpView():
            return tuple(v.results)
        case Value() | mlir.Type():
            return (v,)
        case type() if issubclass(v, Lowerable):
            # Lowerable class
            return v.lower_class()
        case _ if issubclass(type(v), Lowerable):
            # Lowerable class instance
            return v.lower()
        case _:
            raise TypeError(f"{v} is not Lowerable")


def lower_single(
    v: Lowerable | type | OpView | Value | mlir.Type,
) -> mlir.Type | Value:
    """
    lower with the return value stripped of its tuple.
    Lowered output tuple must have length of exactly 1. Otherwise,
    `ValueError` is raised.

    This function is idempotent.
    """

    res = lower(v)
    if len(res) != 1:
        raise ValueError(f"lowering expected single element, got {res}")
    return res[0]


def lower_flatten(li):
    """
    Apply lower to each element of the list, then unpack the resulting tuples
    within the list.
    """
    output = []
    for elem in li:
        output.extend([*lower(elem)])

    return output
