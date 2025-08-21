import ast
import collections.abc as cabc
import ctypes
import math
import numbers
import operator
import sys
import typing
from collections import namedtuple
from enum import Enum
from functools import cache, reduce
from math import log2
from typing import TYPE_CHECKING, Any, Protocol, Self, runtime_checkable, Type, Self

import mlir.dialects.index as mlir_index
import mlir.ir as mlir
from mlir.dialects import arith, transform
from mlir.dialects import math as mlirmath
from mlir.dialects.transform.extras import OpHandle
from mlir.ir import (
    F16Type,
    F32Type,
    F64Type,
    IndexType,
    IntegerType,
    Operation,
    OpView,
    Value,
)

from pydsl.protocols import ToMLIRBase, ArgContainer

if TYPE_CHECKING:
    # This is for imports for type hinting purposes only and which can result
    # in cyclic imports.
    from pydsl.frontend import CTypeTree


class PyDSLTypeMetaclass(type):
    class PyDSLSuperclass():
        """
        A common super class so we can
        do isinstance(my_obj, PyDSLType)
        """
        # TODO: Maybe move lowering protocol here
        # Also maybe CType interface
        pass

    def __new__(cls, name, bases, attrs):
        return super().__new__(
            name, bases + [cls.PyDSLSuperclass], attrs
        )

    def on_Call(self, val: PyDSLSuperclass):
        return val.to(self)

    def op_is(self, val: Type[PyDSLSuperclass]):
        if (self == val):
            return Bool(arith.ConstantOp(Bool.lower_class(), 1))
        else:
            return Bool(arith.ConstantOp())


# A useful alias
PyDSLType = PyDSLTypeMetaclass.PyDSLSuperclass
TargetType = typing.TypeVar("TargetType", bound="PyDSLType")


class Sign(Enum):
    SIGNED = -1
    UNSIGNED = 1


class Int(metaclass=PyDSLTypeMetaclass):
    sign: Sign
    width: int
    value: Value

    def __init_subclass__(
        cls,
        width: int,
        sign: Sign,
        ctype: type,
        **kwargs
    ) -> None:
        super().__init_subclass__(**kwargs)
        cls.width = width
        cls.sign = sign
        cls.ctype = ctype

    def __init__(self, rep: Any):
        if not self.sign or not self.width or not self.ctype:
            raise TypeError(
                "Integer class must have sign and width specified")

        def _init_from_mlir_value(rep):
            if (rep_type := type(rep.type)) is not IntegerType:
                raise TypeError(
                    f"{rep_type.__name__} cannot be casted as a "
                    f"{type(self).__name__}"
                )
            if (width := rep.type.width) != self.width:
                raise TypeError(
                    f"{type(self).__name__} expected to have width of "
                    f"{self.width}, got {width}"
                )
            if not rep.type.is_signless:
                raise TypeError(
                    f"ops passed into {type(self).__name__} must have "
                    f"signless result, but was signed or unsigned"
                )

            self.value = rep

        match rep:
            case numbers.Real() if math.isclose(rep, int(rep)):
                rep = int(rep)

                if not self.in_range(rep):
                    raise ValueError(
                        f"{rep} is out of range for {type(self).__name__}"
                    )

                self.value = arith.ConstantOp(
                    self.lower_class()[0], rep
                ).result

            case Value():
                _init_from_mlir_value(rep)
            case OpView():
                _init_from_mlir_value(rep.value)
            case _:
                raise TypeError(
                    f"{rep} cannot be casted as {type(self).__name__}"
                )

    @classmethod
    def val_range(cls) -> tuple[int, int]:
        match cls.sign:
            case Sign.SIGNED:
                return (-(1 << (cls.width - 1)), (1 << (cls.width - 1)) - 1)
            case Sign.UNSIGNED:
                return (0, (1 << cls.width) - 1)
            case _:
                AssertionError("invalid sign")

    @classmethod
    def in_range(cls, val) -> bool:
        return cls.val_range()[0] <= val <= cls.val_range()[1]

    def lower(self) -> tuple[Value]:
        return (self.value,)

    @classmethod
    def lower_class(cls) -> tuple[mlir.Type]:
        return (IntegerType.get_signless(cls.width),)

    def _to_Int(self, target_type: Type[TargetType]) -> TargetType:
        if target_type.sign != self.sign:
            raise TypeError(
                "Int cannot be casted into another "
                "Int with differing signs"
            )

        if target_type.width < self.width:
            raise TypeError(
                f"Int of width {self.width} cannot be casted into width "
                f"{target_type.width}. Width must be extended"
            )

        if target_type.width == self.width:
            return target_type(self.value)

        if target_type.width > self.width:
            match self.sign:
                case Sign.SIGNED:
                    new_val = arith.ExtSIOp(
                        lower_single(target_type), lower_single(self)
                    )
                case Sign.UNSIGNED:
                    new_val = arith.ExtUIOp(
                        lower_single(target_type), lower_single(self)
                    )

            return target_type(new_val)

    def _to_Float(self, target_type: Type[TargetType]) -> Type:
        match self.sign:
            case Sign.SIGNED:
                return target_type(
                    arith.sitofp(lower_single(target_type),
                                 lower_single(self))
                )

            case Sign.UNSIGNED:
                return target_type(
                    arith.uitofp(lower_single(target_type),
                                 lower_single(self))
                )

    def to(self, target_type: Type[TargetType]) -> TargetType:
        if issubclass(target_type, Int):
            return self._to_Int(target_type)
        elif issubclass(target_type, Float):
            return self._to_Float(target_type)
        else:
            raise TypeError(
                f"Cannot cast integral type to non integral type {
                    target_type.__name__}"
            )

    @classmethod
    def op_abs(cls, t: Self) -> Self:
        return cls(mlirmath.AbsIOp(t.value))

    # TODO: figure out how to do unsigned -> signed conversion
    # TODO: these arith operators should have automatic width-expansion
    @classmethod
    def op_add(cls, lhs: Self, rhs: Self) -> Self:
        return cls(arith.AddIOp(lhs.value, rhs.value))

    @classmethod
    def op_sub(cls, lhs: Self, rhs: Self) -> Self:
        return cls(arith.SubIOp(lhs.value, rhs.value))

    @classmethod
    def op_mul(cls, lhs: Self, rhs: Self) -> Self:
        return cls(arith.MulIOp(lhs.value, rhs.value))

    def op_neg(self) -> Self:
        # Integer negation and bitwise not are operations that are
        # not present in arith dialect, for reasons related to peepholes:
        # https://discourse.llvm.org/t/arith-noti/4844/3
        return type(self)(
            arith.SubIOp(
                arith.ConstantOp(lower_single(type(self)), 0), self.value
            )
        )

    def op_invert(self) -> Self:
        # Integer negation and bitwise not are operations that are
        # not present in arith dialect, for reasons related to peepholes:
        # https://discourse.llvm.org/t/arith-noti/4844/3
        return type(self)(
            arith.SubIOp(
                arith.ConstantOp(lower_single(type(self)), -1), self.value
            ),
        )

    def op_pos(self) -> Self:
        return self  # yeah, this method didn't do anything before...

    # TODO: op_truediv cannot be implemented right now as it returns floating
    # points

    @classmethod
    def op_floordiv(cls, lhs: Self, rhs: Self) -> Self:
        # assertion ensures that self and rhs have the same sign
        if cls.sign == Sign.SIGNED:
            return cls(arith.FLoorDivSIOp(lhs.value, rhs.value))
        else:
            return cls(arith.DivUIOp(lhs.value, rhs.value))

    @classmethod
    def _compare_with_pred(
            cls, lhs: Self, rhs: Self, pred: arith.CmpIPredicate
    ):
        return Bool(
            arith.CmpIOp(pred, lhs.value, rhs.value)
        )

    @classmethod
    def op_lt(cls, lhs: Self, rhs: Self) -> "Bool":
        match cls.sign:
            case Sign.SIGNED:
                pred = arith.CmpIPredicate.slt
            case Sign.UNSIGNED:
                pred = arith.CmpIPredicate.ult

        return cls._compare_with_pred(lhs, rhs, pred)

    @classmethod
    def op_le(cls, lhs: Self, rhs: Self) -> "Bool":
        match cls.sign:
            case Sign.SIGNED:
                pred = arith.CmpIPredicate.sle
            case Sign.UNSIGNED:
                pred = arith.CmpIPredicate.ule

        return cls._compare_with_pred(lhs, rhs, pred)

    @classmethod
    def op_eq(cls, lhs: Self, rhs: Self) -> "Bool":
        return cls._compare_with_pred(lhs, rhs, arith.CmpIPredicate.eq)

    @classmethod
    def op_ne(cls, lhs: Self, rhs: Self) -> "Bool":
        return cls._compare_with_pred(lhs, rhs, arith.CmpIPredicate.ne)

    @classmethod
    def op_gt(cls, lhs: Self, rhs: Self) -> "Bool":
        match cls.sign:
            case Sign.SIGNED:
                pred = arith.CmpIPredicate.sgt
            case Sign.UNSIGNED:
                pred = arith.CmpIPredicate.ugt

        return cls._compare_with_pred(lhs, rhs, pred)

    @classmethod
    def op_ge(cls, lhs: Self, rhs: Self) -> "Bool":
        match cls.sign:
            case Sign.SIGNED:
                pred = arith.CmpIPredicate.sge
            case Sign.UNSIGNED:
                pred = arith.CmpIPredicate.uge

        return cls._compare_with_pred(lhs, rhs, pred)

    @classmethod
    def CType(cls) -> tuple[type]:
        return (cls.ctype,)

    @classmethod
    def to_CType(cls, arg_cont: ArgContainer, pyval: Any):
        try:
            pyval = int(pyval)
        except Exception as e:
            raise TypeError(
                f"{pyval} cannot be converted into an Int ctype"
            ) from e

        if not cls.in_range(pyval):
            lo, hi = cls.val_range()
            raise ValueError(
                f"{pyval} cannot fit into {cls.__qualname__}, must be in "
                f"the range [{lo}, {hi}]"
            )

        arg_cont.add_arg(pyval)
        return (pyval,)

    @classmethod
    def from_CType(cls, arg_cont: ArgContainer, cval: "CTypeTree"):
        return int(cval[0])


def get_operator(x):
    target = x

    if issubclass(type(target), Lowerable):
        target = lower_single(target)

    if isinstance(target, Value):
        target = target.owner

    if not (
        issubclass(type(target), OpView) or issubclass(type(target), Operation)
    ):
        raise TypeError(f"{x} cannot be cast into an operator")

    return target


def supports_operator(x):
    try:
        get_operator(x)
        return True
    except TypeError:
        return False


def iscompiled(x: Any) -> bool:
    """
    TODO: This is terrible and ugly code just to get things out of the way.

    Ideally, there should be a supertype of all PyDSL values called Value.
    """
    # Number is not lowerable but it is still returned as a SubtreeOut
    return isinstance(x, Number) or isinstance(x, Lowerable)


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
    # Uses map-reduce
    # Map:    lower each element
    # Reduce: flatten the resulting list of tuples into a list of its
    #         constituents
    return reduce(lambda a, b: a + [*b], map(lower, li), [])


AnyInt = typing.TypeVar("AnyInt", bound="Int")


class UInt8(Int, width=8, sign=Sign.UNSIGNED, ctype=ctypes.c_uint8):
    pass


class UInt16(Int, width=16, sign=Sign.UNSIGNED, ctype=ctypes.c_uint16):
    pass


class UInt32(Int, width=32, sign=Sign.UNSIGNED, ctype=ctypes.c_uint32):
    pass


class UInt64(Int, width=64, sign=Sign.UNSIGNED, ctype=ctypes.c_uint64):
    pass


class SInt8(Int, width=8, sign=Sign.SIGNED, ctype=ctypes.c_int8):
    pass


class SInt16(Int, width=16, sign=Sign.SIGNED, ctype=ctypes.c_int16):
    pass


class SInt32(Int, width=32, sign=Sign.SIGNED, ctype=ctypes.c_int32):
    pass


class SInt64(Int, width=64, sign=Sign.SIGNED, ctype=ctypes.c_int64):
    pass


# It's worth noting that Python treat bool as an integer, meaning that e.g.
# (1 + True) == 2. It is also i1 in MLIR.
# To reflect this behavior, Bool inherits all integer operator overloading
# functions

# TODO: Bool currently does not accept anything except for Python value.
# It should also support ops returning i1


class Bool(Int, width=1, sign=Sign.UNSIGNED, ctype=ctypes.c_bool):
    def __init__(self, rep: Any) -> None:
        match rep:
            case bool():
                lit_as_bool = 1 if rep else 0
                self.value = arith.ConstantOp(
                    IntegerType.get_signless(1), lit_as_bool
                ).result
            case _:
                return super().__init__(rep)

    @classmethod
    def op_and(cls, lhs: Self, rhs: Self) -> Self:
        return cls(arith.AndIOp(lhs.value, rhs.value))

    @classmethod
    def op_or(cls, lhs: Self, rhs: Self) -> Self:
        return cls(arith.OrIOp(lhs.value, rhs.value))

    def op_not(self) -> Self:
        # MLIR doesn't seem to have bitwise not
        return Bool(
            arith.SelectOp(
                self.value,
                arith.ConstantOp(IntegerType.get_signless(1), 0).result,
                arith.ConstantOp(IntegerType.get_signless(1), 1).result,
            )
        )


AnyFloat = typing.TypeVar("AnyFloat", bound="Float")


@runtime_checkable
class SupportsFloat(Protocol):
    def Float(self, target_type: type[AnyFloat]) -> AnyFloat: ...


class Float(metaclass=PyDSLTypeMetaclass):
    width: int
    mlir_type: mlir.Type
    ctype: type
    value: Value

    def __init_subclass__(
        cls, width: int, mlir_type: mlir.Type, ctype: type, **kwargs
    ) -> None:
        super().__init_subclass__(**kwargs)
        cls.width = width
        cls.mlir_type = mlir_type
        cls.ctype = ctype

    def __init__(self, rep: Any) -> None:
        if not all([self.width, self.mlir_type, self.ctype]):
            raise TypeError(
                "attempted to initialize Float without defined width, "
                "mlir_type or ctype"
            )

        # TODO: Code duplication in many classes. Consider a superclass?
        if isinstance(rep, OpView):
            rep = rep.result

        match rep:
            case float() | int() | bool():
                rep = float(rep)
                self.value = arith.ConstantOp(
                    self.lower_class()[0], rep
                ).result

            case Value():
                if (rep_type := type(rep.type)) is not self.mlir_type:
                    raise TypeError(
                        f"{rep_type.__name__} cannot be casted as a "
                        f"{type(self).__name__}"
                    )

                self.value = rep

            case _:
                raise TypeError(
                    f"{rep} cannot be casted as a {type(self).__name__}"
                )

    def lower(self) -> tuple[Value]:
        return (self.value,)

    @classmethod
    def lower_class(cls) -> tuple[mlir.Type]:
        return (cls.mlir_type.get(),)

    def op_abs(self) -> "Float":
        return type(self)(mlirmath.AbsFOp(self.value))

    @classmethod
    def op_add(cls, lhs: Self, rhs: Self) -> Self:
        return cls(arith.AddFOp(lhs.value, rhs.value))

    @classmethod
    def op_sub(cls, lhs: Self, rhs: Self) -> Self:
        return cls(arith.SubFOp(lhs.value, rhs.value))

    @classmethod
    def op_mul(cls, lhs: Self, rhs: Self) -> Self:
        return cls(arith.MulFOp(lhs.value, rhs.value))

    @classmethod
    def op_truediv(cls, lhs: Self, rhs: Self) -> Self:
        return cls(arith.DivFOp(lhs.value, rhs.value))

    def op_neg(self) -> "Float":
        return type(self)(arith.NegFOp(self.value))

    def op_pos(self) -> "Float":
        return self

    @classmethod
    def op_pow(cls, lhs: Self, rhs: Self) -> Self:
        return cls(mlirmath.PowFOp(lhs.value, rhs.value))

    @classmethod
    def _compare_with_pred(
        cls, lhs: Self, rhs: Self, pred: arith.CmpFPredicate
    ) -> Bool:
        return Bool(arith.CmpFOp(pred, lhs.value, rhs.value))

    @classmethod
    def op_lt(cls, lhs: Self, rhs: Self) -> Bool:
        return cls._compare_with_pred(lhs, rhs, arith.CmpFPredicate.OLT)

    @classmethod
    def op_le(cls, lhs: Self, rhs: Self) -> Bool:
        return cls._compare_with_pred(lhs, rhs, arith.CmpFPredicate.OLE)

    @classmethod
    def op_eq(cls, lhs: Self, rhs: Self) -> Bool:
        return cls._compare_with_pred(lhs, rhs, arith.CmpFPredicate.OEQ)

    @classmethod
    def op_ne(cls, lhs: Self, rhs: Self) -> Bool:
        return cls._compare_with_pred(lhs, rhs, arith.CmpFPredicate.ONE)

    @classmethod
    def op_gt(cls, lhs: Self, rhs: Self) -> Bool:
        return cls._compare_with_pred(lhs, rhs, arith.CmpFPredicate.OGT)

    @classmethod
    def op_ge(cls, lhs: Self, rhs: Self) -> Bool:
        return cls(lhs, rhs, arith.CmpFPredicate.OGE)

    # TODO: floordiv cannot be implemented so far. float -> int
    # needs floor ops.

    @classmethod
    def CType(cls) -> tuple[type]:
        return cls.ctype

    out_CType = CType

    @classmethod
    def to_CType(cls, arg_cont: ArgContainer, pyval: float | int | bool):
        try:
            pyval = float(pyval)
        except Exception as e:
            raise TypeError(
                f"{pyval} cannot be converted into a "
                f"{cls.__name__} ctype. Reason: {e}"
            )

        arg_cont.add_arg(pyval)
        return (pyval,)

    @classmethod
    def from_CType(cls, arg_cont: ArgContainer, cval: "CTypeTree"):
        return float(cval[0])

    def to(self, target_type: Type[TargetType]) -> TargetType:
        if issubclass(target_type, Float):
            return self._to_Float(target_type)
        else:
            raise TypeError(
                f"Cannot cast floating point type to non floating point type {
                    target_type.__name__}"
            )

    def _to_Float(self, target_type: type[AnyFloat]) -> AnyFloat:
        if target_type.width > self.width:
            return target_type(
                arith.extf(lower_single(target_type), lower_single(self))
            )

        if target_type.width == self.width:
            return target_type(self.value)

        if target_type.width < self.width:
            return target_type(
                arith.truncf(lower_single(target_type), lower_single(self))
            )


class F16(Float, width=16, mlir_type=F16Type, ctype=ctypes.c_float):
    # float is not 16 bits, but ctypes has no 16 bit type
    pass


class F32(Float, width=32, mlir_type=F32Type, ctypes=ctypes.c_float):
    pass


class F64(Float, width=64, mlir_type=F64Type, ctypes=ctypes.c_double):
    pass


# TODO: make this aware of compilation target rather than always making
# the target the current machine this runs on
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

    @classmethod
    def op_add(cls, lhs: Self, rhs: Self) -> Self:
        return cls(mlir_index.AddOp(lhs.value, rhs.value))

    @classmethod
    def op_sub(cls, lhs: Self, rhs: Self) -> Self:
        return cls(mlir_index.SubOp(lhs.value, rhs.value))

    @classmethod
    def op_mul(cls, lhs: Self, rhs: Self) -> Self:
        return cls(mlir_index.MulOp(lhs.value, rhs.value))

    @classmethod
    def op_truediv(cls, lhs: Self, rhs: Self) -> Float:
        raise NotImplementedError()  # TODO

    @classmethod
    def op_floordiv(cls, lhs: Self, rhs: Self) -> Self:
        return cls(mlir_index.FloorDivSOp(lhs.value, rhs.value))

    # TODO: maybe these should be unsigned ops. Actually, why do we treat Index
    # as an unsigned type and not a signed type?
    @classmethod
    def op_ceildiv(cls, lhs: Self, rhs: Self) -> Self:
        return cls(mlir_index.CeilDivSOp(lhs.value, rhs.value))

    @classmethod
    def CType(cls) -> tuple[type]:
        # TODO: this needs to be different depending on the platform.
        # On Python, you use sys.maxsize. However, we should let the
        # user choose in CompilationSetting

        return (ctypes.c_size_t,)

    @classmethod
    def PolyCType(cls) -> tuple[type]:
        return (ctypes.c_int,)


# TODO: this class should be renamed to TransformAnyOp to avoid confusion
# with MLIR's OpView subclasses
class AnyOp:
    value: OpHandle

    def __init__(self, rep: transform.AnyOpType) -> None:
        self.value = rep

    def lower(self) -> tuple[transform.AnyOpType]:
        return (self.value,)

    @classmethod
    def lower_class(cls) -> tuple[mlir.Type]:
        return (transform.AnyOpType.get(),)


NumberLike: typing.TypeAlias = typing.Union["Number", Int, Float, Index]


class Number:
    """
    A class that represents a generic number constant whose exact
    representation at runtime is evaluated lazily. As long as this type isn't
    used by an MLIR operator, it will only exist at compile-time.

    This type supports any value that is an instance of numbers.Number.

    All numeric literals in PyDSL evaluates to this type.

    See _NumberMeta for how its dunder functions are dynamically generated.
    """

    value: numbers.Number
    """
    The internal representation of the number.
    """

    def __init__(self, rep: numbers.Number):
        self.value = rep

    def to(self, target_type: Type[TargetType]) -> TargetType:
        return target_type(self.value)


# These are for unary operators in Number class
UnNumberOp = namedtuple("UnNumberOp", "dunder_name, internal_op")
un_number_op = {
    UnNumberOp("op_neg", operator.neg),
    UnNumberOp("op_not", operator.not_),
    UnNumberOp("op_pos", operator.pos),
    UnNumberOp("op_abs", operator.abs),
    UnNumberOp("op_truth", operator.truth),
    UnNumberOp("op_floor", math.floor),
    UnNumberOp("op_ceil", math.ceil),
    UnNumberOp("op_round", round),
    UnNumberOp("op_invert", operator.invert),
}

for tup in un_number_op:

    def method_gen(tup):
        """
        This function exists simply to allow a unique generic_unary_op to be
        generated whose variables are bound to the arguments of this function
        rather than the variable of the for loop.
        """
        # TODO: why is the above useful? What's wrong with binding to for loop
        # variables?
        _, internal_op = tup

        # perform the unary operation on the underlying value
        def generic_unary_op(self: Number) -> Number:
            return Number(internal_op(self.value))

        return generic_unary_op

    ldunder_name, internal_op = tup
    setattr(Number, ldunder_name, method_gen(tup))

# These are for binary operators in Number
BinNumberOp = namedtuple(
    "BinNumberOp", "ldunder_name, internal_op, rdunder_name"
)

bin_number_op = {
    BinNumberOp("op_add", operator.add, "op_radd"),
    BinNumberOp("op_sub", operator.sub, "op_rsub"),
    BinNumberOp("op_mul", operator.mul, "op_rmul"),
    BinNumberOp("op_truediv", operator.truediv, "op_rtruediv"),
    BinNumberOp("op_pow", operator.pow, "op_rpow"),
    BinNumberOp("op_divmod", divmod, "op_rdivmod"),
    BinNumberOp("op_floordiv", operator.floordiv, "op_rfloordiv"),
    BinNumberOp("op_mod", operator.mod, "op_rmod"),
    BinNumberOp("op_lshift", operator.lshift, "op_rlshift"),
    BinNumberOp("op_rshift", operator.rshift, "op_rrshift"),
    BinNumberOp("op_and", operator.and_, "op_rand"),
    BinNumberOp("op_xor", operator.xor, "op_rxor"),
    BinNumberOp("op_or", operator.or_, "op_ror"),
    BinNumberOp("op_lt", operator.lt, "op_gt"),
    BinNumberOp("op_le", operator.le, "op_ge"),
    BinNumberOp("op_eq", operator.le, "op_eq"),
    BinNumberOp("op_ge", operator.ge, "op_le"),
    BinNumberOp("op_gt", operator.gt, "op_lt"),
}


for tup in bin_number_op:
    """
    This dynamically add left-hand dunder operations to Number without
    repeatedly writing the code in generic_op.

    In order for this to work, new methods must be generated by returning
    a function where all of its variables are bound to arguments of its nested
    function (in this case, method_gen).
    """

    def method_gen(tup):
        """
        This function exists simply to allow a unique generic_bin_op to be
        generated whose variables are bound to the arguments of this function
        rather than the variable of the for loop.
        """
        _, internal_op, rdunder_name = tup

        @classmethod
        def generic_bin_op(cls, lhs: Self, rhs: Self) -> Self:
            return cls(internal_op(lhs.value, rhs.value))

        return generic_bin_op

    ldunder_name, internal_op, rdunder_name = tup
    setattr(Number, ldunder_name, method_gen(tup))


DTypes = typing.TypeVarTuple("DTypes")


class Tuple(typing.Generic[*DTypes], metaclass=PyDSLTypeMetaclass):
    """
    While tuple is not an MLIR type, it is still present in the language
    syntax-wise.

    This class mainly allows users to express multiple returns without having
    to rely on Python's built-in tuple type, which does not contain the
    necessary information for casting to Python CType.

    While it also allows users to group data together, this grouping only
    exists during compile-time and is not reflected in the code.
    As such, indexing cannot be performed, which makes this grouping rather
    useless.

    TODO: Below are some future design considerations:

    If tuple type was to become available and indexable at run-time, one
    would need to think of how to deal with a tuple with different types.
    Depending on the index, the return type of the operation can differ, which
    does not play nicely with MLIR's static type nature.

    One could also opt for a restrictive version of tuple that only allows
    index values known at compile-time so that the type can be inferred.

    One could also opt for a tuple that only accepts a single type, but that
    may be too restrictive to be useful.
    """

    _default_subclass_name = "TupleUnnamedSubclass"
    dtypes: tuple[type]
    value: tuple

    @staticmethod
    @cache
    def class_factory(
        dtypes: tuple[type], name=_default_subclass_name
    ) -> type["Tuple"]:
        """
        Create a new subclass of Tuple dynamically
        """
        if not isinstance(dtypes, cabc.Iterable):
            raise TypeError(
                f"MemRef requires dtypes to be iterable, got {type(dtypes)}"
            )

        if any([issubclass(t, Tuple) for t in dtypes]):
            raise TypeError("Tuples cannot be nested")

        return type(
            name,
            (Tuple,),
            {"dtypes": tuple(dtypes)},
        )

    # TODO: this will temporarily perform casting of other tuples
    # but later we will need to do a major refactor that shifts this behavior
    # to a dedicated member function called `cast`.
    def __init__(self, iterable: typing.Union[cabc.Iterable, "Tuple"]):
        # this is the very bad casting code mentioned in the todo
        if isinstance(iterable, Tuple):
            iterable = iterable.value

        error = TypeError(
            f"Tuple with dtypes {[t.__name__ for t in self.dtypes]} "
            f"received "
            f"{[type(v).__name__ for v in iterable]} types as values"
        )

        # If input is a tuple of a single tuple, stripe it down
        # Note that we do not allow tuples in tuples
        if len(iterable) == 1 and isinstance(iterable[0], Tuple):
            iterable = iterable.value

        if len(self.dtypes) != len(iterable):
            raise error

        # Check if input values match the dtype accepted by the tuple
        # If not, attempt element-wise casting
        try:
            casted = tuple([
                v if isinstance(v, t) else t(v)
                for v, t in zip(iterable, self.dtypes)
            ])
        except TypeError as casting_error:
            error.add_note(
                f"attempted casting failed with error message: "
                f"{repr(casting_error)}"
            )
            raise error from casting_error

        self.value = casted

    @classmethod
    def check_lowerable(cls) -> None | typing.Never:
        if not all([isinstance(t, Lowerable) for t in cls.dtypes]):
            raise TypeError(
                f"lowering a Tuple requires all of its dtypes to be "
                f"lowerable, got {cls.dtypes}"
            )

    def lower(self):
        self.check_lowerable()
        return lower_flatten(self.value)

    @classmethod
    def lower_class(cls) -> type:
        # Since MLIR FuncOps in MLIR accept tuple returns using plain Python
        # tuples
        cls.check_lowerable()
        return lower_flatten(cls.dtypes)

    @classmethod
    def on_class_getitem(
        cls, visitor: ToMLIRBase, slice: ast.AST
    ) -> type["Tuple"]:
        # TODO: this looks boilerplatey, maybe a helper function that takes
        # in a typing.Generic and do automatic binding of arguments?
        match slice:
            case ast.Tuple(elts=elts):
                args = [visitor.resolve_type_annotation(e) for e in elts]
            case t:
                args = [visitor.resolve_type_annotation(t)]

        dtypes = tuple(args)

        return cls.class_factory(dtypes)

    @classmethod
    def CType(cls) -> tuple[mlir.Type]:
        return tuple([d.CType() for d in cls.dtypes])

    @classmethod
    def to_CType(cls, arg_cont: ArgContainer, *_) -> typing.Never:
        raise TypeError("function arguments cannot have type Tuple")

    @classmethod
    def from_CType(cls, arg_cont: ArgContainer, ct: "CTypeTree") -> tuple:
        return tuple(
            t.from_CType(arg_cont, sub_ct)
            for t, sub_ct in zip(cls.dtypes, ct, strict=False)
        )

    @staticmethod
    def from_values(visitor: ToMLIRBase, *values):
        cls = Tuple.class_factory(tuple([type(v) for v in values]))
        return cls(values)

    def as_iterable(self: Self, visitor: "ToMLIRBase"):
        return self.value

    # TODO: maybe need dedicated PolyCType?


class Slice:
    """
    An object to represent a Python slice [lo:hi:step].
    If some of the arguments are missing (e.g. [::3]) they will be
    stored as None.
    Note that most MLIR functions that take in slice-like inputs have
    a different set of arguments from Python: they want lo, size, step.
    """

    lo: Index | None
    hi: Index | None
    step: Index | None

    def __init__(self, lo: Index | None, hi: Index | None, step: Index | None):
        self.lo = lo
        self.hi = hi
        self.step = step

    def get_args(self, max_size) -> tuple[Index, Index, Index]:
        """
        Returns [offset, size, step], which can be used for MLIR functions.
        Returns 0, max_size, 1 instead of None values, respectively.
        """
        lo = Index(0) if self.lo is None else self.lo
        hi = Index(max_size) if self.hi is None else self.hi
        step = Index(1) if self.step is None else self.step
        size = hi.op_sub(lo).op_ceildiv(step)
        return (lo, size, step)
