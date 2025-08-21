from pydsl.type_temp import (
    PyDSLTypeMetaclass,
    PyDSLType,
    TargetType,
    lower_single,
    CompType,
    Bool
)
from enum import Enum
from mlir.ir import (
    # IndexType,
    IntegerType,
    # Operation,
    OpView,
    Value,
)
import mlir.ir as mlir
from pydsl.type_temp import Float
from pydsl.protocols import ArgContainer

from mlir.dialects import arith
from mlir.dialects import math as mlirmath
import math
from numbers import Real
from typing import Type, Self, Any, TYPE_CHECKING

import ctypes

if TYPE_CHECKING:
    # This is for imports for type hinting purposes only and which can result
    # in cyclic imports.
    from pydsl.frontend import CTypeTree


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
            case Real() if math.isclose(rep, int(rep)):
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

    # Unary Ops

    def op_neg(self) -> Self:
        # Integer negation and bitwise not are operations that are
        # not present in arith dialect, for reasons related to peepholes:
        # https://discourse.llvm.org/t/arith-noti/4844/3
        return type(self)(
            arith.SubIOp(
                arith.ConstantOp(lower_single(type(self)), 0), self.value
            )
        )

    def op_pos(self) -> Self:
        return type(self)(self.value)

    def op_invert(self) -> Self:
        # Integer negation and bitwise not are operations that are
        # not present in arith dialect, for reasons related to peepholes:
        # https://discourse.llvm.org/t/arith-noti/4844/3
        return type(self)(
            arith.SubIOp(
                arith.ConstantOp(lower_single(type(self)), -1), self.value
            ),
        )

    # op_not undefined

    def op_abs(self) -> Self:
        return type(self)(mlirmath.AbsIOp(self.value))

    # Boolean Ops
    # undefined

    # Comparison Ops

    def op_spaceship(self, rhs: PyDSLType, pred: CompType) -> Bool:
        if (casted_rhs := rhs.try_to(type(self))):
            rhs = casted_rhs
        else:
            return rhs.op_rspaceship(self)

        pred = {
            (Sign.SIGNED, CompType.eq): arith.CmpIPredicate.eq,
            (Sign.UNSIGNED, CompType.eq): arith.CmpIPredicate.eq,
            (Sign.SIGNED, CompType.ne): arith.CmpIPredicate.ne,
            (Sign.UNSIGNED, CompType.ne): arith.CmpIPredicate.ne,
            (Sign.SIGNED, CompType.lt): arith.CmpIPredicate.slt,
            (Sign.UNSIGNED, CompType.lt): arith.CmpIPredicate.ult,
            (Sign.SIGNED, CompType.gt): arith.CmpIPredicate.sgt,
            (Sign.UNSIGNED, CompType.gt): arith.CmpIPredicate.ugt,
            (Sign.SIGNED, CompType.le): arith.CmpIPredicate.sle,
            (Sign.UNSIGNED, CompType.le): arith.CmpIPredicate.ule,
            (Sign.SIGNED, CompType.ge): arith.CmpIPredicate.sge,
            (Sign.UNSIGNED, CompType.ge): arith.CmpIPredicate.uge
        }[(self.sign, pred)]
        return Bool(
            arith.CmpIOp(pred, self.value, self._try_casting(rhs).value)
        )

    def op_rspaceship(self, lhs: PyDSLType, pred: CompType) -> Bool:
        lhs = lhs.to(type(self))
        return lhs.op_spaceship(self, pred)

    # Binary Ops

    def op_add(self, rhs: PyDSLType) -> Self:
        if (rhs_casted := rhs.try_to(type(self))):
            return type(self)(arith.AddIOp(self.value, rhs_casted.value))
        else:
            return rhs.op_radd(self)

    def op_radd(self, lhs: PyDSLType) -> Self:
        lhs_casted = lhs.to(type(self))
        return type(self)(arith.AddIOp(lhs_casted.value, self.value))

    def op_sub(self, rhs: PyDSLType) -> Self:
        if (rhs_casted := rhs.try_to(self)):
            return type(self)(arith.SubIOp(self.value, rhs_casted.value))
        else:
            return rhs.op_rsub(self)

    def op_rsub(self, lhs: PyDSLType) -> Self:
        lhs_casted = lhs.to(type(self))
        return type(self)(arith.SubIOp(lhs_casted.value, self.value))

    def op_mul(self, rhs: PyDSLType) -> Self:
        if (rhs_casted := rhs.try_to(self)):
            return type(self)(arith.MulIOp(self.value, rhs_casted.value))
        else:
            return rhs.op_rmul(self)

    def op_rmul(self, lhs: PyDSLType) -> Self:
        lhs_casted = lhs.to(type(self))
        return type(self)(arith.MulIOp(lhs_casted.value, self.value))

    def op_floordiv(self, rhs: PyDSLType) -> Self:
        if (rhs_casted := rhs.try_to(type(self))):
            match self.sign:
                case Sign.SIGNED:
                    type(self)(
                        arith.FloorDivSIOp(self.value, rhs_casted.value)
                    )
                case Sign.UNSIGNED:
                    type(self)(arith.DivUIOp(self.value, rhs_casted.value))
        else:
            return rhs.op_rfloordiv(self)

    def op_rfloordiv(self, lhs: PyDSLType) -> Self:
        lhs_casted = lhs.to(type(self))
        match self.sign:
            case Sign.SIGNED:
                type(self)(arith.FloorDivSIOp(lhs_casted.value, self.value))
            case Sign.UNSIGNED:
                type(self)(arith.DivUIOp(lhs_casted.value, self.value))

    # TODO: truediv, mod, pow, lshift, rshift, bitand, bitor, bitxor

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


# It's worth noting that Python treat bool as an integer, meaning that e.g.
# (1 + True) == 2. It is also i1 in MLIR.
# To reflect this behavior, Bool inherits all integer operator overloading
# functions

class Bool(Int):
    def __init__(self, rep: Any) -> None:
        match rep:
            case bool():
                lit_as_bool = 1 if rep else 0
                self.value = arith.ConstantOp(
                    IntegerType.get_signless(1), lit_as_bool
                ).result
            case _:
                return super().__init__(rep)

    def op_and(self, rhs: PyDSLType) -> Self:
        if (rhs_casted := rhs.try_to(type(self))):
            return type(self)(arith.AndIOp(self.value, rhs_casted.value))
        else:
            return rhs.op_rand(self)

    def op_rand(self, lhs: PyDSLType) -> Self:
        lhs_casted = lhs.try_to(type(self))
        return type(self)(arith.AndIOp(lhs_casted.value, self.value))

    def op_or(self, rhs: PyDSLType) -> Self:
        if (rhs_casted := rhs.try_to(type(self))):
            return type(self)(arith.OrIOp(self.value, rhs_casted.value))
        else:
            return rhs.op_ror(self)

    def op_ror(self, lhs: PyDSLType) -> Self:
        lhs_casted = lhs.to(type(self))
        return type(self)(arith.OrIOp(lhs_casted.value, self.value))

    def op_not(self) -> Self:
        # MLIR doesn't seem to have bitwise not
        return Bool(
            arith.SelectOp(
                self.value,
                arith.ConstantOp(IntegerType.get_signless(1), 0).result,
                arith.ConstantOp(IntegerType.get_signless(1), 1).result,
            )
        )


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
