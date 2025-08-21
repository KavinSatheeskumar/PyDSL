from pydsl.type_temp import (
    PyDSLTypeMetaclass,
    PyDSLType,
    TargetType,
    lower_single,
    Bool,
    CompType
)
import mlir.ir as mlir
from mlir.ir import (
    Value,
    OpView,
    F16Type,
    F32Type,
    F64Type
)
from typing import Any, Type, Self, TYPE_CHECKING

import ctypes

from mlir.dialects import arith, math as mlirmath
from pydsl.protocols import ArgContainer

if TYPE_CHECKING:
    # This is for imports for type hinting purposes only and which can result
    # in cyclic imports.
    from pydsl.frontend import CTypeTree


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

    def to(self, target_type: Type[TargetType]) -> TargetType:
        if issubclass(target_type, Float):
            return self._to_Float(target_type)
        else:
            raise TypeError(
                f"Cannot cast floating point type to non floating point type {
                    target_type.__name__}"
            )

    def _to_Float(self, target_type: type[TargetType]) -> TargetType:
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

    # Unary Ops

    def op_neg(self) -> Self:
        return type(self)(arith.NegFOp(self.value))

    def op_pos(self) -> Self:
        return type(self)(self.value)

    # op invert undefined
    # op not undefined

    def op_abs(self) -> Self:
        return type(self)(mlirmath.AbsFOp(self.value))

    # Boolean Ops
    # Undefined

    def op_spaceship(self, rhs: PyDSLType, pred: CompType) -> Bool:
        if (casted_rhs := rhs.try_to(type(self))):
            rhs = casted_rhs
        else:
            return rhs.op_rspaceship(self)

        pred = {
            CompType.eq: arith.CmpFPredicate.OEQ,
            CompType.ne: arith.CmpFPredicate.ONE,
            CompType.lt: arith.CmpFPredicate.OLT,
            CompType.gt: arith.CmpFPredicate.OGT,
            CompType.le: arith.CmpFPredicate.OLE,
            CompType.ge: arith.CmpFPredicate.OGE
        }[pred]

        return Bool(
            arith.CmpFOp(pred, self.value, self._try_casting(rhs).value)
        )

    def op_rspaceship(self, lhs: PyDSLType, pred: CompType) -> Bool:
        lhs = lhs.to(type(self))
        return lhs.op_spaceship(self, pred)

    # Binary Ops
    def op_add(self, rhs: PyDSLType) -> Self:
        if (rhs_casted := rhs.try_to(type(self))):
            return type(self)(arith.AddFOp(self.value, rhs_casted.value))
        else:
            return rhs.op_radd(self)

    def op_radd(self, lhs: PyDSLType) -> Self:
        lhs_casted = lhs.to(type(self))
        return type(self)(arith.AddFOp(lhs_casted.value, self.value))

    def op_sub(self, rhs: PyDSLType) -> Self:
        if (rhs_casted := rhs.try_to(type(self))):
            return type(self)(arith.SubFOp(self.value, rhs_casted.value))
        else:
            return rhs.op_rsub(self)

    def op_rsub(self, lhs: PyDSLType) -> Self:
        lhs_casted = lhs.to(type(self))
        return type(self)(arith.SubFOp(lhs_casted.value, self.value))

    def op_mul(self, rhs: PyDSLType) -> Self:
        if (rhs_casted := rhs.try_to(type(self))):
            return type(self)(arith.MulFOp(self.value, rhs_casted.value))
        else:
            return rhs.op_rmul(self)

    def op_rmul(self, lhs: PyDSLType) -> Self:
        lhs_casted = lhs.to(type(self))
        return type(self)(arith.MulFOp(lhs_casted.value, self.value))

    def op_truediv(self, rhs: PyDSLType) -> Self:
        if (rhs_casted := rhs.try_to(type(self))):
            return type(self)(arith.DivFOp(self.value, rhs_casted.value))
        else:
            return rhs.op_truediv(self)

    def op_rtruediv(self, lhs: PyDSLType) -> Self:
        lhs_casted = lhs.to(type(self))
        return type(self)(arith.DivFOp(lhs_casted.value, self.value))

    def op_pow(self, rhs: PyDSLType) -> Self:
        if (rhs_casted := rhs.try_to(type(self))):
            return type(self)(mlirmath.PowFOp(self.value, rhs_casted.value))
        else:
            return rhs.op_rpow(self)

    def op_rpow(self, lhs: PyDSLType) -> Self:
        lhs_casted = lhs.to(type(self))
        return type(self)(mlirmath.PowFOp(lhs_casted.value, self.value))

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


class F16(Float, width=16, mlir_type=F16Type, ctype=ctypes.c_float):
    # float is not 16 bits, but ctypes has no 16 bit type
    pass


class F32(Float, width=32, mlir_type=F32Type, ctypes=ctypes.c_float):
    pass


class F64(Float, width=64, mlir_type=F64Type, ctypes=ctypes.c_double):
    pass
