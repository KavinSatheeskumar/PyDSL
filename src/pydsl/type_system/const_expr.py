from pydsl.type_temp import PyDSLTypeMetaclass, TargetType, PyDSLType, Bool, CompType
from typing import Any, Type
import operator


class ConstExpr(metaclass=PyDSLTypeMetaclass):
    """
    A class that represents a generic constant whose value is determined
    lazily.

    All numeric literals in PyDSL evaluates to this type.
    """

    value: Any

    def __init__(self, rep: Any):
        match rep:
            case ConstExpr():
                self.value = rep.value
            case PyDSLType():
                raise TypeError(
                    f"Cannot cast non-const value {
                        type(rep).__name__} to ConstExpr"
                )
            case _:
                self.value = rep

    def to(self, target_type: Type[TargetType]) -> TargetType:
        return target_type(self.value)

    def op_spaceship(self, rhs: PyDSLType, pred: CompType) -> Bool:
        if (casted_rhs := rhs.try_to(type(self))):
            rhs = casted_rhs
        else:
            return rhs.op_rspaceship(self)

        pred = {
            CompType.eq: operator.eq,
            CompType.ne: operator.ne,
            CompType.lt: operator.lt,
            CompType.gt: operator.gt,
            CompType.le: operator.le,
            CompType.ge: operator.ge
        }[pred]

        return type(self)(pred(self.value, rhs.value))

    def op_rspaceship(self, lhs: PyDSLType, pred: CompType) -> Bool:
        lhs = lhs.to(type(self))
        return lhs.op_spaceship(self, pred)


# A lazy trick to set all the constant ops
for (name, op) in [
    ("neg", operator.neg),
    ("not", operator.not_),
    ("pos", operator.pos),
    ("abs", operator.abs),
    ("truth", operator.truth),
    ("invert", operator.invert),
]:
    def generic_unary_op(self):
        return ConstExpr(op(self.value))

    setattr(ConstExpr, f"op_{name}", op)

for (name, op) in [
    ("add", operator.add),
    ("sub", operator.sub),
    ("mul", operator.mul),
    ("truediv", operator.truediv),
    ("pow", operator.pow),
    ("divmod", divmod),
    ("floordiv", operator.floordiv),
    ("mod", operator.mod),
    ("lshift", operator.lshift),
    ("rshift", operator.rshift),
    ("and", operator.and_,),
    ("xor", operator.xor),
    ("or", operator.or_),
]:
    def generic_binary_op(self, rhs: PyDSLType):
        if (rhs_casted := rhs.try_to(type(self))):
            return op(self.value, rhs_casted.value)
        else:
            return getattr(f"op_r{name}", rhs)(self)

    def generic_binary_rop(self, lhs: PyDSLType):
        lhs_casted = lhs.to(type(self))
        return op(lhs_casted.value, self.value)

    setattr(ConstExpr, f"op_{name}", generic_binary_op)
    setattr(ConstExpr, f"op_r{name}", generic_binary_rop)
