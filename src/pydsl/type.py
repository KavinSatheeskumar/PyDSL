from __future__ import annotations

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
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    Self,
    runtime_checkable,
    Type
)

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

# TODO: make this aware of compilation target rather than always making
# the target the current machine this runs on


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
    def __init__(self, iterable: cabc.Iterable | Tuple | mlir.OpView):
        # this is the very bad casting code mentioned in the todo
        if isinstance(iterable, Tuple):
            iterable = iterable.value
        elif isinstance(iterable, mlir.OpView):
            iterable = iterable.results

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
