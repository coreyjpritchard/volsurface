"""Sparse linear programs for the quote checker and for SANOS.

``docs/DESIGN.md`` section 17. Rows are stored as SciPy sparse matrices and
solved with HiGHS. ``dense=True`` materialises the same coefficients so a
test can compare the two storages. Production calls leave it false.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.optimize import OptimizeResult, linprog


class ConstraintList:
    """Row, column, and value triplets for one block of constraints."""

    def __init__(self) -> None:
        self._rows: list[np.ndarray] = []
        self._cols: list[np.ndarray] = []
        self._vals: list[np.ndarray] = []
        self._rhs: list[float] = []

    def __len__(self) -> int:
        return len(self._rhs)

    def add(self, cols, vals, rhs: float) -> None:
        """Append one row. Zero entries are dropped."""
        columns = np.asarray(cols, dtype=np.int32).reshape(-1)
        values = np.asarray(vals, dtype=float).reshape(-1)
        if columns.shape != values.shape:
            raise ValueError(f"cols and vals differ in length ({columns.size}, {values.size})")
        keep = values != 0.0
        columns = columns[keep]
        values = values[keep]
        self._rows.append(np.full(columns.size, len(self._rhs), dtype=np.int32))
        self._cols.append(columns)
        self._vals.append(values)
        self._rhs.append(float(rhs))

    def to_csr(self, n_cols: int) -> tuple[sp.csr_matrix, np.ndarray]:
        """A CSR matrix with ``len(self)`` rows and ``n_cols`` columns, and the right-hand side."""
        n_rows = len(self._rhs)
        rhs = np.asarray(self._rhs, dtype=float)
        if n_rows == 0:
            return sp.csr_matrix((0, n_cols)), rhs
        nonempty = [chunk for chunk in self._rows if chunk.size]
        if not nonempty:
            return sp.csr_matrix((n_rows, n_cols)), rhs
        return (
            sp.csr_matrix(
                (
                    np.concatenate(self._vals),
                    (np.concatenate(self._rows), np.concatenate(self._cols)),
                ),
                shape=(n_rows, n_cols),
            ),
            rhs,
        )


def solve_linear(
    c: np.ndarray,
    *,
    A_ub=None,
    b_ub=None,
    A_eq=None,
    b_eq=None,
    bounds=None,
    method: str = "highs",
    dense: bool = False,
) -> OptimizeResult:
    """Solve ``min c' x`` with HiGHS. Sparse inputs stay sparse unless ``dense`` is set."""
    if dense:
        if sp.issparse(A_ub):
            A_ub = A_ub.toarray()
        if sp.issparse(A_eq):
            A_eq = A_eq.toarray()
    if A_ub is not None and getattr(A_ub, "shape", (0,))[0] == 0:
        A_ub = None
        b_ub = None
    return linprog(
        np.asarray(c, dtype=float),
        A_ub=A_ub,
        b_ub=None if b_ub is None else np.asarray(b_ub, dtype=float),
        A_eq=A_eq,
        b_eq=None if b_eq is None else np.asarray(b_eq, dtype=float),
        bounds=bounds,
        method=method,
    )
