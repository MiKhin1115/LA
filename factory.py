"""
Streamlit — Simple Factory Optimizer (A & B) with Integer Inputs + Fairness
No PuLP, no MILP solver. Uses linear algebra corner enumeration.

Run:
  pip install streamlit numpy
  streamlit run app_simple.py
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import itertools
import math

import numpy as np
import streamlit as st


st.set_page_config(
    page_title="Simple Factory Optimizer (A & B)", layout="centered")


# ---------------- Data classes ----------------
@dataclass
class Product:
    profit: int
    demand_max: int  # use a large number if “unlimited”


@dataclass
class Resource:
    capacity: int


# ---------------- Small helpers ----------------
def feasible(A: np.ndarray, b: np.ndarray, x: np.ndarray, tol: float = 1e-8) -> bool:
    if not np.all(np.isfinite(x)):
        return False
    return np.all(A @ x <= b + tol)


def solve_factory_linear(
    products: Dict[str, Product],
    resources: Dict[str, Resource],
    consumption: Dict[Tuple[str, str], int],
    # sum_j a_j x_j <= rhs
    extra_ineq: Optional[List[Tuple[Dict[str, float], float]]] = None,
):
    """
    Maximize c^T x subject to A x <= b.
    Build all constraints (resource caps, demands, nonnegativity, and fairness), then
    enumerate all corners via linear algebra and pick the best profit.
    """
    names = list(products.keys())  # ["A", "B"]
    n = len(names)

    # Objective
    c = np.array([products[p].profit for p in names], dtype=float)

    # Build A x <= b
    Arows, bvals = [], []

    # Resource capacities
    for r, R in resources.items():
        Arows.append([consumption.get((r, p), 0) for p in names])
        bvals.append(R.capacity)

    # Upper bounds x_p <= demand_max
    for j, p in enumerate(names):
        row = [0] * n
        row[j] = 1
        Arows.append(row)
        bvals.append(products[p].demand_max)

    # Non-negativity -x_p <= 0
    for j in range(n):
        row = [0] * n
        row[j] = -1
        Arows.append(row)
        bvals.append(0)

    # Extra inequalities
    if extra_ineq:
        for coef, rhs in extra_ineq:
            Arows.append([float(coef.get(p, 0.0)) for p in names])
            bvals.append(float(rhs))

    A = np.array(Arows, dtype=float)
    b = np.array(bvals, dtype=float)

    # Enumerate corners (choose n binding constraints)
    best_val = -math.inf
    best_x = None
    best_bind = None

    m = A.shape[0]
    for idxs in itertools.combinations(range(m), n):
        Aeq = A[list(idxs), :]
        beq = b[list(idxs)]
        if np.linalg.matrix_rank(Aeq) < n:
            continue
        try:
            x = np.linalg.solve(Aeq, beq)
        except np.linalg.LinAlgError:
            continue
        if not feasible(A, b, x):
            continue
        z = float(c @ x)
        if z > best_val:
            best_val, best_x, best_bind = z, x, list(idxs)

    if best_x is None:
        return {"status": "Infeasible"}

    # Labels for readability
    labels = []
    for r in resources:
        labels.append(f"cap_{r}")
    for p in names:
        labels.append(f"ub_{p}")
    for p in names:
        labels.append(f"lb_{p}_ge0")
    if extra_ineq:
        for k in range(len(extra_ineq)):
            labels.append(f"extra_{k+1}")

    # Resource usage
    sol = {p: float(best_x[i]) for i, p in enumerate(names)}
    usage = []
    for r, R in resources.items():
        used = sum(consumption.get((r, p), 0) * sol[p] for p in names)
        usage.append({"resource": r, "used": used,
                     "capacity": R.capacity, "slack": R.capacity - used})

    return {
        "status": "Optimal",
        "objective": round(best_val, 2),
        "x": sol,
        "binding": [labels[i] for i in best_bind],
        "usage": usage,
    }


# ---------------- UI ----------------
st.title("Simple Factory Optimizer")

st.subheader("Products")
c1, c2 = st.columns(2)
with c1:
    profit_A = st.number_input(
        "Profit of A (per unit)", min_value=0, value=30, step=1)
    demand_A = st.number_input(
        "Max demand of A (units)", min_value=0, value=40, step=1)
with c2:
    profit_B = st.number_input(
        "Profit of B (per unit)", min_value=0, value=20, step=1)
    demand_B = st.number_input(
        "Max demand of B (units)", min_value=0, value=10000, step=1)

st.subheader("Resources")
r1, r2 = st.columns(2)
with r1:
    cap_M = st.number_input("Machine hours capacity",
                            min_value=0, value=100, step=1)
    mach_A = st.number_input("Machine hours per A",
                             min_value=0, value=2, step=1)
    mach_B = st.number_input("Machine hours per B",
                             min_value=0, value=1, step=1)
with r2:
    cap_R = st.number_input("Material (kg) capacity",
                            min_value=0, value=150, step=1)
    mat_A = st.number_input("Material kg per A", min_value=0, value=3, step=1)
    mat_B = st.number_input("Material kg per B", min_value=0, value=2, step=1)

st.subheader("Fairness (minimum units required)")
f1, f2 = st.columns(2)
with f1:
    min_A = st.number_input("A ≥ (units)", min_value=0, value=0, step=1)
with f2:
    min_B = st.number_input("B ≥ (units)", min_value=0, value=0, step=1)

if st.button("Optimize", type="primary"):
    products = {"A": Product(profit_A, demand_A),
                "B": Product(profit_B, demand_B)}
    resources = {"MachineHrs": Resource(cap_M), "MaterialKg": Resource(cap_R)}
    consumption = {
        ("MachineHrs", "A"): mach_A, ("MachineHrs", "B"): mach_B,
        ("MaterialKg", "A"): mat_A,  ("MaterialKg", "B"): mat_B,
    }

    # Extra constraints for fairness: xA ≥ min_A, xB ≥ min_B
    # Convert to form: sum a_j x_j ≤ rhs  by multiplying by -1:
    #   -xA ≤ -min_A,  -xB ≤ -min_B
    extra: List[Tuple[Dict[str, float], float]] = []
    if min_A > 0:
        extra.append(({"A": -1.0}, -float(min_A)))
    if min_B > 0:
        extra.append(({"B": -1.0}, -float(min_B)))

    res = solve_factory_linear(
        products, resources, consumption, extra_ineq=extra)

    if res["status"] != "Optimal":
        st.error(
            "No feasible solution. Try lowering minimums or increasing capacities/demand.")
    else:
        st.success("Optimal solution found ✅")
        st.write("**Quantities (units):**")
        # Show neat integers if very close to an integer; otherwise 2 decimals
        pretty = {}
        for k, v in res["x"].items():
            pretty[k] = int(round(v)) if abs(
                v - round(v)) < 1e-6 else round(v, 2)
        st.write(pretty)

        st.info(f"Optimal Profit: {res['objective']:.2f}")

        # st.write("**Binding constraints:**", ", ".join(res["binding"]) or "-")

        st.write("**Resource usage:**")
        for row in res["usage"]:
            used = row["used"]
            # pretty display for used too
            used_disp = int(round(used)) if abs(
                used - round(used)) < 1e-6 else round(used, 2)
            st.write(
                f"- {row['resource']}: used {used_disp} / {row['capacity']} · slack {round(row['slack'], 2)}")
