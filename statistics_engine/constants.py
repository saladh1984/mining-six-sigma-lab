"""
Constantes de las cartas de control y de los estimadores de sigma.

Referencia: ASTM E2587 / Montgomery, D.C. (2013) "Introduction to Statistical
Quality Control", 7a ed., Apendice VI.

d2(n) = E[R]/sigma  ..... factor de sesgo del rango muestral
d3(n) = sd(R)/sigma ..... desviación estándar del rango muestral
c4(n) = E[S]/sigma  ..... factor de sesgo de la desviación estándar muestral

d2 y d3 se tabulan (no tienen forma cerrada elemental); c4 se calcula de forma
exacta con la función Gamma.
"""
from __future__ import annotations

import math

# ---------------------------------------------------------------- d2, d3 ----
D2_TABLE = {
    2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704, 8: 2.847,
    9: 2.970, 10: 3.078, 11: 3.173, 12: 3.258, 13: 3.336, 14: 3.407,
    15: 3.472, 16: 3.532, 17: 3.588, 18: 3.640, 19: 3.689, 20: 3.735,
    21: 3.778, 22: 3.819, 23: 3.858, 24: 3.895, 25: 3.931,
}

D3_TABLE = {
    2: 0.853, 3: 0.888, 4: 0.880, 5: 0.864, 6: 0.848, 7: 0.833, 8: 0.820,
    9: 0.808, 10: 0.797, 11: 0.787, 12: 0.778, 13: 0.770, 14: 0.763,
    15: 0.756, 16: 0.750, 17: 0.744, 18: 0.739, 19: 0.734, 20: 0.729,
    21: 0.724, 22: 0.720, 23: 0.716, 24: 0.712, 25: 0.708,
}

MAX_SUBGROUP = 25


def d2(n: int) -> float:
    """Factor d2 para subgrupos de tamaño n (2 <= n <= 25)."""
    _check_n(n)
    return D2_TABLE[n]


def d3(n: int) -> float:
    """Factor d3 para subgrupos de tamaño n (2 <= n <= 25)."""
    _check_n(n)
    return D3_TABLE[n]


def c4(n: int) -> float:
    """c4(n) = sqrt(2/(n-1)) * Gamma(n/2) / Gamma((n-1)/2). Exacto."""
    if n < 2:
        raise ValueError("c4 requiere n >= 2.")
    return math.sqrt(2.0 / (n - 1)) * math.gamma(n / 2.0) / math.gamma((n - 1) / 2.0)


# --------------------------------------------------- factores de cartas ----
def a2(n: int) -> float:
    """Carta Xbar-R: LC = Xbarbar +/- A2 * Rbar.  A2 = 3 / (d2 * sqrt(n))."""
    return 3.0 / (d2(n) * math.sqrt(n))


def d3_lower(n: int) -> float:
    """Carta R: LCI = D3 * Rbar.  D3 = max(0, 1 - 3*d3/d2)."""
    return max(0.0, 1.0 - 3.0 * d3(n) / d2(n))


def d4_upper(n: int) -> float:
    """Carta R: LCS = D4 * Rbar.  D4 = 1 + 3*d3/d2."""
    return 1.0 + 3.0 * d3(n) / d2(n)


def a3(n: int) -> float:
    """Carta Xbar-S: LC = Xbarbar +/- A3 * Sbar.  A3 = 3 / (c4 * sqrt(n))."""
    return 3.0 / (c4(n) * math.sqrt(n))


def b3(n: int) -> float:
    """Carta S: LCI = B3 * Sbar.  B3 = max(0, 1 - 3*sqrt(1-c4^2)/c4)."""
    k = c4(n)
    return max(0.0, 1.0 - 3.0 * math.sqrt(1.0 - k * k) / k)


def b4(n: int) -> float:
    """Carta S: LCS = B4 * Sbar.  B4 = 1 + 3*sqrt(1-c4^2)/c4."""
    k = c4(n)
    return 1.0 + 3.0 * math.sqrt(1.0 - k * k) / k


def _check_n(n: int) -> None:
    if n not in D2_TABLE:
        raise ValueError(
            f"Tamaño de subgrupo n={n} fuera de la tabla (2 a {MAX_SUBGROUP}). "
            "Para n=1 use la carta I-MR; para n>25 use la carta Xbar-S."
        )
