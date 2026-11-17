"""Verification script for the audit note on
A. Granville, "Primes in intervals of bounded length",
Bull. Amer. Math. Soc. (N.S.) 52 (2015), 171-222.

Run:  python verify_granville_2015.py
Writes ../data/verify_granville_2015.txt and ../figures/fig_granville_2015_audit.pdf

Notation (Section 6.8 of the paper).  g(t) = 1/(1+At) on [0,T], 1+AT = e^A, so int g = 1.
For t' = (t_1..t_{k-1}) put P(t') = prod_{i<k} g(k t_i) and s = t_1+...+t_{k-1}.  Then
   A(t') = int_{t_k>=0} prod g = P/k,
   B(t') = int_{t_k >= 1-s} prod g = (P/k) * Gtail(k(1-s)),   Gtail(x) = int_{max(x,0)}^T g,
   A - B = (P/k) * Ghead(k(1-s)),  Ghead = 1 - Gtail.
Substituting u_i = k t_i, with S_{k-1} = u_1+...+u_{k-1}, u_i iid with density g^2/gamma on [0,T]:
   N_k      = k int (A-B)^2 = k^{-k} gamma^{k-1} E[Ghead(k - S_{k-1})^2]
   L_k      = k int A^2     = k^{-k} gamma^{k-1}
   k M_k    = k int B^2     = k^{-k} gamma^{k-1} E[Gtail(k - S_{k-1})^2]
   I_k (denominator)        = k^{-k} gamma^k   P(S_k <= k)
so  rho(F) = E[Ghead(k-S_{k-1})^2] / (gamma * P(S_k <= k))   exactly.
"""
import os
import sys

import numpy as np
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_TXT = os.path.join(ROOT, "data", "verify_granville_2015.txt")
OUT_FIG = os.path.join(ROOT, "figures", "fig_granville_2015_audit.pdf")

_lines = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    _lines.append(s)


def check(name, ok):
    say(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        check.failures += 1


check.failures = 0

# ---------------------------------------------------------------------------
# 1. Symbolic checks
# ---------------------------------------------------------------------------
say("=" * 72)
say("1. Symbolic checks")
say("=" * 72)
A, t, x, u = sp.symbols("A t x u", positive=True)
T = (sp.exp(A) - 1) / A
g = 1 / (1 + A * t)
gamma = sp.simplify(sp.integrate(g**2, (t, 0, T)))
mu = sp.simplify(sp.integrate(t * g**2, (t, 0, T)))
m2 = sp.simplify(sp.integrate(t**2 * g**2, (t, 0, T)))
m0 = sp.simplify(sp.integrate(g, (t, 0, T)))
check("gamma = (1-e^{-A})/A", sp.simplify(gamma - (1 - sp.exp(-A)) / A) == 0)
check("mu = (A-1+e^{-A})/A^2", sp.simplify(mu - (A - 1 + sp.exp(-A)) / A**2) == 0)
check("int t^2 g^2 = (e^A-2A-e^{-A})/A^3", sp.simplify(m2 - (sp.exp(A) - 2 * A - sp.exp(-A)) / A**3) == 0)
check("int g = 1", sp.simplify(m0 - 1) == 0)

eta_true = 1 - mu / gamma
eta_true_closed = (1 - (A + 1) * sp.exp(-A)) / (A * (1 - sp.exp(-A)))
eta_printed = (1 - (A - 1) * sp.exp(-A)) / (A * (1 - sp.exp(-A)))
check("eta = 1-mu/gamma = (1-(A+1)e^{-A})/(A(1-e^{-A}))", sp.simplify(eta_true - eta_true_closed) == 0)
diff_eta = sp.simplify(eta_printed - eta_true_closed)
say("   eta_printed - eta_true =", diff_eta)
check("eta_printed - eta_true = 2e^{-A}/(A(1-e^{-A})) (nonzero)",
      sp.simplify(diff_eta - 2 * sp.exp(-A) / (A * (1 - sp.exp(-A)))) == 0)

# middle expression of (6.8):  1/gamma - eta^{-2} (T/k) m2 / gamma
k = sp.symbols("k", positive=True)
pen = (T / k) * m2 / (eta_true**2 * gamma)
D = 1 - (A + 1) * sp.exp(-A)
pen_closed = sp.exp(2 * A) / (A * k) * (1 - 2 * A * sp.exp(-A) - sp.exp(-2 * A)) * (1 - sp.exp(-A))**2 / D**2
check("(6.8) middle term with eta_true: e^{2A}/(Ak) (1-2Ae^{-A}-e^{-2A})(1-e^{-A})^2/(1-(A+1)e^{-A})^2",
      sp.simplify(pen - pen_closed) == 0)
check("1/gamma = A/(1-e^{-A})", sp.simplify(1 / gamma - A / (1 - sp.exp(-A))) == 0)

# key identity: D^2 - (1-x)^2(1-2Ax-x^2) = x^2 (A-1+x)^2   (x = e^{-A})
idl = sp.expand((1 - (A + 1) * x)**2 - (1 - x)**2 * (1 - 2 * A * x - x**2) - x**2 * (A - 1 + x)**2)
check("(1-(A+1)x)^2 - (1-x)^2(1-2Ax-x^2) = x^2(A-1+x)^2  => R(A) <= 1", idl == 0)

# variance line on p.200: int prod g^2 ((1/k) sum u - mu/gamma)^2 vs printed (u1^2 - mu^2/gamma^2)
# both equal k^{-1}... check one-dimensional reduction:  E[(Xbar - m)^2] = (E X^2 - m^2)/k
# printed: k^{-1} * gamma^{k-1} * int g^2 (u^2 - mu^2/gamma^2) du  ; correct: same with (u - mu/gamma)^2
uu = sp.symbols("uu", positive=True)
gu = 1 / (1 + A * uu)
I_print = sp.integrate(gu**2 * (uu**2 - mu**2 / gamma**2), (uu, 0, T))
I_corr = sp.integrate(gu**2 * (uu - mu / gamma)**2, (uu, 0, T))
check("p.200 variance line: int g^2 (u^2-mu^2/gamma^2) == int g^2 (u-mu/gamma)^2 (printed line is CORRECT)",
      sp.simplify(I_print - I_corr) == 0)

# Young: (a-b)^2 - (1-th)(a^2 - b^2/th) = (sqrt(th) a - b/sqrt(th))^2 >= 0
a_, b_, th = sp.symbols("a b theta", positive=True)
yng = sp.simplify((a_ - b_)**2 - (1 - th) * (a_**2 - b_**2 / th) - (sp.sqrt(th) * a_ - b_ / sp.sqrt(th))**2)
check("Young: (a-b)^2 - (1-th)(a^2-b^2/th) = (sqrt(th)a - b/sqrt(th))^2", yng == 0)
check("(a-b)^2 - (a^2-b^2) = -2b(a-b)  (<= 0 for 0<=b<=a)", sp.expand((a_ - b_)**2 - (a_**2 - b_**2) + 2 * b_ * (a_ - b_)) == 0)
# max over theta of (1-th)(a - b/th) = (sqrt a - sqrt b)^2 at th = sqrt(b/a)
f_th = (1 - th) * (a_ - b_ / th)
check("max_theta (1-theta)(a-b/theta) = (sqrt a - sqrt b)^2 at theta = sqrt(b/a)",
      sp.simplify(f_th.subs(th, sp.sqrt(b_ / a_)) - (sp.sqrt(a_) - sp.sqrt(b_))**2) == 0)

# ---------------------------------------------------------------------------
# 2. Missing factor k in M_k
# ---------------------------------------------------------------------------
say("=" * 72)
say("2. Scaling of M_k (p.200): literal definition vs printed bound")
say("=" * 72)
say("   Cauchy on t_k-interval of length <= T/k:  (int_tail g(kt))^2 <= (T/k) int_tail g(kt)^2.")
say("   Literal M_k (no factor k):  M_k <= (T/k) * k^{-k} int_{sum u>=k} prod g(u_i)^2  = k^{-k-1} T (...)")
say("   Printed bound k^{-k} T (...) is the bound for k*M_k, i.e. the factor k was dropped in the definition.")

# ---------------------------------------------------------------------------
# 3. Exact numerics via FFT convolution
# ---------------------------------------------------------------------------
say("=" * 72)
say("3. Exact numerics: N_k vs L_k - k M_k, and rho(F)")
say("=" * 72)


def gfuncs(Aval):
    Tval = (np.exp(Aval) - 1) / Aval
    gam = (1 - np.exp(-Aval)) / Aval
    return Tval, gam


def pmf_cells(Aval, h):
    """masses of density g^2/gamma on cells [jh,(j+1)h] of [0,T] (last cell truncated)."""
    Tval, gam = gfuncs(Aval)
    edges = np.arange(0.0, Tval + h, h)
    edges[-1] = Tval
    G2 = lambda y: -1.0 / (Aval * (1 + Aval * y))  # antiderivative of g^2
    mass = (G2(edges[1:]) - G2(edges[:-1])) / gam
    centers = 0.5 * (edges[1:] + edges[:-1])
    return mass, centers


def sum_dist(mass, h, K):
    """pmf of the sum of K iid copies on grid j*h + K*h/2 (cell centres), via FFT power."""
    n = len(mass)
    L = K * (n - 1) + 1
    nfft = 1 << int(np.ceil(np.log2(L)))
    F = np.fft.rfft(mass, nfft)
    out = np.fft.irfft(F**K, nfft)[:L]
    out = np.clip(out, 0, None)
    out /= out.sum()
    pos = np.arange(L) * h + K * h / 2
    return out, pos


def Ghead(xv, Aval):
    Tval, _ = gfuncs(Aval)
    y = np.clip(xv, 0, Tval)
    return np.log1p(Aval * y) / Aval


def exact_quantities(kk, Aval):
    Tval, gam = gfuncs(Aval)
    h = min(Tval / 4000.0, 0.01)
    mass, _ = pmf_cells(Aval, h)
    pS1, pos1 = sum_dist(mass, h, kk - 1)
    gh = Ghead(kk - pos1, Aval)
    gt = 1 - gh
    N_over_L = float(np.sum(pS1 * gh**2))
    LkM_over_L = float(1 - np.sum(pS1 * gt**2))
    pS, pos = sum_dist(mass, h, kk)
    Pk = float(np.sum(pS[pos <= kk]))
    rho = N_over_L / (gam * Pk)
    return N_over_L, LkM_over_L, rho


say("   (a) Integrated ratios with fixed A = 2 (T = 3.19) and A = 3 (T = 6.36)")
say(f"   {'k':>4} {'A':>3} {'N_k/L_k':>12} {'(L_k-kM_k)/L_k':>16} {'gap':>11}")
ratio_rows = []
for Aval in (2.0, 3.0):
    for kk in (2, 3, 5, 10, 20, 50):
        nl, lm, _ = exact_quantities(kk, Aval)
        ratio_rows.append((kk, Aval, nl, lm))
        say(f"   {kk:>4} {Aval:>3.0f} {nl:>12.6f} {lm:>16.6f} {lm - nl:>11.3e}")
check("N_k < L_k - k M_k strictly for every tested k and A",
      all(r[3] - r[2] > 1e-6 for r in ratio_rows))

say("   (b) Exact rho(F) at Granville's choice A = (log k + log log k)/2 vs his claimed bound")
say(f"   {'k':>5} {'A':>7} {'rho(F) exact':>13} {'claim: (l+loglog)/2-2':>22} {'Tao upper':>10}")
rho_rows = []
for kk in (5, 10, 20, 50, 100, 200, 500, 1000):
    l = np.log(kk)
    Aval = 0.5 * (l + np.log(l))
    _, _, rho = exact_quantities(kk, Aval)
    claim = 0.5 * l + 0.5 * np.log(l) - 2
    tao = kk * l / (kk - 1)
    rho_rows.append((kk, Aval, rho, claim, tao))
    say(f"   {kk:>5} {Aval:>7.4f} {rho:>13.6f} {claim:>22.6f} {tao:>10.4f}")
check("exact rho(F) exceeds Granville's claimed lower bound at all tested k (the conclusion survives)",
      all(r[2] > r[3] for r in rho_rows))
check("exact rho(F) below Tao's upper bound k log k/(k-1)", all(r[2] < r[4] for r in rho_rows))

# ---------------------------------------------------------------------------
# 4. Proposition 1 (Young) and Proposition 2 (support truncation)
# ---------------------------------------------------------------------------
say("=" * 72)
say("4. Repaired lower bounds")
say("=" * 72)


def E_pen(Aval, kk):
    """Granville's (corrected) variance penalty eta^{-2}(T/k) int u^2 g^2 / gamma, true eta."""
    xx = np.exp(-Aval)
    Dv = 1 - (Aval + 1) * xx
    return np.exp(2 * Aval) / (Aval * kk) * (1 - 2 * Aval * xx - xx**2) * (1 - xx)**2 / Dv**2


def prop1_opt(l):
    """max over A>1, theta in (0,1) of (1-theta)(1/gamma - E/theta) = max_A (sqrt(1/gamma)-sqrt E)^2."""
    kk = np.exp(l)
    As = np.linspace(1.0001, max(1.01, min(l, 0.5 * l + 12)), 20000)
    a = As / (1 - np.exp(-As))
    b = E_pen(As, kk)
    val = np.where(b < a, (np.sqrt(a) - np.sqrt(b))**2, -np.inf)
    return float(val.max())


def prop1_simple(l):
    return 0.5 * l - 2 + 2 / l  # A = l/2: A(1 - e^A/(A sqrt k))^2 = (l/2)(1-2/l)^2


def prop2_exact(l, Aval=None):
    """rigorous Chebyshev form: (1/gamma) * (1 - (k-1) v / ((k-1) eta + 1 - T)^2), v = variance of g^2/gamma."""
    kk = np.exp(l)
    if Aval is None:
        Aval = l - 2 * np.log(l)
    xx = np.exp(-Aval)
    Tval = (np.exp(Aval) - 1) / Aval
    gam = (1 - xx) / Aval
    eta = (1 - (Aval + 1) * xx) / (Aval * (1 - xx))
    s2 = (np.exp(Aval) - 2 * Aval - xx) / (Aval**3 * gam)
    mean = 1 - eta
    v = s2 - mean**2
    den = (kk - 1) * eta + 1 - Tval
    if den <= 0:
        return -np.inf
    P = (kk - 1) * v / den**2
    return (1 / gam) * (1 - P)


def prop2_opt(l):
    As = np.linspace(1.5, l, 4000)
    return max(prop2_exact(l, a) for a in As)


# (i) (6.8) final inequality holds because R(A) <= 1
Agrid = np.linspace(1, 60, 100000)
xx = np.exp(-Agrid)
R = (1 - xx)**2 * (1 - 2 * Agrid * xx - xx**2) / (1 - (Agrid + 1) * xx)**2
check("R(A) <= 1 on A in [1,60] (so no 'borrowing' is needed for (6.8))", bool(np.all(R <= 1 + 1e-15)))

# (ii) Prop 1 simple bound vs optimum, and Prop 2 clean bound
lgrid = np.linspace(7, 200, 1500)
p1s = np.array([prop1_simple(l) for l in lgrid])
p2e = np.array([prop2_exact(l) for l in lgrid])
p2c = lgrid - 2 * np.log(lgrid) - 0.5
check("Prop 2: Chebyshev form at A = l - 2 log l  >=  log k - 2 log log k - 1/2  for log k in [7,200]",
      bool(np.all(p2e >= p2c)))
say(f"   min over grid of [Prop2 exact - (l - 2 log l)] = {np.min(p2e - (lgrid - 2 * np.log(lgrid))):.4f}")
# check analytic chain: P <= 1/(0.563 l^2)
def P_of(l):
    kk = np.exp(l); Aval = l - 2 * np.log(l)
    xx = np.exp(-Aval); Tval = (np.exp(Aval) - 1) / Aval; gam = (1 - xx) / Aval
    eta = (1 - (Aval + 1) * xx) / (Aval * (1 - xx))
    s2 = (np.exp(Aval) - 2 * Aval - xx) / (Aval**3 * gam); v = s2 - (1 - eta)**2
    return (kk - 1) * v / ((kk - 1) * eta + 1 - Tval)**2
check("analytic tail bound P <= 1/(0.563 (log k)^2) on log k in [7,200]",
      bool(all(P_of(l) <= 1 / (0.563 * l**2) for l in lgrid)))
for l in lgrid[:3]:
    Aval = l - 2 * np.log(l)
    check(f"A = l - 2 log l >= 3 at l = {l:.3f}", Aval >= 3)

say(f"   {'log k':>6} {'claim(6.8)':>11} {'Prop1 simple':>13} {'Prop1 opt':>10} {'Prop2 A=l-2logl':>16} {'Prop2 opt':>10} {'l-2loglog l-1/2':>16}")
for l in (8, 10, 12, 15, 20, 30, 40, 60, 100):
    say(f"   {l:>6} {0.5*l+0.5*np.log(l)-2:>11.4f} {prop1_simple(l):>13.4f} {prop1_opt(l):>10.4f} "
        f"{prop2_exact(l):>16.4f} {prop2_opt(l):>10.4f} {l-2*np.log(l)-0.5:>16.4f}")
check("Prop 1 optimum >= simple bound l/2 - 2 + 2/l",
      all(prop1_opt(l) >= prop1_simple(l) - 1e-9 for l in (8, 10, 15, 20, 30, 60)))

# ---------------------------------------------------------------------------
# 5. Theorem 6.4 / Corollary 6.5 thresholds
# ---------------------------------------------------------------------------
say("=" * 72)
say("5. Theorem 6.4: k minimal with k log k > e^{8m+4}; need rho(F) > 4m")
say("=" * 72)


def log_kmin(m):
    """log of the least integer k with k log k > e^{8m+4} (exact integer search for m <= 4)."""
    X = 8 * m + 4
    lo, hi = 1.0, float(X)  # solve l + log l = X
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if mid + np.log(mid) <= X else (lo, mid)
    if m <= 4:
        kk = int(np.floor(np.exp(hi))) - 2
        while not (kk * np.log(kk) > np.exp(X)):
            kk += 1
        return float(np.log(kk))
    return hi


def kmin(m):
    return np.exp(log_kmin(m))


say(f"   {'m':>3} {'log k_m':>9} {'Prop2 clean':>12} {'4m':>4} {'Prop1 simple':>13} {'Prop1 opt':>10}")
ok2 = True
p1fail = []
for m in range(1, 61):
    l = log_kmin(m)
    c2 = l - 2 * np.log(l) - 0.5
    ok2 &= (c2 > 4 * m) and l >= 7
    p1 = prop1_opt(l)
    if p1 <= 4 * m:
        p1fail.append(m)
    if m <= 6 or m in (10, 50, 100):
        say(f"   {m:>3} {l:>9.4f} {c2:>12.4f} {4*m:>4} {prop1_simple(l):>13.4f} {p1:>10.4f}")
check("Prop 2 gives rho(F) > 4m at k = k_m for every m in 1..60 (Theorem 6.4 recovered as stated)", ok2)
say(f"   Prop 1 (Young) fails to give rho > 4m at k_m for m in: {p1fail[:10]}{'...' if len(p1fail) > 10 else ''} ({len(p1fail)} of 60)")
ok1 = all(prop1_simple(8 * m + 4 + 1e-9) > 4 * m for m in range(1, 61))
check("Prop 1 gives rho > 4m as soon as log k > 8m+4 (Theorem 6.4 with k log k replaced by k)", ok1)

# ---------------------------------------------------------------------------
# 6. Section 11.2 CRT identity and Section 8.2 / 9.3 remarks
# ---------------------------------------------------------------------------
say("=" * 72)
say("6. Section 11.2: e_q(y) = e_{q1}(y * inv(q2) mod q1) e_{q2}(y * inv(q1) mod q2)")
say("=" * 72)
rng = np.random.default_rng(1)
ok_mod, bad_real = True, 0
for _ in range(2000):
    while True:
        q1, q2 = rng.integers(2, 60, size=2)
        if np.gcd(q1, q2) == 1:
            break
    q1, q2 = int(q1), int(q2)
    q = q1 * q2
    y = int(rng.integers(0, 10**6))
    i2 = pow(q2, -1, q1)
    i1 = pow(q1, -1, q2)
    lhs = np.exp(2j * np.pi * (y % q) / q)
    rhs_mod = np.exp(2j * np.pi * ((y * i2) % q1) / q1) * np.exp(2j * np.pi * ((y * i1) % q2) / q2)
    ok_mod &= abs(lhs - rhs_mod) < 1e-9
    rhs_real = np.exp(2j * np.pi * (y / q2) / q1) * np.exp(2j * np.pi * (y / q1) / q2)
    bad_real += abs(lhs - rhs_real) > 1e-9
check("CRT identity holds with modular inverses (2000 random coprime pairs)", ok_mod)
say(f"   With real division (e_m(t)=e(t/m) read literally) the identity fails in {bad_real}/2000 cases")
# reduction f(n + k q1) == f(n) mod q1 for f(n) = a nbar + b (n+Delta)bar + c n mod q
ok_red = True
for _ in range(500):
    while True:
        q1, q2 = (int(v) for v in rng.integers(3, 40, size=2))
        if np.gcd(q1, q2) == 1:
            break
    q = q1 * q2
    a, b, c, Dl = (int(v) for v in rng.integers(0, q, size=4))
    def f(n):
        if np.gcd(n, q) != 1 or np.gcd(n + Dl, q) != 1:
            return None
        return (a * pow(n, -1, q) + b * pow(n + Dl, -1, q) + c * n) % q
    n = int(rng.integers(1, 10**4)); kk = int(rng.integers(1, 50))
    f0, f1 = f(n), f(n + kk * q1)
    if f0 is None or f1 is None:
        continue
    ok_red &= (f0 - f1) % q1 == 0
check("f(n + k q1) == f(n) (mod q1) for rational f, so e_{q1}(inv(q2) f(n+kq1)) = e_{q1}(inv(q2) f(n))", ok_red)

say("=" * 72)
say(f"TOTAL FAILURES: {check.failures}")

# ---------------------------------------------------------------------------
# 7. Figure
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42, "axes.linewidth": 0.7,
})
C = {"true": "#1f4e79", "gran": "#c0392b", "young": "#e08e0b", "trunc": "#2e8b57", "tao": "#7f7f7f",
     "exact": "#000000", "print": "#c0392b"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.7))
fig.subplots_adjust(left=0.085, right=0.985, bottom=0.12, top=0.715, wspace=0.26)

# Panel (a)
r = np.linspace(0, 1, 400)
th0 = 0.5
h1, = ax1.plot(r, (1 - r)**2, color=C["true"], lw=1.6, label=r"true integrand $(A-B)^2$")
h2, = ax1.plot(r, 1 - r**2, color=C["gran"], lw=1.4, ls="--", label=r"$A^2-B^2$ (upper bound, used as lower)")
h3, = ax1.plot(r, (1 - th0) * (1 - r**2 / th0), color=C["young"], lw=1.4, ls="-.",
               label=r"Young: $(1-\theta)(A^2-\theta^{-1}B^2)$, $\theta=1/2$")
ax1.axhline(0, color="0.6", lw=0.5)
ax1.set_xlim(0, 1)
ax1.set_ylim(-0.55, 1.08)
ax1.set_xlabel(r"$B/A$  (pointwise, $A=1$)")
ax1.set_ylabel("integrand")
ax1.text(0.97, 0.95, "(a)", transform=ax1.transAxes, ha="right", va="top", fontsize=10, fontweight="bold")
ins = ax1.inset_axes([0.11, 0.10, 0.35, 0.28])
kk2 = [rw[0] for rw in ratio_rows if rw[1] == 2.0]
ins.plot(kk2, [rw[2] for rw in ratio_rows if rw[1] == 2.0], "o-", color=C["true"], ms=2.6, lw=1)
ins.plot(kk2, [rw[3] for rw in ratio_rows if rw[1] == 2.0], "s--", color=C["gran"], ms=2.6, lw=1)
ins.set_xscale("log")
ins.set_xticks([2, 5, 10, 20, 50])
ins.set_xticklabels(["2", "5", "10", "20", "50"])
ins.minorticks_off()
ins.tick_params(labelsize=6.5, length=2, pad=1)
ins.set_xlabel(r"$k$  ($A=2$)", fontsize=6.5, labelpad=0)
ins.set_ylabel("ratio", fontsize=6.5, labelpad=1)
ins.set_facecolor("white")

# Panel (b)
lx = np.linspace(7, 40, 400)
g1, = ax2.plot(lx, 0.5 * lx + 0.5 * np.log(lx) - 2, color=C["gran"], ls="--", lw=1.4,
               label=r"claimed (6.8): $\frac{1}{2}\log k+\frac{1}{2}\log\log k-2$")
g2, = ax2.plot(lx, [prop1_opt(l) for l in lx], color=C["young"], ls="-.", lw=1.4,
               label=r"Prop. 1 (Young, optimised $\theta$)")
g3, = ax2.plot(lx, [prop2_exact(l) for l in lx], color=C["trunc"], lw=1.6,
               label=r"Prop. 2 (truncation), $A=\log k-2\log\log k$")
g4, = ax2.plot(lx, lx, color=C["tao"], lw=1.0, ls=":", label=r"Tao upper bound $\approx\log k$")
mm = np.arange(1, 5)
lm_ = [log_kmin(m) for m in mm]
g5 = ax2.scatter(lm_, 4 * mm, marker="x", color="k", s=16, lw=1, zorder=5,
                 label=r"target $4m$ at $k\log k=e^{8m+4}$")
ax2.set_xlim(7, 40)
ax2.set_ylim(0, 41)
ax2.set_xlabel(r"$\log k$")
ax2.set_ylabel(r"lower bound for $\rho(F)$")
ax2.text(0.97, 0.05, "(b)", transform=ax2.transAxes, ha="right", va="bottom", fontsize=10, fontweight="bold")
ins2 = ax2.inset_axes([0.10, 0.60, 0.34, 0.35])
Ag = np.linspace(1, 8, 300)
xg = np.exp(-Ag)
ins2.plot(Ag, (1 - (Ag + 1) * xg) / (Ag * (1 - xg)), color=C["true"], lw=1.2)
ins2.plot(Ag, (1 - (Ag - 1) * xg) / (Ag * (1 - xg)), color=C["print"], lw=1.2, ls="--")
ins2.text(3.2, 0.45, r"$\eta_{\rm printed}$", color=C["print"], fontsize=6.5)
ins2.text(1.3, 0.06, r"$\eta_{\rm true}$", color=C["true"], fontsize=6.5)
ins2.set_xlabel(r"$A$", fontsize=6.5, labelpad=0)
ins2.tick_params(labelsize=6.5, length=2, pad=1)
ins2.set_ylim(0, 0.9)
ins2.set_facecolor("white")

handles = [h1, h2, h3, g1, g2, g3, g4, g5]
fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0),
           columnspacing=1.2, handlelength=2.4, borderaxespad=0.2)
fig.savefig(OUT_FIG)
fig.savefig(os.path.join(os.environ.get("AUDIT_PREVIEW_DIR", ROOT), "fig_preview.png"), dpi=170) if os.environ.get("AUDIT_PREVIEW_DIR") else None

with open(OUT_TXT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(_lines) + "\n")
sys.exit(1 if check.failures else 0)
