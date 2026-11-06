import os
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import solve_discrete_are
from scipy.optimize import minimize_scalar

SEED = 42
OUT_DIR = "outputs"
L1_CSV = os.path.join(OUT_DIR, "l1_nbu_exchange_rates_1y.csv")
L1_COLUMN = "USD"
FALLBACK_PROPS = dict(n=366, coeffs=np.array([-0.00002201, 0.017799, 41.300157]), sigma=0.246, phi=0.946)


N0_INIT = 10
IDENT_FRAC = 1 / 3
SWING = 100
ANOM_RATE = 0.07
ANOM_MULT = (4.0, 8.0)
HAMPEL_WINDOW = 15
HAMPEL_K = 3.5
Q_LOG10_RANGE = (-10.0, 2.0)
MC_RUNS = 40

C = dict(
    raw="#78909C", anomaly="#EF5350", clean="#42A5F5", ab="#FFCA28",
    lsm_d="#EF5350", abg="#66BB6A", lsm_n="#AB47BC", extrap="#FF7043",
    true="#26C6DA", bg="#1A1D27", panel="#161B22", grid="#30363D",
    text="#E6EDF3", subtext="#8B949E",
)


def main():
    rng = np.random.default_rng(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    props = get_lab1_properties()
    n = props["n"]
    t = np.arange(n, dtype=float)
    ntr = int(n * IDENT_FRAC)


    print("============= 1.1: Input data =============")
    print(f"Source of properties: {props['source']}")
    print(f"Trend: y = {props['coeffs'][0]:.8f}*t^2 + {props['coeffs'][1]:.6f}*t + "
        f"{props['coeffs'][2]:.6f}; noise sigma = {props['sigma']:.4f}; N={n}")
    truth, z_noisy = make_signal("quad", n, props, rng)
    z_raw, anom_mask = inject_anomalies(z_noisy, props["sigma"], rng)


    print("\n============= 1.2: Data Exploration and Anomalies =============")
    sr = study_data(z_raw, t, "with anomalies")
    z, det = clean_data(z_raw, t)
    tp = int((det & anom_mask).sum())
    fp = int((det & ~anom_mask).sum())
    fn = int((~det & anom_mask).sum())
    print(f"Anomalies injected: {anom_mask.sum()}, detected {det.sum()} (TP={tp}, FP={fp}, FN={fn}); "
        f"RMSE relative to truth: {rmse(z_raw, truth):.4f} -> {rmse(z, truth):.4f} after compensation")


    print("\n============= 1.3-1.5: alpha-beta and alpha-beta-gamma =============")
    fits = {}
    for order, nm in ((1, "a-b"), (2, "a-b-g")):
        q, nll = tune_q(z, order, sr)
        X, nu, S, _, Ks = kalman_run(z, order, sr, q)
        g = steady_state_gains(order, sr, q)
        a_end, b_end = Ks[-1, 0], Ks[-1, 1]
        gtxt = ""
        diff = max(abs(a_end - g["alpha"]), abs(b_end - g["beta"]))
        if order == 2:
            gtxt = f", gamma={2 * Ks[-1, 2]:.6f}"
            diff = max(diff, abs(2 * Ks[-1, 2] - g["gamma"]))
        aic = 2 * nll + 2
        fits[nm] = dict(order=order, q=q, nll=nll, aic=aic, X=X, nu=nu, S=S, g=g)
        print(f"{nm:<6}: q/R={q:.3e}; final alpha={a_end:.4f}, beta={b_end:.5f}{gtxt}; "
            f"deviation from steady-state (Riccati) {diff:.1e}; -LL={nll:.2f}, AIC={aic:.2f}")
        print_report(nm, innovation_report(nu, S))
        err = X[N0_INIT:, 0] - truth[N0_INIT:]
        print(f"       relative to truth: mean={err.mean():+.4f}, RMSE={rmse(X[N0_INIT:, 0], truth[N0_INIT:]):.4f} "
            f"(measurement RMSE {rmse(z[N0_INIT:], truth[N0_INIT:]):.4f})")



    
    plot_input(t, truth, z_raw, z, det, anom_mask, f"{OUT_DIR}/l3_input_and_cleaning.png")
    plot_filters(t, truth, z, fits, f"{OUT_DIR}/l3_filters.png")
    plot_innovations(fits, f"{OUT_DIR}/l3_innovation_diagnostics.png")

# input data

def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def lag1_autocorr(r):
    r = r - np.mean(r)
    return float(np.sum(r[1:] * r[:-1]) / np.sum(r ** 2))


def get_lab1_properties():
    if os.path.exists(L1_CSV):
        y = pd.read_csv(L1_CSV)[L1_COLUMN].to_numpy(float)
        t = np.arange(len(y), dtype=float)
        coeffs = np.polyfit(t, y, 2)
        resid = y - np.polyval(coeffs, t)
        return dict(n=len(y), coeffs=coeffs, sigma=float(resid.std(ddof=1)),
                    phi=lag1_autocorr(resid), source=f"{L1_CSV} ({L1_COLUMN})", real=y)
    return dict(FALLBACK_PROPS, source="saved parameters of Lr_1 (USD)", real=None)


def make_signal(kind, n, props, rng, swing=SWING):
    t = np.arange(n, dtype=float)
    u = t / (n - 1)
    base, sigma = props["coeffs"][2], props["sigma"]
    sw = swing * sigma
    if kind == "quad":
        truth = np.polyval(props["coeffs"], t)
    elif kind == "linear":
        truth = base + sw * u
    elif kind == "cubic":
        truth = base + sw * (2 * u - 1) ** 3 / 2
    elif kind == "turn":
        truth = base + sw * np.where(u < 0.6, 0.5 * u, 0.3 + 4 * (u - 0.6))
    elif kind == "harmonic":
        truth = base + sw / 2 * np.sin(2 * np.pi * 1.5 * u)
    else:
        raise ValueError(kind)
    return truth, truth + rng.normal(0, sigma, n)


# ===

def inject_anomalies(y, sigma, rng, rate=ANOM_RATE, mult=ANOM_MULT):
    n = len(y)
    y_noisy = y.copy()
    n_anom = int(n * rate)
    idx = rng.choice(n, size=n_anom, replace=False)
    signs = rng.choice([-1, 1], size=n_anom)
    amps = rng.uniform(mult[0], mult[1], size=n_anom) * sigma
    y_noisy[idx] += signs * amps
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return y_noisy, mask


def hampel_detector(y, window=HAMPEL_WINDOW, k=HAMPEL_K):
    med = pd.Series(y).rolling(window, center=True, min_periods=1).median().to_numpy()
    resid = y - med
    mad = np.median(np.abs(resid - np.median(resid))) * 1.4826
    if mad == 0:
        mad = np.std(resid) or 1.0
    return np.abs(resid) > k * mad


def clean_data(y_noisy, t):
    detected = hampel_detector(y_noisy)
    y_cleaned = y_noisy.copy()
    if detected.any():
        good = ~detected
        y_cleaned[detected] = np.interp(t[detected], t[good], y_noisy[good])
    return y_cleaned, detected


def estimate_sigma_r(z, anomaly_mask=None):
    d2 = np.diff(z, 2)
    if anomaly_mask is not None:
        d2 = d2[~(anomaly_mask[:-2] | anomaly_mask[1:-1] | anomaly_mask[2:])]
    return float(1.4826 * np.median(np.abs(d2 - np.median(d2))) / math.sqrt(6))


def study_data(z, t, label):
    n = len(z)
    d2 = np.diff(z, 2)
    det = hampel_detector(z)
    sr = estimate_sigma_r(z, det)
    ok = ~(det[:-2] | det[1:-1] | det[2:])
    tau = t / t[-1]
    print(f"[{label}] N={n}, mean={z.mean():.4f}, std={z.std(ddof=1):.4f}, "
          f"sigma_R(MAD d2)={sr:.4f}, lag1(d2)={lag1_autocorr(d2):.3f}")
    print(f"    Hampel anomalies: {det.sum()} ({det.mean() * 100:.1f}%); "
          f"Shapiro(d2 without anomalies) p={stats.shapiro(d2[ok])[1]:.3f}")
    rss_prev = np.sum((z - z.mean()) ** 2)
    line = []
    for d in (1, 2, 3):
        rss = np.sum((z - np.polyval(np.polyfit(tau, z, d), tau)) ** 2)
        f = (rss_prev - rss) / (rss / (n - d - 1))
        line.append(f"deg{d}: p={stats.f.sf(f, 1, n - d - 1):.1e}")
        rss_prev = rss
    print("    Sequential F-test for significance of trend degrees: " + "; ".join(line))
    return sr



# state model and filters

def model(order):
    if order == 1:
        F = np.array([[1, 1], [0, 1.]])
        Qs = np.array([[1 / 3, 1 / 2], [1 / 2, 1.]])
    else:
        F = np.array([[1, 1, .5], [0, 1, 1], [0, 0, 1.]])
        Qs = np.array([[1 / 20, 1 / 8, 1 / 6], [1 / 8, 1 / 3, 1 / 2], [1 / 6, 1 / 2, 1.]])
    H = np.zeros((1, order + 1))
    H[0, 0] = 1
    return F, H, Qs


def ls_init(z0, order, sigma_r):
    m = len(z0)
    s = np.arange(m) - (m - 1.0)
    phi = np.column_stack([s ** j / math.factorial(j) for j in range(order + 1)])
    theta = np.linalg.lstsq(phi, z0, rcond=None)[0]
    return theta, sigma_r ** 2 * np.linalg.inv(phi.T @ phi), phi @ theta, s


def _fill_start(X, theta, s):
    for j in range(len(theta)):
        X[:len(s), j] = sum(theta[i] * s ** (i - j) / math.factorial(i - j)
                            for i in range(j, len(theta)))


def steady_state_gains(order, sigma_r, q_rel):
    F, H, Qs = model(order)
    R = sigma_r ** 2
    Q = q_rel * R * Qs
    try:
        Pp = solve_discrete_are(F.T, H.T, Q, np.array([[R]]))
    except Exception:
        Pp = np.eye(order + 1) * R
        for _ in range(20000):
            S = Pp[0, 0] + R
            Pn = F @ (Pp - np.outer(Pp[:, 0], Pp[:, 0]) / S) @ F.T + Q
            if np.max(np.abs(Pn - Pp)) < 1e-14:
                break
            Pp = Pn
    S = Pp[0, 0] + R
    K = Pp[:, 0] / S
    return dict(alpha=K[0], beta=K[1], gamma=2 * K[2] if order == 2 else None, S=S)


_CHI2_CACHE = {}


def _chi2_thr(m, n_tests):
    key = (m, n_tests)
    if key not in _CHI2_CACHE:
        _CHI2_CACHE[key] = stats.chi2.ppf(1 - FAMILY_ALPHA / n_tests, m)
    return _CHI2_CACHE[key]


def kalman_run(z, order, sigma_r, q_rel, adaptive=False, window=None, n0=N0_INIT):
    n = len(z)
    if window is None:
        window = int(math.ceil(math.sqrt(n)))
    F, H, Qs = model(order)
    R = sigma_r ** 2
    Q = q_rel * R * Qs
    HQH = (H @ Q @ H.T)[0, 0]
    theta, P, _, s = ls_init(z[:n0], order, sigma_r)
    x = theta.copy()
    X = np.zeros((n, order + 1))
    _fill_start(X, theta, s)
    nu = np.full(n, np.nan)
    Sk = np.full(n, np.nan)
    lam = np.ones(n)
    Ks = np.full((n, order + 1), np.nan)
    eye = np.eye(order + 1)
    buf = []
    for k in range(n0, n):
        xp = F @ x
        Pp0 = F @ P @ F.T
        e = z[k] - xp[0]
        l = 1.0
        hph = Pp0[0, 0]
        if adaptive:
            S0 = hph + HQH + R
            buf.append(e * e / S0)
            w = buf[-window:]
            m = len(w)
            T = float(np.sum(w))
            if T > _chi2_thr(m, n):
                l = max(1.0, (T / m * S0 - R - HQH) / hph)
        Pp = l * Pp0 + Q
        S = Pp[0, 0] + R
        K = Pp[:, 0] / S
        x = xp + K * e
        A = eye - np.outer(K, H[0])
        P = A @ Pp @ A.T + R * np.outer(K, K)
        X[k] = x
        nu[k], Sk[k], lam[k], Ks[k] = e, S, l, K
    return X, nu, Sk, lam, Ks



def tune_q(z, order, sigma_r, n0=N0_INIT):
    def nll(u):
        _, nu, S, _, _ = kalman_run(z, order, sigma_r, 10 ** u, n0=n0)
        return float(0.5 * np.sum(np.log(S[n0:]) + nu[n0:] ** 2 / S[n0:]))

    lo, hi = Q_LOG10_RANGE
    grid = np.arange(lo, hi + 1e-9, 2.0)
    vals = [nll(u) for u in grid]
    i = int(np.argmin(vals))
    res = minimize_scalar(nll, bounds=(grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]),
                          method="bounded", options=dict(xatol=0.05))
    return 10 ** res.x, float(res.fun)


# innovation diagnostics

def innovation_report(nu, S):
    nu = nu[~np.isnan(nu)]
    S = np.asarray(S)
    S = np.full(nu.shape, float(S)) if S.ndim == 0 else S[-len(nu):]
    m = len(nu)
    zn = nu / np.sqrt(S)
    d = nu - nu.mean()
    rho = [float(np.sum(d[k:] * d[:-k]) / np.sum(d ** 2)) for k in range(1, 11)]
    lb = m * (m + 2) * sum(r ** 2 / (m - k) for k, r in enumerate(rho, 1))
    nis = float(np.sum(zn ** 2))
    return dict(mean_nu=float(nu.mean()), p_mean=float(stats.ttest_1samp(nu, 0).pvalue),
                rho1=rho[0], p_lb=float(stats.chi2.sf(lb, 10)), nis_mean=nis / m,
                inconsistent=nis > stats.chi2.ppf(0.975, m) or nis < stats.chi2.ppf(0.025, m))


def print_report(name, rep):
    print(f"    {name:<15} mean(nu)={rep['mean_nu']:+.4f} (p={rep['p_mean']:.3f})  "
          f"lag1={rep['rho1']:+.3f}  Ljung-Box p={rep['p_lb']:.3f}  mean NIS={rep['nis_mean']:.2f} "
          f"({'inconsistent' if rep['inconsistent'] else 'consistent'})")





# plots

def ax_style(ax, title=""):
    ax.set_facecolor(C["bg"])
    ax.tick_params(colors="#AAAAAA")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333344")
    if title:
        ax.set_title(title, color="#FFFFFF", fontsize=10, fontweight="bold", pad=5)
    ax.xaxis.label.set_color("#AAAAAA")
    ax.yaxis.label.set_color("#AAAAAA")
    ax.margins(x=0.01)
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=7.5, facecolor=C["panel"], edgecolor=C["grid"],
                  labelcolor=C["text"], loc="best")
    ax.grid(linestyle="--", color="#555566", alpha=0.15, linewidth=0.6)


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"saved: {path}")


def plot_input(t, truth, z_raw, z, det, mask, path):
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=C["bg"])
    ax.plot(t, z_raw, color=C["raw"], lw=0.8, alpha=0.6, label="measurements with anomalies")
    ax.plot(t, z, color=C["clean"], lw=1.0, label="after anomaly compensation")
    ax.plot(t, truth, color=C["true"], lw=1.8, ls="--", label="true trend")
    ax.scatter(t[det], z_raw[det], color=C["anomaly"], s=22, zorder=5, label=f"detected ({det.sum()})")
    missed = mask & ~det
    if missed.any():
        ax.scatter(t[missed], z_raw[missed], color=C["ab"], marker="x", s=40, zorder=6,
                   label=f"missed ({missed.sum()})")
    ax_style(ax, "Input data, anomaly detection and compensation (Hampel filter)")
    ax.set_xlabel("t")
    ax.set_ylabel("Value")
    _save(fig, path)


def plot_filters(t, truth, z, fits, path):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), facecolor=C["bg"], sharex=True)
    ax = axes[0]
    ax.plot(t, z, color=C["raw"], lw=0.8, alpha=0.5, label="measurements (cleaned)")
    ax.plot(t, truth, color=C["true"], lw=1.8, ls="--", label="true trend")
    ax.plot(t, fits["a-b"]["X"][:, 0], color=C["ab"], lw=1.5, label="a-b")
    ax.plot(t, fits["a-b-g"]["X"][:, 0], color=C["abg"], lw=1.5, label="a-b-g")
    ax_style(ax, "Recursive smoothing")
    ax.set_ylabel("Value")
    ax = axes[1]
    ax.plot(t, fits["a-b"]["X"][:, 0] - truth, color=C["ab"], lw=1.0, label="a-b: error")
    ax.plot(t, fits["a-b-g"]["X"][:, 0] - truth, color=C["abg"], lw=1.0, label="a-b-g: error")
    ax.axhline(0, color=C["text"], lw=0.8, alpha=0.5)
    ax_style(ax, "Estimation error relative to the true trend (unbiasedness)")
    ax.set_xlabel("t")
    ax.set_ylabel("Error")
    _save(fig, path)


def plot_innovations(fits, path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor=C["bg"])
    for j, nm in enumerate(("a-b", "a-b-g")):
        nu = fits[nm]["nu"][N0_INIT:]
        nu_n = nu / np.sqrt(fits[nm]["S"][N0_INIT:])
        col = C["ab"] if nm == "a-b" else C["abg"]
        ax = axes[0, j]
        ax.hist(nu_n, bins=30, density=True, color=col, alpha=0.7, label="normalized innovations")
        xs = np.linspace(-4, 4, 200)
        ax.plot(xs, stats.norm.pdf(xs), color=C["text"], lw=1.2, label="N(0,1)")
        ax_style(ax, f"{nm}: innovations")
        ax = axes[1, j]
        lags = np.arange(1, 21)
        d = nu - nu.mean()
        acf = [np.sum(d[k:] * d[:-k]) / np.sum(d ** 2) for k in lags]
        ax.bar(lags, acf, color=col, alpha=0.8)
        b = 1.96 / math.sqrt(len(nu))
        ax.axhline(b, color=C["anomaly"], ls=":", label="95% white-noise bound")
        ax.axhline(-b, color=C["anomaly"], ls=":")
        ax_style(ax, f"{nm}: innovation autocorrelation")
        ax.set_xlabel("lag")
    _save(fig, path)





if __name__ == "__main__":
    main()

