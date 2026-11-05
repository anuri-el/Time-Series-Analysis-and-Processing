import os
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import solve_triangular

SEED = 42
OUT_DIR = "outputs"
L1_CSV = os.path.join(OUT_DIR, "l1_nbu_exchange_rates_1y.csv")
L1_COLUMN = "USD"
FALLBACK_PROPS = dict(n=366, coeffs=np.array([-0.00002201, 0.017799, 41.300157]), sigma=0.246, phi=0.946)


N0_INIT = 10
IDENT_FRAC = 1 / 3
SWING = 100
MAX_DEG = 8
FORECAST_FRAC = 0.5
CV_ORIGINS = (0.5, 0.6, 0.7, 0.8)
CV_HORIZON = 0.2
ANOM_RATE = 0.07
ANOM_MULT = (4.0, 8.0)
HAMPEL_WINDOW = 15
HAMPEL_K = 3.5
MC_RUNS = 100

C = dict(
    raw="#78909C", anomaly="#EF5350", clean="#42A5F5", ab="#FFCA28",
    lsm_d="#EF5350", lsm_c="#66BB6A", lsm_n="#AB47BC", extrap="#FF7043",
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

    print("============= 1.1 Input data =============")
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
    
    plot_input(t, truth, z_raw, z, det, anom_mask, f"{OUT_DIR}/l3_input_and_cleaning.png")


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
    ax.plot(t, z_raw, color=C["raw"], lw=0.8, alpha=0.6, label="вимірювання з аномаліями")
    ax.plot(t, z, color=C["clean"], lw=1.0, label="після компенсації аномалій")
    ax.plot(t, truth, color=C["true"], lw=1.8, ls="--", label="істинний тренд")
    ax.scatter(t[det], z_raw[det], color=C["anomaly"], s=22, zorder=5, label=f"виявлено ({det.sum()})")
    missed = mask & ~det
    if missed.any():
        ax.scatter(t[missed], z_raw[missed], color=C["ab"], marker="x", s=40, zorder=6,
                   label=f"пропущено ({missed.sum()})")
    ax_style(ax, "Вхідні дані, виявлення та компенсація аномалій (фільтр Хампеля)")
    ax.set_xlabel("t")
    ax.set_ylabel("Значення")
    _save(fig, path)







if __name__ == "__main__":
    main()

