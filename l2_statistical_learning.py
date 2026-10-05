import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import solve_triangular

SEED = 42
OUT_DIR = "outputs"
L1_CSV = os.path.join(OUT_DIR, "l1_nbu_exchange_rates_2y.csv")
L1_COLUMN = "USD"
FALLBACK_PROPS = dict(n=366, coeffs=np.array([-0.00002201, 0.017799, 41.300157]), sigma=0.246, phi=0.946)

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
    raw="#78909C", anomaly="#EF5350", clean="#42A5F5", trend="#FFCA28",
    lsm_d="#EF5350", lsm_c="#66BB6A", lsm_n="#AB47BC", extrap="#FF7043",
    true="#26C6DA", bg="#1A1D27", panel="#161B22", grid="#30363D",
    text="#E6EDF3", subtext="#8B949E",
)

def main():
    rng = np.random.default_rng(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    props = get_lab1_properties()
    trend_fn = lambda x: np.polyval(props["coeffs"], x)
    t, trend, y_clean = generate_base_data(props, rng)
    n = len(t)
    tau = to_tau(t, t[0], t[-1] - t[0])

    print("============= Input data (properties of Lr_1) =============")
    print(f"Source of properties: {props['source']}")
    print(f"N={n}; trend: y = {props['coeffs'][0]:.8f}*t^2 + {props['coeffs'][1]:.6f}*t + "
        f"{props['coeffs'][2]:.6f}")
    print(f"AR(1) noise: sigma={props['sigma']:.4f}, phi={props['phi']:.3f}")
    print(f"Generated: mean={y_clean.mean():.4f}, std={y_clean.std(ddof=1):.4f}, "
        f"lag1(resid)={lag1_autocorr(y_clean - trend):.3f}")

    print("\n============= Anomalous measurements and cleaning =============")
    y_noisy, anom_mask = inject_anomalies(y_clean, props["sigma"], rng)
    y_cleaned, detected = clean_data(y_noisy, t)
    tp, fp, fn, prec, rec = detection_scores(detected, anom_mask)
    print(f"Anomalies injected: {anom_mask.sum()} ({anom_mask.mean() * 100:.1f}%), "
        f"amplitude {ANOM_MULT[0]}..{ANOM_MULT[1]} sigma")
    print(f"Method: Hampel filter (window {HAMPEL_WINDOW}, k={HAMPEL_K}); detected {detected.sum()}")
    print(f"TP={tp}, FP={fp}, FN={fn}, precision={prec:.3f}, recall={rec:.3f}")
    print(f"RMSE relative to ideal trend: with anomalies {rmse(trend, y_noisy):.4f}, "
        f"after cleaning {rmse(trend, y_cleaned):.4f}, without anomalies {rmse(trend, y_clean):.4f}")

    plot_input_data(t, trend, y_clean, y_noisy, anom_mask, f"{OUT_DIR}/l2_input_data.png")
    plot_cleaning_result(t, y_noisy, y_cleaned, anom_mask, detected, f"{OUT_DIR}/l2_cleaning_results.png")







# metrics

def rmse(y, y_hat):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(y_hat)) ** 2)))


def r_squared(y, y_hat):
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def adj_r_squared(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p)


def lag1_autocorr(r):
    r = r - np.mean(r)
    return float(np.sum(r[1:] * r[:-1]) / np.sum(r ** 2))


def compute_aic_bic(y, y_hat, k):
    n = len(y)
    rss = max(float(np.sum((y - y_hat) ** 2)), 1e-12)
    ll = -n / 2 * np.log(rss / n) - n / 2 * (1 + np.log(2 * np.pi))
    return 2 * (k + 1) - 2 * ll, np.log(n) * (k + 1) - 2 * ll


# group 1.1 and 2.1

def get_lab1_properties():
    if os.path.exists(L1_CSV):
        y = pd.read_csv(L1_CSV)[L1_COLUMN].to_numpy(float)
        t = np.arange(len(y), dtype=float)
        coeffs = np.polyfit(t, y, 2)
        resid = y - np.polyval(coeffs, t)
        return dict(n=len(y), coeffs=coeffs, sigma=float(resid.std(ddof=1)), phi=lag1_autocorr(resid), source=f"{L1_CSV} ({L1_COLUMN})")
    return dict(FALLBACK_PROPS, source="saved parameters of Lr_1 (USD)")


def generate_base_data(props, rng):
    n = props["n"]
    t = np.arange(n, dtype=float)
    trend = np.polyval(props["coeffs"], t)
    sigma, phi = props["sigma"], props["phi"]
    sigma_eps = sigma * np.sqrt(1 - phi ** 2)
    noise = np.empty(n)
    noise[0] = rng.normal(0, sigma)
    for k in range(1, n):
        noise[k] = phi * noise[k - 1] + rng.normal(0, sigma_eps)
    return t, trend, trend + noise


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


# group 1.2

def cv_rmse(tau, y, deg):
    n = len(y)
    h = int(CV_HORIZON * n)
    errs = []
    for f in CV_ORIGINS:
        k = int(f * n)
        theta, _, _ = lsm_fit(tau[:k], y[:k], deg)
        pred = design_matrix(tau[k:k + h], deg) @ theta
        errs.append(np.mean((y[k:k + h] - pred) ** 2))
    return float(np.sqrt(np.mean(errs)))


# group 2.2

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


def detection_scores(detected, mask):
    tp = int((detected & mask).sum())
    fp = int((detected & ~mask).sum())
    fn = int((~detected & mask).sum())
    return tp, fp, fn, tp / max(tp + fp, 1), tp / max(tp + fn, 1)


# group 1.3

def to_tau(t, t0, span):
    return (np.asarray(t, dtype=float) - t0) / span


def design_matrix(tau, deg):
    return np.vander(tau, deg + 1, increasing=False)


def lsm_fit(tau, y, deg):
    phi = design_matrix(tau, deg)
    q, r = np.linalg.qr(phi)
    theta = solve_triangular(r, q.T @ y)
    r_inv = solve_triangular(r, np.eye(r.shape[0]))
    return theta, phi @ theta, r_inv @ r_inv.T




# plots

def ax_style(ax, title=""):
    ax.set_facecolor(C["bg"])
    ax.tick_params(colors="#AAAAAA")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")
    if title:
        ax.set_title(title, color="#FFFFFF", fontsize=10, fontweight="bold", pad=5)
    ax.xaxis.label.set_color("#AAAAAA")
    ax.yaxis.label.set_color("#AAAAAA")
    ax.margins(x=0.01)
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=7.5, facecolor=C["panel"], edgecolor=C["grid"],
                  labelcolor=C["text"], loc="upper left")
    ax.grid(linestyle="--", color="#555566", alpha=0.15, linewidth=0.6)


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"saved: {path}")


def plot_input_data(t, trend, y_clean, y_noisy, anom_mask, path):
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=C["bg"])
    ax.plot(t, y_clean, color=C["clean"], lw=1, alpha=0.7, label="дані без аномалій")
    ax.plot(t, trend, color=C["true"], lw=1.8, ls="--", label="ідеальний тренд")
    ax.scatter(t[anom_mask], y_noisy[anom_mask], color=C["anomaly"], s=20, zorder=5,
               label=f"аномальні виміри ({anom_mask.sum()})")
    ax_style(ax, "Вхідні дані та аномалії")
    ax.set_xlabel("t")
    ax.set_ylabel("Значення")
    _save(fig, path)


def plot_cleaning_result(t, y_noisy, y_cleaned, mask, detected, path):
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=C["bg"])
    ax.plot(t, y_noisy, color=C["raw"], lw=0.8, alpha=0.6, label="з аномаліями")
    ax.plot(t, y_cleaned, color=C["clean"], lw=1.4, label="після очищення")
    ax.scatter(t[detected], y_noisy[detected], color=C["anomaly"], s=25, zorder=5,
               label=f"виявлено ({detected.sum()})")
    missed = mask & ~detected
    if missed.any():
        ax.scatter(t[missed], y_noisy[missed], color=C["trend"], marker="x", s=40, zorder=6,
                   label=f"пропущено ({missed.sum()})")
    ax_style(ax, "Очищення від аномалій (фільтр Хампеля)")
    ax.set_xlabel("t")
    ax.set_ylabel("Значення")
    _save(fig, path)









if __name__ == "__main__":
    main()
