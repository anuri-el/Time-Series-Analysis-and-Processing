import os
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

    datasets = {"no_anomalies": y_clean, "anomalies": y_noisy, "cleaned": y_cleaned}
    fits = {}
    for name, y in datasets.items():
        print(f"\n============= Model: {name} =============")
        fit = fit_pipeline(t, tau, y, trend_fn, name)
        fits[name] = fit
        print_quality_table(fit["results"], fit["deg"])
        bic_best = min(fit["results"], key=lambda r: r["bic"])["deg"]
        print(f"Selected degree (min. CV-RMSE): {fit['deg']}; by BIC it would be: {bic_best}")
        print(f"theta (in terms of tau=(t-t0)/span, from highest degree): "
            + ", ".join(f"{v:.4f}" for v in fit["theta"]))
        fc = fit["y_ext"]
        print(f"Forecast for {len(fit['t_ext'])} points (0.5*N): [{fc[0]:.4f} .. {fc[-1]:.4f}], "
            f"ideal trend: [{fit['trend_ext'][0]:.4f} .. {fit['trend_ext'][-1]:.4f}]")
        print(f"Approximation RMSE relative to ideal trend: {rmse(trend, fit['y_hat']):.4f}")
        print(f"Forecast RMSE relative to ideal trend: {rmse(fit['trend_ext'], fc):.4f}")
        cover = np.mean((fit["trend_ext"] >= fit["lo"]) & (fit["trend_ext"] <= fit["hi"]))
        print(f"Share of ideal trend within 95% forecast interval: {cover * 100:.1f}%")

    print("\n============= Script verification =============")
    verify_script(tau, datasets, fits)
    print(f"--- Monte Carlo ({MC_RUNS} repetitions) ---")
    mc = monte_carlo(props)
    print(f"Anomaly detection: precision={mc.precision.mean():.3f}, recall={mc.recall.mean():.3f}")
    for name in datasets:
        col = mc[f"fc_{name}"]
        degs = mc[f"deg_{name}"].value_counts().sort_index().to_dict()
        print(f"{name:<13}: Forecast RMSE mean={col.mean():.3f}, median={col.median():.3f}, "
            f"std={col.std():.3f}; selected degrees: {degs}")
    better = (mc.fc_cleaned < mc.fc_anomalies).mean() * 100
    print(f"Cleaning improved the forecast in {better:.0f}% of repetitions")

    print("\n============= Results analysis =============")
    summary = pd.DataFrame({
        "model": list(datasets),
        "deg": [fits[k]["deg"] for k in datasets],
        "RMSE_fit_vs_trend": [rmse(trend, fits[k]["y_hat"]) for k in datasets],
        "RMSE_forecast_vs_trend": [rmse(fits[k]["trend_ext"], fits[k]["y_ext"]) for k in datasets],
    })
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    pd.DataFrame({
        "t": t, "trend": trend, "y_clean": y_clean, "y_noisy": y_noisy,
        "y_cleaned": y_cleaned, "anomaly_true": anom_mask, "anomaly_detected": detected,
        "lsm_no_anomalies": fits["no_anomalies"]["y_hat"],
        "lsm_anomalies": fits["anomalies"]["y_hat"],
        "lsm_cleaned": fits["cleaned"]["y_hat"],
    }).to_csv(f"{OUT_DIR}/l2_results.csv", index=False)
    f0 = fits["cleaned"]
    pd.DataFrame({
        "t": f0["t_ext"], "ideal_trend": f0["trend_ext"],
        **{f"forecast_{k}": fits[k]["y_ext"] for k in datasets},
        "cleaned_lo95": f0["lo"], "cleaned_hi95": f0["hi"],
    }).to_csv(f"{OUT_DIR}/l2_forecast.csv", index=False)
    mc.to_csv(f"{OUT_DIR}/l2_monte_carlo.csv", index=False)
    print(f"CSV saved to {OUT_DIR}/")

    plot_input_data(t, trend, y_clean, y_noisy, anom_mask, f"{OUT_DIR}/l2_input_data.png")
    plot_cleaning_result(t, y_noisy, y_cleaned, anom_mask, detected, f"{OUT_DIR}/l2_cleaning_results.png")
    plot_degree_selection(fits, f"{OUT_DIR}/l2_degree_selection.png")
    plot_regression_and_forecast(t, trend, datasets, fits, f"{OUT_DIR}/l2_lsm_regression_and_forecast.png")
    plot_residuals(t, y_cleaned, fits["cleaned"], f"{OUT_DIR}/l2_residuals.png")
    plot_monte_carlo(mc, f"{OUT_DIR}/l2_monte_carlo.png")


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


def evaluate_degrees(tau, y, max_deg=MAX_DEG):
    n = len(y)
    results = []
    for deg in range(1, max_deg + 1):
        theta, y_hat, _ = lsm_fit(tau, y, deg)
        aic, bic = compute_aic_bic(y, y_hat, deg + 1)
        r2 = r_squared(y, y_hat)
        results.append(dict(deg=deg, theta=theta, aic=aic, bic=bic, r2=r2,
                            adj_r2=adj_r_squared(r2, n, deg + 1),
                            rmse=rmse(y, y_hat), cv_rmse=cv_rmse(tau, y, deg)))
    best = min(results, key=lambda r: r["cv_rmse"])
    return best, results


def print_quality_table(results, best_deg):
    print(f"{'Deg':>4} {'RMSE':>8} {'R2':>8} {'adjR2':>8} {'AIC':>10} {'BIC':>10} {'CV-RMSE':>9}")
    for r in results:
        star = " <- BEST (CV)" if r["deg"] == best_deg else ""
        print(f"{r['deg']:>4} {r['rmse']:>8.4f} {r['r2']:>8.5f} {r['adj_r2']:>8.5f} "
              f"{r['aic']:>10.2f} {r['bic']:>10.2f} {r['cv_rmse']:>9.4f}{star}")


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


# group 1.4
def extrapolate(theta, XtX_inv, s, t, t0, span, frac=FORECAST_FRAC):
    n = len(t)
    n_ext = int(n * frac)
    dt = t[1] - t[0]
    t_ext = t[-1] + dt * np.arange(1, n_ext + 1)
    deg = len(theta) - 1
    phi_ext = design_matrix(to_tau(t_ext, t0, span), deg)
    y_ext = phi_ext @ theta
    se = s * np.sqrt(1 + np.einsum("ij,jk,ik->i", phi_ext, XtX_inv, phi_ext))
    q = stats.t.ppf(0.975, n - (deg + 1))
    return t_ext, y_ext, y_ext - q * se, y_ext + q * se


def fit_pipeline(t, tau, y, trend_fn, label):
    best, results = evaluate_degrees(tau, y)
    deg = best["deg"]
    theta, y_hat, XtX_inv = lsm_fit(tau, y, deg)
    n = len(y)
    s = float(np.sqrt(np.sum((y - y_hat) ** 2) / (n - deg - 1)))
    t_ext, y_ext, lo, hi = extrapolate(theta, XtX_inv, s, t, t[0], t[-1] - t[0])
    trend_ext = trend_fn(t_ext)
    return dict(label=label, best=best, results=results, deg=deg, theta=theta,
                y_hat=y_hat, s=s, t_ext=t_ext, y_ext=y_ext, lo=lo, hi=hi,
                trend_ext=trend_ext, XtX_inv=XtX_inv)


# group 1.7

def verify_script(tau, y_by_name, fits, rng_seed=0):
    print("--- MNK verification ---")
    true_theta = np.array([0.7, -1.9, 2.5])
    y_exact = design_matrix(tau, 2) @ true_theta
    theta_hat, _, _ = lsm_fit(tau, y_exact, 2)
    err = np.max(np.abs(theta_hat - true_theta))
    print(f"[1] Recovery of known coefficients (noise-free): max|dtheta|={err:.2e} "
          f"{'OK' if err < 1e-8 else 'FAIL'}")

    for name, fit in fits.items():
        y, deg = y_by_name[name], fit["deg"]
        diff = np.max(np.abs(fit["theta"] - np.polyfit(tau, y, deg)))
        print(f"[2] {name:<13} deg={deg}: max|theta_LSM - polyfit|={diff:.2e} "
              f"{'OK' if diff < 1e-6 else 'FAIL'}")
        resid = y - fit["y_hat"]
        ortho = np.max(np.abs(design_matrix(tau, deg).T @ resid))
        print(f"    {'':<13} orthogonality Phi^T*r: max={ortho:.2e} "
              f"{'OK' if ortho < 1e-6 else 'FAIL'}; "
              f"mean(r)={resid.mean():.2e}")


def monte_carlo(props, n_runs=MC_RUNS):
    rows = []
    for seed in range(1000, 1000 + n_runs):
        rng = np.random.default_rng(seed)
        t, trend, y0 = generate_base_data(props, rng)
        tau = to_tau(t, t[0], t[-1] - t[0])
        trend_fn = lambda x: np.polyval(props["coeffs"], x)
        y_n, mask = inject_anomalies(y0, props["sigma"], rng)
        y_c, det = clean_data(y_n, t)
        _, _, _, prec, rec = detection_scores(det, mask)
        row = dict(precision=prec, recall=rec)
        for name, y in (("no_anomalies", y0), ("anomalies", y_n), ("cleaned", y_c)):
            f = fit_pipeline(t, tau, y, trend_fn, name)
            row[f"fc_{name}"] = rmse(f["trend_ext"], f["y_ext"])
            row[f"deg_{name}"] = f["deg"]
        rows.append(row)
    return pd.DataFrame(rows)


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
    ax.plot(t, y_clean, color=C["clean"], lw=1, alpha=0.7, label="data without anomalies")
    ax.plot(t, trend, color=C["true"], lw=1.8, ls="--", label="ideal trend")
    ax.scatter(t[anom_mask], y_noisy[anom_mask], color=C["anomaly"], s=20, zorder=5,
               label=f"anomalous measurements ({anom_mask.sum()})")
    ax_style(ax, "Input data and anomalies")
    ax.set_xlabel("t")
    ax.set_ylabel("Value")
    _save(fig, path)


def plot_cleaning_result(t, y_noisy, y_cleaned, mask, detected, path):
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=C["bg"])
    ax.plot(t, y_noisy, color=C["raw"], lw=0.8, alpha=0.6, label="with anomalies")
    ax.plot(t, y_cleaned, color=C["clean"], lw=1.4, label="after cleaning")
    ax.scatter(t[detected], y_noisy[detected], color=C["anomaly"], s=25, zorder=5,
               label=f"detected ({detected.sum()})")
    missed = mask & ~detected
    if missed.any():
        ax.scatter(t[missed], y_noisy[missed], color=C["trend"], marker="x", s=40, zorder=6,
                   label=f"missed ({missed.sum()})")
    ax_style(ax, "Anomaly cleaning (Hampel filter)")
    ax.set_xlabel("t")
    ax.set_ylabel("Value")
    _save(fig, path)


def plot_degree_selection(fits, path):
    colors = dict(no_anomalies=C["lsm_n"], anomalies=C["lsm_d"], cleaned=C["lsm_c"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=C["bg"])
    for name, fit in fits.items():
        degs = [r["deg"] for r in fit["results"]]
        axes[0].plot(degs, [r["cv_rmse"] for r in fit["results"]], "o-", color=colors[name],
                     label=f"{name} (best deg={fit['deg']})")
        axes[0].axvline(fit["deg"], color=colors[name], ls=":", alpha=0.6)
        axes[1].plot(degs, [r["bic"] for r in fit["results"]], "o-", color=colors[name], label=name)
    axes[0].set_yscale("log")
    ax_style(axes[0], "CV-RMSE of forecast (selection criterion)")
    axes[0].set_xlabel("Polynomial degree")
    axes[0].set_ylabel("CV-RMSE")
    ax_style(axes[1], "BIC (for comparison)")
    axes[1].set_xlabel("Polynomial degree")
    axes[1].set_ylabel("BIC")
    _save(fig, path)


def plot_regression_and_forecast(t, trend, datasets, fits, path):
    colors = dict(no_anomalies=C["lsm_n"], anomalies=C["lsm_d"], cleaned=C["lsm_c"])
    fig, ax = plt.subplots(figsize=(13, 7), facecolor=C["bg"])
    ax.plot(t, datasets["anomalies"], color=C["raw"], lw=0.8, alpha=0.4, label="with anomalies")
    ax.plot(t, datasets["cleaned"], color=C["clean"], lw=0.9, alpha=0.6, label="cleaned")
    f_c = fits["cleaned"]
    ax.plot(t, trend, color=C["true"], lw=2, ls="--", alpha=0.8, label="ideal trend")
    ax.plot(f_c["t_ext"], f_c["trend_ext"], color=C["true"], lw=2, ls="--", alpha=0.8)
    for name, f in fits.items():
        ax.plot(t, f["y_hat"], color=colors[name], lw=1.8,
                label=f"MNK {name} (deg={f['deg']})")
        ax.plot(f["t_ext"], f["y_ext"], "--", color=colors[name], lw=1.8)
    ax.fill_between(f_c["t_ext"], f_c["lo"], f_c["hi"], color=C["lsm_c"], alpha=0.15,
                    label="95% forecast interval (cleaned)")
    ax.axvline(t[-1], color=C["text"], lw=1, ls=":", alpha=0.5)
    ax_style(ax, "Polynomial regression (MNK) and extrapolation to 0.5 of the sample")
    ax.set_xlabel("t")
    ax.set_ylabel("Value")
    _save(fig, path)


def plot_residuals(t, y_cleaned, fit, path):
    resid = y_cleaned - fit["y_hat"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=C["bg"])
    axes[0].bar(t, resid, width=1.0, color=C["lsm_c"], alpha=0.7)
    axes[0].axhline(0, color=C["text"], lw=0.8, alpha=0.5)
    ax_style(axes[0], f"MNK residuals (cleaned, deg={fit['deg']}), lag1={lag1_autocorr(resid):.2f}")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("Residual")
    (osm, osr), (slope, intercept, r) = stats.probplot(resid, dist="norm")
    axes[1].scatter(osm, osr, color=C["lsm_c"], s=8, alpha=0.7, label="residuals")
    xs = np.array([osm[0], osm[-1]])
    axes[1].plot(xs, slope * xs + intercept, color=C["anomaly"], label=f"r={r:.3f}")
    ax_style(axes[1], "Q-Q plot of residuals")
    axes[1].set_xlabel("Theoretical quantiles")
    axes[1].set_ylabel("Sample quantiles")
    _save(fig, path)


def plot_monte_carlo(mc, path):
    names = ["no_anomalies", "anomalies", "cleaned"]
    colors = [C["lsm_n"], C["lsm_d"], C["lsm_c"]]
    fig, ax = plt.subplots(figsize=(9, 6), facecolor=C["bg"])
    bp = ax.boxplot([mc[f"fc_{k}"] for k in names], patch_artist=True, showfliers=False,
                    medianprops=dict(color="#FFFFFF"))
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.8)
    ax.set_xticks(range(1, 4), names)
    ax_style(ax, f"Monte Carlo ({len(mc)} repetitions): forecast RMSE relative to ideal trend")
    ax.set_ylabel("Forecast RMSE")
    _save(fig, path)


if __name__ == "__main__":
    main()
