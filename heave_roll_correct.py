import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression


# ====== CONFIG: nếu file 5 cột thì map theo đây ======
# dữ liệu mẫu: 1,579845.274,921207.775,-116.724,296675
# => Bm,E,N,Z,P
COLS_5 = ["Bm", "E", "N", "Z", "P"]


def load_txt(path: str, sep=None) -> pd.DataFrame:
    """
    Đọc txt/csv. Tự đoán dấu phân cách:
    - Nếu sep=None: pandas sẽ thử engine python với regex r"[,\s]+" (vừa comma vừa space).
    """
    if sep is None:
        df = pd.read_csv(path, header=None, engine="python", sep=r"[,\s]+")
    else:
        df = pd.read_csv(path, header=None, sep=sep)

    if df.shape[1] == 5:
        df.columns = COLS_5
    elif df.shape[1] == 6:
        # nếu bạn có đủ E,N,Z,P,Bm (hoặc thứ tự khác) thì sửa ở đây cho đúng
        # ví dụ: E,N,Z,P,Bm,Skip
        df.columns = ["E", "N", "Z", "P", "Bm", "Skip"]
        df = df.drop(columns=["Skip"])
    else:
        raise ValueError(f"Không hỗ trợ số cột={df.shape[1]}. Hãy chỉnh mapping cột.")

    # ép kiểu số
    for c in ["Bm", "E", "N", "Z", "P"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Bm", "E", "N", "Z", "P"])

    # đảm bảo integer cho ping/beam nếu cần
    df["P"] = df["P"].astype(np.int64)
    df["Bm"] = df["Bm"].astype(np.int64)

    return df


def preclean_outliers(df: pd.DataFrame, z_quantile_clip=(0.001, 0.999)) -> pd.DataFrame:
    """
    Tiền xử lý coarse outliers (đúng bước 1 trong bài).
    Cắt theo quantile để loại spike cực đoan làm hỏng regression.
    """
    lo, hi = df["Z"].quantile(list(z_quantile_clip))
    return df[(df["Z"] >= lo) & (df["Z"] <= hi)].copy()


def ping_spike_clean(
    df: pd.DataFrame, bm_min=0, bm_max=257, residual_tol=0.4, verbose=True
) -> pd.DataFrame:
    """
    Bước 3 (Cleansing loop): theo từng ping:
    fit LinearRegression(Z ~ Bm), tính residual |Z_pred - Z|,
    giữ điểm có residual < residual_tol.
    """
    model = LinearRegression()

    out_parts = []
    pings = np.sort(df["P"].unique())
    n = len(pings)

    for i, p in enumerate(pings, start=1):
        if verbose and (i % max(1, n // 20) == 0 or i == 1 or i == n):
            print(f"[CLEAN] ping {i}/{n} (P={p})")

        s = df[(df["P"] == p) & (df["Bm"] > bm_min) & (df["Bm"] < bm_max)]
        if len(s) < 5:
            continue

        X = s["Bm"].to_numpy().reshape(-1, 1)
        y = s["Z"].to_numpy()

        model.fit(X, y)
        z_pred = model.predict(X)
        dz = np.abs(z_pred - y)

        keep = dz < residual_tol
        out_parts.append(s.loc[keep, ["E", "N", "Z", "P", "Bm"]])

    if not out_parts:
        return df[["E", "N", "Z", "P", "Bm"]].copy()

    return pd.concat(out_parts, ignore_index=True)


def heave_roll_correct(
    df_clean: pd.DataFrame,
    window=600,  # số ping lân cận (±window ping)
    bm_min=5,
    bm_max=251,
    verbose=True,
) -> pd.DataFrame:
    """
    Sửa theo đúng tinh thần bài viết:
    - Fit model theo subset beam (bm_min..bm_max) để tránh mép swath
    - Nhưng áp correction lên FULL beam của ping hiện tại
    - Window theo index ping (±window ping), không dùng P±window (vì P có thể không liên tục)
    """
    model_cur = LinearRegression()
    model_win = LinearRegression()

    dfc = df_clean.sort_values(["P", "Bm"]).copy()
    pings = np.sort(dfc["P"].unique())
    n = len(pings)

    out_parts = []

    # tạo map ping -> index để lấy window theo vị trí
    ping_to_idx = {p: i for i, p in enumerate(pings)}

    for i, p in enumerate(pings, start=1):
        if verbose and (i % max(1, n // 20) == 0 or i == 1 or i == n):
            print(f"[CORR] ping {i}/{n} (P={p})")

        # FULL ping để output + plot
        slice_all = dfc[dfc["P"] == p].copy()
        if len(slice_all) < 5:
            continue

        # Subset để FIT (beam tốt)
        slice_fit = slice_all[
            (slice_all["Bm"] > bm_min) & (slice_all["Bm"] < bm_max)
        ].copy()
        if len(slice_fit) < 5:
            continue

        # Window theo INDEX ping
        idx = ping_to_idx[p]
        lo = max(0, idx - window)
        hi = min(n - 1, idx + window)
        p_lo = pings[lo]
        p_hi = pings[hi]

        slice_df = dfc[
            (dfc["P"] >= p_lo)
            & (dfc["P"] <= p_hi)
            & (dfc["Bm"] > bm_min)
            & (dfc["Bm"] < bm_max)
        ]
        if len(slice_df) < 20:
            slice_df = slice_fit

        # Fit model current ping (subset)
        X_fit = slice_fit["Bm"].to_numpy().reshape(-1, 1)
        y_fit = slice_fit["Z"].to_numpy()
        model_cur.fit(X_fit, y_fit)

        # Fit model window reference (subset)
        X_win = slice_df["Bm"].to_numpy().reshape(-1, 1)
        y_win = slice_df["Z"].to_numpy()
        model_win.fit(X_win, y_win)

        # Predict lên FULL ping
        X_all = slice_all["Bm"].to_numpy().reshape(-1, 1)
        slice_all["Z_T"] = model_cur.predict(X_all)
        slice_all["Z_TT"] = model_win.predict(X_all)

        # Correction
        slice_all["Z_cor"] = slice_all["Z"] + (slice_all["Z_TT"] - slice_all["Z_T"])

        out_parts.append(slice_all[["E", "N", "P", "Bm", "Z", "Z_T", "Z_TT", "Z_cor"]])

    return pd.concat(out_parts, ignore_index=True) if out_parts else pd.DataFrame()


def plot_before_after(
    df_raw: pd.DataFrame, df_corr: pd.DataFrame, out_png="before_after.png"
):
    """
    Vẽ scatter đơn giản (E vs N, màu theo Z/Z_cor) để bạn nhìn artefact trước/sau.
    Với file lớn, nên sample để vẽ nhanh.
    """

    def sample_df(d, max_points=200_000):
        if len(d) > max_points:
            return d.sample(max_points, random_state=0)
        return d

    r = sample_df(df_raw)
    c = sample_df(df_corr)

    plt.figure(figsize=(12, 4))
    plt.scatter(r["E"], r["N"], c=r["Z"], s=1)
    plt.title("BEFORE (color=Z)")
    plt.xlabel("E")
    plt.ylabel("N")
    plt.tight_layout()
    plt.savefig("before.png", dpi=200)
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.scatter(c["E"], c["N"], c=c["Z_cor"], s=1)
    plt.title("AFTER (color=Z_cor)")
    plt.xlabel("E")
    plt.ylabel("N")
    plt.tight_layout()
    plt.savefig("after.png", dpi=200)
    plt.close()

    print("Saved: before.png, after.png")


def plot_one_ping_profile(
    df_raw: pd.DataFrame, df_corr: pd.DataFrame, ping=None, out_png="ping_profile.png"
):
    if ping is None:
        ping = int(np.median(df_raw["P"].unique()))

    a = df_raw[df_raw["P"] == ping].sort_values("Bm")
    b = df_corr[df_corr["P"] == ping].sort_values("Bm")

    if len(a) == 0 or len(b) == 0:
        print(f"Không có dữ liệu ping={ping} để plot.")
        return

    plt.figure(figsize=(14, 5))
    plt.plot(a["Bm"], a["Z"], label="Z raw")

    # nếu có các cột model thì vẽ thêm
    if "Z_T" in b.columns:
        plt.plot(b["Bm"], b["Z_T"], label="Z_T (ping model)")
    if "Z_TT" in b.columns:
        plt.plot(b["Bm"], b["Z_TT"], label="Z_TT (window model)")

    plt.plot(b["Bm"], b["Z_cor"], label="Z corrected")
    plt.title(f"Ping profile P={ping}")
    plt.xlabel("Beam (Bm)")
    plt.ylabel("Depth (Z)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"Saved: {out_png}")


def main():
    in_path = "mydata.txt"  # đổi theo file của bạn
    out_path = "mydata_corrected.csv"

    # ====== tham số giống bài ======
    BM_CLEAN_MIN, BM_CLEAN_MAX = 0, 257
    RESIDUAL_TOL = 0.4

    WINDOW = 600
    BM_CORR_MIN, BM_CORR_MAX = 5, 251

    df = load_txt(in_path)
    print("Loaded:", df.shape)

    df0 = preclean_outliers(df, z_quantile_clip=(0.001, 0.999))
    print("After preclean:", df0.shape)

    df_clean = ping_spike_clean(
        df0,
        bm_min=BM_CLEAN_MIN,
        bm_max=BM_CLEAN_MAX,
        residual_tol=RESIDUAL_TOL,
        verbose=True,
    )
    print("After spike clean:", df_clean.shape)

    df_corr = heave_roll_correct(
        df_clean, window=WINDOW, bm_min=BM_CORR_MIN, bm_max=BM_CORR_MAX, verbose=True
    )
    print("Corrected:", df_corr.shape)
    # df_flat = flatten_by_ping(df_corr)
    # df_flat = detrend_along_track(df_flat)

    # chỉ giữ E, N, Z_cor
    df_out = df_corr[["E", "N", "Z_cor"]]
    df_out.to_csv(out_path, index=False)

    # df_corr.to_csv(out_path, index=False)
    print("Saved:", out_path)

    # plot
    plot_before_after(df0, df_corr)
    plot_one_ping_profile(df0, df_corr, ping=None, out_png="ping_profile.png")


if __name__ == "__main__":
    main()
