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

    # NOTE: Nếu có dòng lỗi/thiếu dữ liệu thì không thể "giữ điểm" vì bản thân điểm đó không có số.
    # Ta chỉ loại những dòng không thể dùng được.
    df = df.dropna(subset=["Bm", "E", "N", "Z", "P"]).copy()

    # đảm bảo integer cho ping/beam nếu cần
    df["P"] = df["P"].astype(np.int64)
    df["Bm"] = df["Bm"].astype(np.int64)

    # Identity để đảm bảo không mất điểm về sau
    df = df.reset_index(drop=True)
    df["_row_id"] = df.index

    return df


def preclean_outliers_inplace(df: pd.DataFrame, z_quantile_clip=(0.001, 0.999)) -> pd.DataFrame:
    """
    Tiền xử lý coarse outliers (đúng bước 1 trong bài) nhưng KHÔNG drop rows.
    Thay vì lọc bỏ điểm, ta winsorize/clip Z theo quantile để tránh spike cực đoan phá regression.
    """
    lo, hi = df["Z"].quantile(list(z_quantile_clip))
    df = df.copy()
    df["Z_preclean"] = df["Z"].clip(lo, hi)
    return df


def ping_spike_flag(
    df: pd.DataFrame, bm_min=0, bm_max=257, residual_tol=0.4, verbose=True
) -> pd.DataFrame:
    """
    Bước 3 (Cleansing loop) nhưng KHÔNG drop rows.

    Theo từng ping:
      - Fit LinearRegression(Z_preclean ~ Bm) trên subset beam (bm_min..bm_max)
      - residual = |Z_pred - Z_preclean|
      - Flag outlier: residual >= residual_tol
      - Tạo Z_despike để phục vụ fit:
            * điểm bình thường: Z_preclean
            * điểm spike: dùng Z_pred (thay thế) -> giúp model/rolling không bị lỗ
    """
    model = LinearRegression()
    df = df.copy()
    df["is_spike"] = False
    df["resid"] = np.nan
    df["Z_despike"] = df["Z_preclean"].astype(float)

    pings = np.sort(df["P"].unique())
    n = len(pings)

    for i, p in enumerate(pings, start=1):
        if verbose and (i % max(1, n // 20) == 0 or i == 1 or i == n):
            print(f"[FLAG] ping {i}/{n} (P={p})")

        m = (df["P"] == p) & (df["Bm"] > bm_min) & (df["Bm"] < bm_max)
        s = df.loc[m]
        if len(s) < 5:
            # không đủ điểm để fit -> giữ nguyên, không flag
            continue

        X = s["Bm"].to_numpy().reshape(-1, 1)
        y = s["Z_preclean"].to_numpy()

        model.fit(X, y)
        z_pred = model.predict(X)
        dz = np.abs(z_pred - y)

        spike = dz >= residual_tol

        # ghi flag/residual cho subset
        df.loc[m, "resid"] = dz
        # spike chỉ áp cho subset beam fit
        # (ngoài range bm_min..bm_max thì không đánh spike vì không fit)
        df.loc[s.index[spike], "is_spike"] = True
        # thay thế Z_despike bằng prediction cho spike (để dùng cho fit window)
        df.loc[s.index[spike], "Z_despike"] = z_pred[spike]

    return df


def heave_roll_correct_keep_all(
    df_in: pd.DataFrame,
    window=600,  # số ping lân cận (±window ping)
    bm_min=5,
    bm_max=251,
    verbose=True,
) -> pd.DataFrame:
    """
    Phiên bản KHÔNG LÀM MẤT ĐIỂM.

    - Fit model theo subset beam (bm_min..bm_max) để tránh mép swath
    - Model dùng Z_despike (đã thay spike) để ổn định
    - Nhưng áp correction lên FULL beam của ping hiện tại
    - Nếu ping nào không đủ dữ liệu để fit -> Z_cor = Z_preclean (hoặc Z), không bỏ ping
    """
    model_cur = LinearRegression()
    model_win = LinearRegression()

    dfc = df_in.sort_values(["P", "Bm", "_row_id"]).copy()
    pings = np.sort(dfc["P"].unique())
    n = len(pings)

    out_parts = []

    # map ping -> index để lấy window theo vị trí (P có thể không liên tục)
    ping_to_idx = {p: i for i, p in enumerate(pings)}

    for i, p in enumerate(pings, start=1):
        if verbose and (i % max(1, n // 20) == 0 or i == 1 or i == n):
            print(f"[CORR] ping {i}/{n} (P={p})")

        slice_all = dfc[dfc["P"] == p].copy()
        if len(slice_all) == 0:
            continue

        # Subset để FIT (beam tốt) + ưu tiên điểm không spike
        fit_mask = (slice_all["Bm"] > bm_min) & (slice_all["Bm"] < bm_max)
        slice_fit = slice_all.loc[fit_mask].copy()

        # Nếu quá ít điểm trong vùng fit, fallback: dùng tất cả beam
        if len(slice_fit) < 5:
            slice_fit = slice_all.copy()

        # Nếu vẫn quá ít -> không fit, giữ nguyên
        if len(slice_fit) < 5:
            slice_all["Z_T"] = np.nan
            slice_all["Z_TT"] = np.nan
            slice_all["Z_cor"] = slice_all["Z_preclean"]
            out_parts.append(slice_all[["E", "N", "P", "Bm", "Z", "Z_preclean", "is_spike", "resid", "Z_T", "Z_TT", "Z_cor", "_row_id"]])
            continue

        # Window theo INDEX ping
        idx = ping_to_idx[p]
        lo = max(0, idx - window)
        hi = min(n - 1, idx + window)
        p_lo = pings[lo]
        p_hi = pings[hi]

        # subset window dùng beam tốt
        slice_df = dfc[
            (dfc["P"] >= p_lo)
            & (dfc["P"] <= p_hi)
            & (dfc["Bm"] > bm_min)
            & (dfc["Bm"] < bm_max)
        ].copy()

        # Nếu window quá ít, fallback dùng slice_fit
        if len(slice_df) < 20:
            slice_df = slice_fit.copy()

        # Fit model current ping (subset)
        X_fit = slice_fit["Bm"].to_numpy().reshape(-1, 1)
        y_fit = slice_fit["Z_despike"].to_numpy()
        model_cur.fit(X_fit, y_fit)

        # Fit model window reference (subset)
        X_win = slice_df["Bm"].to_numpy().reshape(-1, 1)
        y_win = slice_df["Z_despike"].to_numpy()
        model_win.fit(X_win, y_win)

        # Predict lên FULL ping
        X_all = slice_all["Bm"].to_numpy().reshape(-1, 1)
        slice_all["Z_T"] = model_cur.predict(X_all)
        slice_all["Z_TT"] = model_win.predict(X_all)

        # Correction (giữ nguyên số điểm)
        slice_all["Z_cor"] = slice_all["Z_preclean"] + (slice_all["Z_TT"] - slice_all["Z_T"])

        out_parts.append(slice_all[["E", "N", "P", "Bm", "Z", "Z_preclean", "is_spike", "resid", "Z_T", "Z_TT", "Z_cor", "_row_id"]])

    if not out_parts:
        return pd.DataFrame()

    out = pd.concat(out_parts, ignore_index=True)

    # Restore exact original row order & assert no loss
    out = out.sort_values("_row_id")
    # N0 is not available here; keep-all guarantee via merge on _row_id below in main
    return out


def plot_before_after(df_raw: pd.DataFrame, df_corr: pd.DataFrame):
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


def plot_one_ping_profile(df_raw: pd.DataFrame, df_corr: pd.DataFrame, ping=None, out_png="ping_profile.png"):
    if ping is None:
        ping = int(np.median(df_raw["P"].unique()))

    a = df_raw[df_raw["P"] == ping].sort_values("Bm")
    b = df_corr[df_corr["P"] == ping].sort_values("Bm")

    if len(a) == 0 or len(b) == 0:
        print(f"Không có dữ liệu ping={ping} để plot.")
        return

    plt.figure(figsize=(14, 5))
    plt.plot(a["Bm"], a["Z"], label="Z raw")

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
    N0 = len(df)
    print("Loaded:", df.shape)

    df1 = preclean_outliers_inplace(df, z_quantile_clip=(0.001, 0.999))
    print("After preclean (no drop):", df1.shape)

    df2 = ping_spike_flag(
        df1,
        bm_min=BM_CLEAN_MIN,
        bm_max=BM_CLEAN_MAX,
        residual_tol=RESIDUAL_TOL,
        verbose=True,
    )
    print("After spike flag (no drop):", df2.shape)

    df_corr = heave_roll_correct_keep_all(
        df2, window=WINDOW, bm_min=BM_CORR_MIN, bm_max=BM_CORR_MAX, verbose=True
    )
    print("Corrected:", df_corr.shape)

    # GUARANTEE: keep all points by left-merge on _row_id
    base = df2[["_row_id", "E", "N", "P", "Bm", "Z", "Z_preclean", "is_spike", "resid"]].copy()
    df_corr_small = df_corr[["_row_id", "Z_T", "Z_TT", "Z_cor"]].copy()
    out = base.merge(df_corr_small, on="_row_id", how="left", validate="1:1")

    # If some rows couldn't be corrected (should be rare), fallback to Z_preclean
    out["Z_cor"] = out["Z_cor"].fillna(out["Z_preclean"])

    out = out.sort_values("_row_id")
    assert len(out) == N0, f"Lost points: input={N0}, output={len(out)}"

    # chỉ giữ E, N, Z_cor (như bạn đang làm)
    out_out = out[["E", "N", "Z_cor"]]
    out_out.to_csv(out_path, index=False)
    print("Saved:", out_path)

    # plot (before uses original Z, after uses corrected)
    plot_before_after(df, out)
    plot_one_ping_profile(df, out, ping=None, out_png="ping_profile.png")


if __name__ == "__main__":
    main()
