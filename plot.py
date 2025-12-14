import pandas as pd
import matplotlib.pyplot as plt


CSV = "out.csv"  # output file
PING_ID = None  # set to specific ping or None = first ping


df = pd.read_csv(CSV)

if PING_ID is None:
    PING_ID = df["Ping"].iloc[0]

d = df[df["Ping"] == PING_ID].sort_values("Beam")

plt.figure()
plt.plot(d["Beam"], d["Z"], "o-", label="Original Z", alpha=0.7)
plt.plot(d["Beam"], d["Z_corrected"], "o-", label="Corrected Z", alpha=0.7)
plt.plot(d["Beam"], d["Z_small_pred"], "--", label="Small model")
plt.plot(d["Beam"], d["Z_big_pred"], "--", label="Big model")

plt.gca().invert_yaxis()  # depth convention
plt.xlabel("Beam")
plt.ylabel("Depth (Z)")
plt.title(f"Heave/Roll correction – Ping {PING_ID}")
plt.legend()
plt.grid(True)
plt.show()
