
import sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hope_phy.data.llm4cp_dataset import LLM4CPDataset
from hope_phy.metrics.nmse import nmse

HIS = "data/llm4cp/Testing Dataset/H_U_his_test.mat"
PRE = "data/llm4cp/Testing Dataset/H_U_pre_test.mat"

def db(x):
    return float(10 * np.log10(max(float(x), 1e-12)))

def main():
    ds = LLM4CPDataset(HIS, PRE)
    X = torch.stack([ds[i][0] for i in range(len(ds))])
    Y = torch.stack([ds[i][1] for i in range(len(ds))])
    sp = ds.speed_index
    print("shapes", tuple(X.shape), tuple(Y.shape))
    P1 = X[:, -1:].repeat(1, Y.shape[1], 1)
    print("[1] persistence      :", round(db(nmse(P1, Y)), 2), "dB")
    slope = X[:, -1] - X[:, -2]
    steps = torch.arange(1, Y.shape[1] + 1).view(1, -1, 1).float()
    P2 = X[:, -1:] + slope.unsqueeze(1) * steps
    print("[2] linear extrap    :", round(db(nmse(P2, Y)), 2), "dB")
    print("    persistence per velocity:")
    for s in sorted(set(sp.tolist())):
        m = sp == s
        print("      ", (s+1)*10, "km/h :", round(db(nmse(P1[m], Y[m])), 2), "dB")
    Yf = Y.reshape(Y.shape[0], -1)
    Yc = Yf - Yf.mean(0, keepdim=True)
    sub = Yc[torch.randperm(Yc.shape[0])[:1000]]
    U, S, V = torch.linalg.svd(sub, full_matrices=False)
    energy = (S ** 2).cumsum(0) / (S ** 2).sum()
    print("[3] bottleneck ceiling:")
    for d in (64, 128, 256, 512, 1024):
        if d <= len(S):
            resid = 1.0 - float(energy[d - 1])
            print("      d=", d, ": residual", round(db(resid), 2), "dB",
                  "| energy kept", round(100*float(energy[d-1]), 1), "%")

main()
