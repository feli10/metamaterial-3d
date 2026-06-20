import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
from scipy.signal import medfilt

COMPONENTS = {
    1: "total",
    2: "reconstruction",
    3: "property prediction",
    4: "encoder epistemic",
    5: "decoder epistemic",
    6: "mid-value reconstruction",
}

repo = osp.dirname(osp.dirname(__file__))
ARCHIVE = osp.join(repo, "Archive")
runs = ["e1200", "e1200-1800"]

plt.figure(figsize=(8, 5))
for name in runs:
    train = np.genfromtxt(osp.join(ARCHIVE, name, "Freq_FNO_train.log"),
                          delimiter="\t", comments="ep")
    test  = np.genfromtxt(osp.join(ARCHIVE, name, "Freq_FNO_test.log"),
                          delimiter="\t", comments="ep")
    train[:, 1] = medfilt(train[:, 1], 5)   # smooth out numerical spikes
    test[:, 1]  = medfilt(test[:, 1], 3)

    # for col in [2, 3, 4, 5, 6]:
    #     plt.plot(train[:, 0], train[:, col], label=COMPONENTS[col], alpha=0.8)

    line, = plt.plot(train[:, 0], train[:, 1], label=f"{name} train", alpha=0.6)
    plt.plot(test[:, 0], test[:, 1], "--", label=f"{name} test")

plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Total loss")
plt.legend()
plt.savefig(osp.join(ARCHIVE, "loss_curves.png"), dpi=150)
plt.show()