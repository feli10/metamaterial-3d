import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os.path as osp
from scipy.signal import medfilt

COMPONENTS = {
    1: "total",
    2: "reconstruction",
    3: "property prediction",
    4: "encoder variance",
    5: "decoder epistemic",
    6: "latent recovery",
}

repo = osp.dirname(osp.dirname(__file__))
ARCHIVE = osp.join(repo, "Archive")
runs = ["e1200"]

plt.figure(figsize=(8, 5))
for name in runs:
    train = np.genfromtxt(osp.join(ARCHIVE, name, "Freq_FNO_train.log"),
                          delimiter="\t", comments="ep")
    test  = np.genfromtxt(osp.join(ARCHIVE, name, "Freq_FNO_test.log"),
                          delimiter="\t", comments="ep")
    train[:, 1] = medfilt(train[:, 1], 3)   # smooth out numerical spikes
    test[:, 1]  = medfilt(test[:, 1], 3)

    for col in [2, 3, 4, 5, 6]:
        plt.plot(train[:, 0], train[:, col], label=COMPONENTS[col], alpha=0.8)

    # line, = plt.plot(train[:, 0], train[:, 1], label=f"{name} train", alpha=0.6)
    # plt.plot(test[:, 0], test[:, 1], "--", label=f"{name} test")

plt.yscale("log")

ax = plt.gca()
fmt = mticker.FuncFormatter(lambda y, _: f"{y:g}")
ax.yaxis.set_major_formatter(fmt)
ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=(2,4,6,8)))
ax.yaxis.set_minor_formatter(fmt)
ax.tick_params(axis="y", which="both", labelsize=8)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()