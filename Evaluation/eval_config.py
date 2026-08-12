"""
One switch to point the eval toolkit at the old (10^3) or new (15^3) model.

Flip OLD below; every eval script that imports from here follows. The grid size itself is not configured here — scripts read it from the loaded model via `model.im_x`, so the same code runs at either resolution.
"""

import os.path as osp

_repo = osp.dirname(osp.dirname(__file__))

OLD = False # True = old 10^3 model, False = new 15^3 model

if OLD:
    MODEL_PATH = osp.join(_repo, "Archive", "e1200_100", "Freq_FNO.pth")
    DATASET = osp.join(_repo, "dataset.npz")
    LATENT = osp.join(_repo, "Evaluation", "latent_data", "Freq_FNO_training_latent_data.txt")
    BASELINE = 0.04
else:
    MODEL_PATH = osp.join(_repo, "Archive", "d15e200", "Freq_FNO.pth")
    DATASET = osp.join(_repo, "dataset_15.npz")
    LATENT = osp.join(_repo, "Evaluation", "latent_data", "Freq_FNO_15_training_latent_data.txt")
    BASELINE = 0.13
