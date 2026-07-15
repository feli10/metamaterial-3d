import sys
import torch
import models_epi_3d_10  # the preserved OLD (10^3) Model class


def load_old_checkpoint(path, map_location="cpu"):
    """Load a 10^3 checkpoint that was pickled against the OLD model class.

    `models_epi_3d.py` is now the NEW (15^3-capable) version, but these old pickles hardcode
    `models_epi_3d.Model`. Temporarily alias that name to the preserved old class for the load,
    then restore it so the real `models_epi_3d` stays importable afterward (e.g. if the same
    process also builds a new-code model).

    Remove all uses of this once we retire the old 10^3 checkpoints for the 15^3 model.
    """
    saved = sys.modules.get("models_epi_3d")
    sys.modules["models_epi_3d"] = models_epi_3d_10
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    finally:
        if saved is not None:
            sys.modules["models_epi_3d"] = saved
        else:
            sys.modules.pop("models_epi_3d", None)


def load_model(path, old=False, map_location="cpu"):
    """Load any checkpoint, old or new.

    old=True  -> a pre-15^3 pickle (aliases the preserved old class, see load_old_checkpoint).
    old=False -> a checkpoint trained with the current code (plain torch.load).

    Sets `model.device` to match `map_location` (the pickled value may be 'cuda').
    """
    if old:
        model = load_old_checkpoint(path, map_location)
    else:
        model = torch.load(path, map_location=map_location, weights_only=False)
    model.device = (
        map_location if isinstance(map_location, torch.device) else torch.device(map_location)
    )
    return model
