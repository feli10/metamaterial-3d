/**
 * savedDesign.ts — the Save/Load file format + (de)serialization.
 *
 * FORMAT: JSON. Chosen because a saved design is best kept human-readable, self-describing,
 * and easy to reload in the research code (json -> numpy). It bundles everything needed to
 * REPRODUCE and RESTORE a design:
 *   - `target`      the user's inputs (null = auto) — with `model`, this reproduces generate()
 *   - `voxels`      the generated geometry, so it reloads without regenerating
 *   - `filledTarget`, `fa`, `epsE`   the generate() outputs
 *   - `achieved`    the FEM-measured properties (if Evaluate was run)
 * Plus `format`/`version`/`savedAt` metadata and the channel layout so the file stands alone.
 *
 * Note: generate() is deterministic given (target, model), so (target + model) IS the
 * reproducibility key — there's no separate random seed to store.
 */

import {
  CHANNEL_COUNT,
  CHANNEL_LABELS,
  gridDimForVoxelCount,
  type GenerateResponse,
  type PropertyVector,
  type TargetVector,
} from "../api/types";

export const SAVE_FORMAT = "metamaterial-design";
export const SAVE_VERSION = 1;

export interface SavedDesign {
  format: string;
  version: number;
  savedAt: string; // ISO timestamp
  model: string;
  grid: [number, number, number];
  channelLabels: string[]; // ["vf","C11",...,"C66"] — documents the vector layout
  target: TargetVector; // 22 entries, null = auto
  filledTarget: PropertyVector | null; // 22 concrete, or null if never generated
  voxels: number[] | null; // 1000 flat 0/1, or null
  fa: number | null;
  epsE: number | null;
  achieved: PropertyVector | null; // 22 measured, or null if never evaluated
}

/** Bundle the current app state into a SavedDesign. */
export function buildSavedDesign(p: {
  model: string;
  target: TargetVector;
  generated: GenerateResponse | null;
  achieved: PropertyVector | null;
}): SavedDesign {
  // Grid size comes from the generated geometry (N = ∛length). Save is only reachable once a
  // design has been generated, so voxels are always present here; default to 0 defensively.
  const n = p.generated ? gridDimForVoxelCount(p.generated.voxels.length) : 0;
  return {
    format: SAVE_FORMAT,
    version: SAVE_VERSION,
    savedAt: new Date().toISOString(),
    model: p.model,
    grid: [n, n, n],
    channelLabels: CHANNEL_LABELS,
    target: p.target,
    filledTarget: p.generated?.filledTarget ?? null,
    voxels: p.generated?.voxels ?? null,
    fa: p.generated?.fa ?? null,
    epsE: p.generated?.epsE ?? null,
    achieved: p.achieved,
  };
}

/** Serialize to a downloadable, pretty-printed JSON file (triggers a browser download). */
export function downloadSavedDesign(design: SavedDesign): void {
  const json = JSON.stringify(design, null, 2);
  const blob = new Blob([json], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `metamaterial-design-${design.savedAt.replace(/[:.]/g, "-")}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

/** Parse + validate a loaded file. Throws a human-readable Error if it's not a valid design. */
export function parseSavedDesign(text: string): SavedDesign {
  let data: SavedDesign;
  try {
    data = JSON.parse(text) as SavedDesign;
  } catch {
    throw new Error("not valid JSON");
  }
  if (data.format !== SAVE_FORMAT) {
    throw new Error("not a metamaterial design file");
  }
  if (!Array.isArray(data.target) || data.target.length !== CHANNEL_COUNT) {
    throw new Error(`target must have ${CHANNEL_COUNT} channels`);
  }
  if (data.voxels != null) {
    const n = gridDimForVoxelCount(data.voxels.length);
    if (!Array.isArray(data.voxels) || n ** 3 !== data.voxels.length) {
      throw new Error("voxels length must be a perfect cube (N³)");
    }
  }
  if (
    data.filledTarget != null &&
    (!Array.isArray(data.filledTarget) || data.filledTarget.length !== CHANNEL_COUNT)
  ) {
    throw new Error(`filledTarget must have ${CHANNEL_COUNT} channels`);
  }
  if (
    data.achieved != null &&
    (!Array.isArray(data.achieved) || data.achieved.length !== CHANNEL_COUNT)
  ) {
    throw new Error(`achieved must have ${CHANNEL_COUNT} channels`);
  }
  return data;
}
