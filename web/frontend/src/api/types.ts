/**
 * types.ts — THE API CONTRACT.
 *
 * This file is the single source of truth for the data that flows between the UI
 * and the model. The UI is built against these types; `client.ts` (the mock now,
 * a real `fetch()` later) is built to satisfy them. If the shapes here are right,
 * swapping the mock for the real backend changes only `client.ts` — nothing else.
 *
 * A note on TypeScript, since it's new to you: `type X = ...` and `interface X {...}`
 * describe the *shape* of data — they don't produce any runtime code, they just let
 * the compiler catch mistakes (e.g. calling generate() without a `model`). At runtime
 * these are plain JS objects, arrays, and numbers.
 *
 * Domain reference: this mirrors `Evaluation/interactive_generation.py`, the reference
 * implementation. See CLAUDE.md ("API contract" + "Domain facts") for the why.
 */

/* ------------------------------------------------------------------ *
 * The 22-channel property vector
 * ------------------------------------------------------------------ *
 * Every property vector — target, filledTarget, achieved — is 22 numbers:
 *
 *   channel 0        = vf (volume fraction)
 *   channels 1..21   = the 21 independent entries of the 6x6 stiffness tensor C,
 *                      in numpy's `np.triu_indices(6)` order (upper triangle,
 *                      row by row): C11 C12 C13 C14 C15 C16 C22 C23 ... C66.
 *
 * The 6x6 C is symmetric, so only the 21 upper-triangle entries are independent;
 * that's why 21, not 36. Voigt index order 1..6 = xx, yy, zz, yz, xz, xy.
 */
export const CHANNEL_COUNT = 22;

/** The voxel grid is 10x10x10 = 1000 cells. */
export const GRID_DIM = 10;
export const VOXEL_COUNT = GRID_DIM * GRID_DIM * GRID_DIM; // 1000

/**
 * Human-readable label for each of the 22 channels: ["vf", "C11", "C12", ..., "C66"].
 * Built from np.triu_indices(6) so it can never drift out of sync with the ordering.
 */
export const CHANNEL_LABELS: string[] = buildChannelLabels();

/**
 * For each stiffness channel (1..21), its position in the 6x6 matrix as [row, col],
 * 0-based. Channel 0 (vf) is not in the matrix, so it maps to null. Lets a matrix
 * component ask "which channel is cell (row, col)?" and vice-versa.
 */
export const CHANNEL_TO_RC: (readonly [number, number] | null)[] = buildChannelToRC();

/**
 * The inverse of CHANNEL_TO_RC: a 6x6 table where RC_TO_CHANNEL[row][col] is the channel
 * index for that matrix cell (0-based row/col). Because C is symmetric, cell (row,col) and
 * (col,row) map to the SAME channel — so the lower triangle mirrors the upper. Lets the
 * matrix component ask "what channel is this cell?" for any of the 36 positions.
 */
export const RC_TO_CHANNEL: number[][] = buildRcToChannel();

/**
 * ALL 22 channels are user-settable inputs. The user may specify ANY subset (at least
 * one); every channel they leave blank becomes `null` = auto and is filled by the model
 * from the nearest training cell. There is no fixed "exposed" set — the chosen subset is
 * simply the non-null entries of the target vector.
 *
 * (The matplotlib reference `interactive_generation.py` only exposed 7 sliders — vf + the
 * diagonals — but that was a demo limitation, not a constraint of the model. This web tool
 * generalizes to any subset. NOTE for backend time: the reference `generate()` hardcodes
 * that 7-channel EXPOSED set; wiring the real backend will require generalizing it so the
 * "known" channels are computed as the non-null entries of `target`.)
 *
 * `DIAGONAL_CHANNELS` (vf + the six diagonal stiffnesses) are called out only as the
 * physically most common / "simple" targets — useful for optional visual emphasis in the
 * matrix. This is a UI hint, NOT a restriction on what can be set.
 */
export const DIAGONAL_CHANNELS: readonly number[] = [0, 1, 7, 12, 16, 19, 21];

/* ------------------------------------------------------------------ *
 * Request / response shapes
 * ------------------------------------------------------------------ */

/**
 * A single target channel value.
 *   number  -> the user (or a filled default) dialed in this value.
 *   null    -> "auto": leave it to the model to fill from the nearest training cell.
 *
 * `null = auto` applies ONLY to editable target channels. It is never used for the
 * lower triangle of the matrix (those are determined by symmetry, not "auto") and
 * never appears in `filledTarget` or `achieved` (those are always concrete numbers).
 */
export type PropertyValue = number | null;

/**
 * A target property vector: 22 entries, each a number or null (auto). The user must set
 * at least one entry (all-null is not a valid target). The non-null entries are the
 * user's chosen subset; the null entries are auto-filled by the model.
 */
export type TargetVector = PropertyValue[]; // length === CHANNEL_COUNT

/** A fully-concrete property vector: 22 numbers, no nulls. */
export type PropertyVector = number[]; // length === CHANNEL_COUNT

/** An entry in the model dropdown. `id` maps to a real checkpoint dir in the backend. */
export interface ModelInfo {
  id: string;    // e.g. "s10v1" (UI name) -> backend maps to Archive/e1200_100
  label: string; // what the dropdown shows
}

/** POST /generate request body. */
export interface GenerateRequest {
  target: TargetVector; // 22 entries; null entries = auto
  model: string;        // ModelInfo.id
}

/** POST /generate response. */
export interface GenerateResponse {
  /**
   * The generated geometry: 1000 values, each 0 or 1, flattened from the 10x10x10
   * grid with X FASTEST — flat index i decodes as
   *   x = i % 10,  y = (i / 10) % 10,  z = i / 100
   * Matching numpy's ordering here keeps the 3D preview from coming out mirrored.
   */
  voxels: number[];
  /**
   * The target with every auto channel filled in — all 22 concrete. The results
   * panel needs this so auto-filled channels still have a target to compare against.
   */
  filledTarget: PropertyVector;
  /** Confidence / familiarity, fa = exp(-(eps_e - 0.04)). Higher is better (~0.99+). */
  fa: number;
  /** Reconstruction error eps_e. Lower is better. */
  epsE: number;
}

/** POST /evaluate request body. */
export interface EvaluateRequest {
  voxels: number[]; // the geometry to homogenize (1000 flat 0/1)
}

/** POST /evaluate response. */
export interface EvaluateResponse {
  /**
   * The homogenized GROUND-TRUTH properties of the geometry (22-channel). This is
   * measured by FEM, not predicted — it may differ from the target you dialed in.
   */
  achieved: PropertyVector;
}

/**
 * Progress reported DURING generate() — one tick per Adam iteration (there are ~100).
 * Mirrors the `progress(i, fa)` callback in interactive_generation.py.
 */
export interface GenerateProgress {
  iter: number; // 1-based current iteration
  total: number; // total iterations (~100)
  fa: number; // current familiarity/confidence
}

/**
 * Progress reported DURING evaluate() — one tick per FEM unit-strain load case.
 * homogenize_3d solves `for load_case in range(6)`, so total is 6.
 */
export interface EvaluateProgress {
  loadCase: number; // 1-based current load case
  total: number; // total load cases (6)
}

/**
 * The API surface the UI talks to. The UI only ever calls these three functions and
 * never knows whether a mock or a real server is behind them.
 */
export interface MetamaterialApi {
  listModels(): Promise<ModelInfo[]>;
  generate(
    req: GenerateRequest,
    onProgress?: (p: GenerateProgress) => void,
  ): Promise<GenerateResponse>;
  evaluate(
    req: EvaluateRequest,
    onProgress?: (p: EvaluateProgress) => void,
  ): Promise<EvaluateResponse>;
}

/* ------------------------------------------------------------------ *
 * Helpers that build the channel-layout constants above.
 * Kept at the bottom so the exported constants read top-down.
 * ------------------------------------------------------------------ */

/** Upper-triangle (i, j) pairs of a 6x6, in np.triu_indices(6) order. 21 of them. */
function triuIndices6(): Array<[number, number]> {
  const pairs: Array<[number, number]> = [];
  for (let i = 0; i < 6; i++) {
    for (let j = i; j < 6; j++) {
      pairs.push([i, j]);
    }
  }
  return pairs;
}

function buildChannelLabels(): string[] {
  const labels = ["vf"];
  for (const [i, j] of triuIndices6()) {
    labels.push(`C${i + 1}${j + 1}`);
  }
  return labels; // 1 + 21 = 22
}

function buildChannelToRC(): (readonly [number, number] | null)[] {
  const map: (readonly [number, number] | null)[] = [null]; // channel 0 = vf, not in matrix
  for (const [i, j] of triuIndices6()) {
    map.push([i, j]);
  }
  return map;
}

function buildRcToChannel(): number[][] {
  const m: number[][] = Array.from({ length: 6 }, () => new Array(6).fill(-1));
  CHANNEL_TO_RC.forEach((rc, ch) => {
    if (rc) {
      const [i, j] = rc;
      m[i][j] = ch;
      m[j][i] = ch; // symmetry: lower triangle points at the same channel
    }
  });
  return m;
}
