/**
 * client.ts — THE DATA LAYER (mock implementation).
 *
 * This is the "worker" that fulfills the contract in types.ts. The UI imports `api`
 * from here and calls `api.generate(...)`, `api.evaluate(...)`, `api.listModels()`.
 * It never knows whether the data is real or faked.
 *
 * RIGHT NOW: everything here is FAKE but plausible, so the whole UI runs and can be
 * screenshotted with NO Python backend. That's the poster deliverable.
 *
 * LATER: we rewrite the *bodies* of these functions to `fetch()` against the FastAPI
 * server. The function signatures and the types never change, so no other file changes.
 * Anywhere a value is invented, it's marked `MOCK:` so the real-backend swap is obvious.
 */

import {
  CHANNEL_COUNT,
  CHANNEL_TO_RC,
  GRID_DIM,
  VOXEL_COUNT,
  type EvaluateProgress,
  type EvaluateRequest,
  type EvaluateResponse,
  type GenerateProgress,
  type GenerateRequest,
  type GenerateResponse,
  type MetamaterialApi,
  type ModelInfo,
  type PropertyVector,
} from "./types";

const GEN_ITERS = 100; // matches ITERS in interactive_generation.py
const EVAL_LOAD_CASES = 6; // matches `for load_case in range(6)` in homogenize_3d

/* ------------------------------------------------------------------ *
 * Small utilities
 * ------------------------------------------------------------------ */

/** Pause for `ms` milliseconds — used to fake network / compute latency. */
function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * A tiny seeded pseudo-random generator (mulberry32). Given the same seed it produces
 * the same stream — so a given target always generates the same geometry. That keeps
 * "Generate" reproducible (Save stores the inputs; re-running gives the same design).
 */
function makeRng(seed: number): () => number {
  let s = seed >>> 0;
  return function () {
    s |= 0;
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Derive a stable integer seed from a property vector, so the same target is reproducible. */
function seedFromVector(v: readonly (number | null)[]): number {
  let h = 2166136261;
  v.forEach((x, i) => {
    const n = x === null ? -1 : x;
    h = Math.imul(h ^ (Math.round(n * 1000) + i * 7919), 16777619);
  });
  return h >>> 0;
}

/** Is this channel a diagonal stiffness (C11,C22,...C66) or vf? Off-diagonals return false. */
function isDiagonalChannel(ch: number): boolean {
  if (ch === 0) return true; // vf
  const rc = CHANNEL_TO_RC[ch];
  return rc !== null && rc[0] === rc[1];
}

/* ------------------------------------------------------------------ *
 * Mock generate()
 * ------------------------------------------------------------------ */

/**
 * MOCK: fill the user's null (auto) channels with plausible concrete values.
 * The real backend fills these from the nearest training cell; here we just pick
 * physically-sensible stand-ins (diagonals bigger, off-diagonals small) so the results
 * panel has something to show for auto channels.
 */
function fillTarget(target: readonly (number | null)[], rng: () => number): PropertyVector {
  return Array.from({ length: CHANNEL_COUNT }, (_, ch) => {
    const v = target[ch];
    if (v !== null && v !== undefined) return v; // user set it — keep as-is
    if (ch === 0) return 0.15 + rng() * 0.15; // vf ~ 0.15–0.30
    return isDiagonalChannel(ch)
      ? 0.1 + rng() * 0.2 // diagonal stiffness ~ 0.1–0.3
      : rng() * 0.05; // off-diagonal ~ 0.00–0.05
  });
}

/**
 * MOCK: build a 10x10x10 geometry that (a) looks like a real lattice for screenshots and
 * (b) has volume fraction equal to the target vf. We use a gyroid level-set — a smooth
 * triply-periodic surface common in real metamaterials — and pick the threshold that makes
 * exactly `vf` of the cells solid. Returned flat with X FASTEST (see types.ts voxel note).
 */
function mockVoxels(vf: number): number[] {
  const a = (2 * Math.PI) / GRID_DIM;
  const field: number[] = new Array(VOXEL_COUNT);

  for (let z = 0; z < GRID_DIM; z++) {
    for (let y = 0; y < GRID_DIM; y++) {
      for (let x = 0; x < GRID_DIM; x++) {
        // Gyroid: sin x cos y + sin y cos z + sin z cos x. |g| small => near the surface.
        const g =
          Math.sin(a * x) * Math.cos(a * y) +
          Math.sin(a * y) * Math.cos(a * z) +
          Math.sin(a * z) * Math.cos(a * x);
        field[x + GRID_DIM * y + GRID_DIM * GRID_DIM * z] = Math.abs(g);
      }
    }
  }

  // Choose the threshold so the fraction of solid cells is ~vf (exact up to ties in the
  // very-symmetric gyroid field, which is close enough for a mock preview).
  const clampedVf = Math.min(0.95, Math.max(0.02, vf));
  const solidCount = Math.max(1, Math.round(clampedVf * VOXEL_COUNT));
  const cutoff = [...field].sort((p, q) => p - q)[solidCount - 1];

  return field.map((val) => (val <= cutoff ? 1 : 0));
}

/* ------------------------------------------------------------------ *
 * The API implementation
 * ------------------------------------------------------------------ */

export const api: MetamaterialApi = {
  async listModels(): Promise<ModelInfo[]> {
    await delay(100);
    // MOCK: one model. Its id maps to Archive/e1200_100 in the real backend.
    // "s10v1" is Felix's display name (s10 = 10^3 grid, v1); see CLAUDE.md.
    return [{ id: "s10v1", label: "s10v1" }];
  },

  async generate(
    req: GenerateRequest,
    onProgress?: (p: GenerateProgress) => void,
  ): Promise<GenerateResponse> {
    const rng = makeRng(seedFromVector(req.target));

    // MOCK: a confident-looking familiarity score, and the eps_e that's consistent with it
    // via the real relationship fa = exp(-(eps_e - 0.04)).
    const fa = 0.985 + rng() * 0.012; // ~0.985–0.997
    const epsE = 0.04 - Math.log(fa);

    // MOCK: stand in for the ~100-iter Adam optimization, reporting progress each step with
    // fa ramping up toward its final value (the real optimizer maximizes fa over iterations).
    for (let i = 1; i <= GEN_ITERS; i++) {
      await delay(20);
      onProgress?.({ iter: i, total: GEN_ITERS, fa: fa * (0.9 + 0.1 * (i / GEN_ITERS)) });
    }

    const filledTarget = fillTarget(req.target, rng);
    const voxels = mockVoxels(filledTarget[0]); // filledTarget[0] = vf

    return { voxels, filledTarget, fa, epsE };
  },

  async evaluate(
    req: EvaluateRequest,
    onProgress?: (p: EvaluateProgress) => void,
  ): Promise<EvaluateResponse> {
    // MOCK: stand in for FEM homogenization's 6 unit-strain load cases.
    for (let k = 1; k <= EVAL_LOAD_CASES; k++) {
      await delay(200);
      onProgress?.({ loadCase: k, total: EVAL_LOAD_CASES });
    }

    // The one channel we can compute honestly from the geometry: the real volume fraction.
    const vf =
      req.voxels.reduce((sum, v) => sum + v, 0) / (req.voxels.length || 1);

    // MOCK: fake but plausible stiffnesses derived from vf alone (real values come from
    // homogenize_3d later). Denser cell => stiffer; diagonals dominate; small deterministic
    // noise so "achieved" differs slightly from "target" (which is the whole point of the
    // target-vs-achieved comparison). Seeded from the geometry so evaluate is reproducible.
    const rng = makeRng(seedFromVector(req.voxels));
    const stiffness = Math.pow(vf, 1.5); // rough monotonic scaling

    const achieved: PropertyVector = Array.from({ length: CHANNEL_COUNT }, (_, ch) => {
      if (ch === 0) return vf;
      const rc = CHANNEL_TO_RC[ch]!;
      const isDiag = rc[0] === rc[1];
      const isShear = isDiag && rc[0] >= 3; // C44,C55,C66
      const base = isDiag ? (isShear ? 0.4 : 1.0) * stiffness : 0.15 * stiffness;
      const noise = (rng() - 0.5) * 0.02;
      return Math.max(0, base + noise);
    });

    return { achieved };
  },
};
