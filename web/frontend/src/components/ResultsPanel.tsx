import { RC_TO_CHANNEL, type PropertyVector } from "../api/types";
import "./ResultsPanel.css";

/**
 * ResultsPanel — the dark "Achieved Material Properties" readout.
 *
 * It MIRRORS the input matrix's 6x6 layout so each achieved cell lines up with its target
 * cell by position. It's a readout (no inputs): monospace values, upper triangle shows the
 * measured C_ij, lower triangle shows the "= C_ji" symmetry relationship. Each SPECIFIED
 * channel's value is colored by error magnitude: green (low) → yellow (medium) → red (high).
 * Auto/unspecified channels have no target to compare to, so they're shown white. vf and
 * mean-absolute-error (over specified channels) sit in the footer.
 *
 * Before Evaluate runs, `achieved` is null and we show a pending message (never all-zeros —
 * that would read as a failed design).
 */

interface Props {
  achieved: PropertyVector | null;
  target: PropertyVector | null; // filledTarget from generate (all 22 concrete) for comparison
  autoMask: boolean[] | null; // true where the channel was auto-filled (user left it blank)
}

// Error magnitude at which the color saturates to red. ~5% of the ~0–1 value range.
const ERR_SCALE = 0.05;
// Sequential error scale by MAGNITUDE: green (low) … yellow (medium) … red (high).
const GREEN: RGB = [29, 158, 117]; // #1D9E75 — low error (on target)
const YELLOW: RGB = [245, 158, 11]; // #F59E0B — medium error
const RED: RGB = [220, 38, 38]; // #DC2626 — high error
const AUTO_COLOR = "#ffffff"; // unspecified (auto) channel: no target to compare → white

type RGB = [number, number, number];

function lerp(a: RGB, b: RGB, t: number): string {
  const c = [0, 1, 2].map((i) => Math.round(a[i] + (b[i] - a[i]) * t));
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}

/** Color for an error magnitude: green (0) → yellow (½ scale) → red (full scale). */
function errorColor(absErr: number): string {
  const t = Math.min(1, absErr / ERR_SCALE);
  return t < 0.5 ? lerp(GREEN, YELLOW, t / 0.5) : lerp(YELLOW, RED, (t - 0.5) / 0.5);
}

/** Hover text for a channel: absolute error vs. target, or "auto" when unspecified. */
function tooltipFor(achieved: number, target: number, auto: boolean): string {
  return auto
    ? "no target set"
    : `target ${target.toFixed(3)} | error ${(achieved - target).toFixed(3)}`;
}

/**
 * One measured value, colored green→yellow→red by error magnitude (white when auto). The
 * tooltip lives on the surrounding cell, so hovering the Cᵢⱼ label triggers it too.
 */
function ResValue({
  achieved,
  target,
  auto,
}: {
  achieved: number;
  target: number;
  auto: boolean;
}) {
  const color = auto ? AUTO_COLOR : errorColor(Math.abs(achieved - target));
  return (
    <span className="res-value" style={{ color }}>
      {achieved.toFixed(3)}
    </span>
  );
}

/** "C" + a 1-based subscript for the given 0-based row/col. */
function label(row: number, col: number) {
  return (
    <>
      C
      <sub>
        {row + 1}
        {col + 1}
      </sub>
    </>
  );
}

export default function ResultsPanel({ achieved, target, autoMask }: Props) {
  if (!achieved || !target) {
    return <p className="placeholder">Run Evaluate to see the measured properties.</p>;
  }

  const rows = [0, 1, 2, 3, 4, 5];

  // Mean absolute error over the SPECIFIED channels only (auto channels have no user target).
  const specified = target
    .map((_, ch) => ch)
    .filter((ch) => !(autoMask?.[ch] ?? false));
  const mae = specified.length
    ? specified.reduce((sum, ch) => sum + Math.abs(achieved[ch] - target[ch]), 0) /
      specified.length
    : 0;

  return (
    <div>
      <div className="results-matrix">
        {rows.map((row) =>
          rows.map((col) => {
            const ch = RC_TO_CHANNEL[row][col];
            const isUpper = col >= row;
            const auto = autoMask?.[ch] ?? false;
            return (
              <div
                className="res-cell"
                key={`${row}-${col}`}
                data-tooltip={
                  isUpper ? tooltipFor(achieved[ch], target[ch], auto) : undefined
                }
              >
                <span className="res-label">{label(row, col)}</span>
                {isUpper ? (
                  <ResValue achieved={achieved[ch]} target={target[ch]} auto={auto} />
                ) : (
                  <span className="res-relation">= {label(col, row)}</span>
                )}
              </div>
            );
          }),
        )}
      </div>

      <div className="results-footer">
        <span
          className="res-vf"
          data-tooltip={tooltipFor(achieved[0], target[0], autoMask?.[0] ?? false)}
        >
          vf ={" "}
          <ResValue
            achieved={achieved[0]}
            target={target[0]}
            auto={autoMask?.[0] ?? false}
          />
        </span>
        <span className="res-mae">
          mean absolute error ={" "}
          <span style={{ color: errorColor(mae) }}>{mae.toFixed(4)}</span>
        </span>
      </div>
    </div>
  );
}
