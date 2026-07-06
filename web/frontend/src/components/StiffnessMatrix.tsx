import { RC_TO_CHANNEL, type PropertyValue, type TargetVector } from "../api/types";
import NumberField from "./NumberField";
import "./StiffnessMatrix.css";

/**
 * StiffnessMatrix — the 6x6 stiffness tensor C as an input grid.
 *
 * Domain rules (CLAUDE.md):
 *  - Upper triangle + diagonal are EDITABLE inputs. Blank = "auto" (null): the model fills it.
 *  - Lower triangle is NEVER an input and NEVER a number. It shows "= Cᵢⱼ" — the relationship
 *    to its mirror source in the upper triangle — because C is symmetric.
 *  - Voigt index 1..6 = xx, yy, zz, yz, xz, xy.
 *
 * This is a "controlled" component: it doesn't own the values. The parent (App) holds the
 * `target` vector and passes it down; when a cell changes, we call `onChange(channel, value)`
 * and the parent updates. That way Generate can read one authoritative target from App.
 */

interface Props {
  target: TargetVector;
  onChange: (channel: number, value: PropertyValue) => void;
}

/** Render "C" followed by a 1-based subscript like ₂₃ for the given 0-based row/col. */
function cellLabel(row: number, col: number) {
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

export default function StiffnessMatrix({ target, onChange }: Props) {
  const rows = [0, 1, 2, 3, 4, 5];

  return (
    <div className="matrix">
      {rows.map((row) =>
        rows.map((col) => {
          const channel = RC_TO_CHANNEL[row][col];
          const isUpper = col >= row; // upper triangle + diagonal => editable

          return (
            <div className="mx-cell" key={`${row}-${col}`}>
              <span className="mx-label">{cellLabel(row, col)}</span>
              {isUpper ? (
                <NumberField
                  className="mx-input"
                  value={target[channel]}
                  onCommit={(v) => onChange(channel, v)}
                />
              ) : (
                // lower triangle: show the mirror relationship, never a value
                <span className="mx-relation">= {cellLabel(col, row)}</span>
              )}
            </div>
          );
        }),
      )}
    </div>
  );
}
