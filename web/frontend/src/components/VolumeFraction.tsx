import type { PropertyValue } from "../api/types";
import NumberField from "./NumberField";
import "./VolumeFraction.css";

/**
 * VolumeFraction — the vf input (channel 0). A numeric box (blank = auto) linked to a slider.
 *
 * - The box is the source of "auto": empty box => null => model picks vf.
 * - The slider is a convenience; dragging it sets a concrete vf (leaving auto). When vf is auto
 *   the slider is shown muted at a neutral position so it doesn't imply a chosen value.
 *
 * NOTE: the slider range is a sensible default (0..1). CLAUDE.md wants per-channel ranges from
 * the dataset min/max; we'll set that once the backend can supply the dataset stats.
 */

interface Props {
  value: PropertyValue;
  onChange: (v: PropertyValue) => void;
}

const VF_MIN = 0;
const VF_MAX = 1;
const VF_STEP = 0.01;
const VF_NEUTRAL = 0.2; // where the slider sits while vf is auto

export default function VolumeFraction({ value, onChange }: Props) {
  const isAuto = value === null;
  const sliderValue = isAuto ? VF_NEUTRAL : value;

  return (
    <div className="vf-row">
      <span className="vf-label">vf</span>
      <NumberField className="vf-input" value={value} onCommit={onChange} />
      <input
        type="range"
        className={`vf-slider${isAuto ? " vf-slider--auto" : ""}`}
        min={VF_MIN}
        max={VF_MAX}
        step={VF_STEP}
        value={sliderValue}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  );
}
