import { useEffect, useRef, useState } from "react";
import StiffnessMatrix from "./components/StiffnessMatrix";
import VolumeFraction from "./components/VolumeFraction";
import VoxelPreview from "./components/VoxelPreview";
import ResultsPanel from "./components/ResultsPanel";
import { api } from "./api/client";
import {
  CHANNEL_COUNT,
  type GenerateResponse,
  type ModelInfo,
  type PropertyValue,
  type PropertyVector,
  type TargetVector,
} from "./api/types";
import {
  buildSavedDesign,
  downloadSavedDesign,
  parseSavedDesign,
} from "./lib/savedDesign";
import "./App.css";

/**
 * App — the top-level layout + the generate/evaluate flow.
 *
 * Layout (from CLAUDE.md):
 *   LEFT  = action/output spine: model -> Generate/Evaluate -> 3D preview -> confidence -> Save/Load
 *   RIGHT = inputs on top (Stiffness Matrix + Volume Fraction), results readout on the bottom
 */
function App() {
  // The one authoritative target vector (22 channels, null = auto). The matrix edits
  // channels 1..21; the volume-fraction input edits channel 0. Generate reads this.
  const [target, setTarget] = useState<TargetVector>(() =>
    new Array(CHANNEL_COUNT).fill(null),
  );

  // Available models (from the backend) + the currently selected one. Fetched once on mount.
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [modelId, setModelId] = useState<string>("");

  useEffect(() => {
    api
      .listModels()
      .then((list) => {
        setModels(list);
        if (list.length > 0) setModelId(list[0].id);
      })
      .catch((err) => setStatus(`Could not load models: ${(err as Error).message}`));
  }, []);

  // Which operation is running (disables buttons, drives the status message). null = idle.
  const [busy, setBusy] = useState<"generating" | "evaluating" | null>(null);
  const [status, setStatus] = useState("");

  // Last generated design (voxels + filledTarget + fa/epsE) and last measured properties.
  const [generated, setGenerated] = useState<GenerateResponse | null>(null);
  const [achieved, setAchieved] = useState<PropertyVector | null>(null);
  // The raw target at the moment Generate ran — snapshotted so the results panel knows which
  // channels were auto-filled (null) even if the user edits the inputs afterward.
  const [submittedTarget, setSubmittedTarget] = useState<TargetVector | null>(null);

  // Update a single channel immutably (React needs a new array to detect the change).
  function setChannel(channel: number, value: PropertyValue) {
    setTarget((prev) => {
      const next = [...prev];
      next[channel] = value;
      return next;
    });
  }

  // Hidden <input type="file"> we click programmatically for Load.
  const fileInputRef = useRef<HTMLInputElement>(null);

  // At least one channel must be set (CLAUDE.md: all-null is not a valid target).
  const hasTarget = target.some((v) => v !== null);

  async function handleGenerate() {
    if (busy || !hasTarget || !modelId) return;
    setBusy("generating");
    setAchieved(null); // a new geometry invalidates any previous measurement
    setStatus("Generating… (this runs the model, ~10s)");
    const submitted = [...target]; // snapshot the target used for this run
    setSubmittedTarget(submitted);
    try {
      const result = await api.generate({ target: submitted, model: modelId }, (p) =>
        setStatus(`Generating… ${p.iter}/${p.total}  fₐ=${p.fa.toFixed(4)}`),
      );
      setGenerated(result);
      setStatus("");
    } catch {
      setStatus("Generation failed.");
    } finally {
      setBusy(null);
    }
  }

  async function handleEvaluate() {
    if (busy || !generated) return;
    setBusy("evaluating");
    setStatus("Evaluating… (FEM homogenization, ~8s)");
    try {
      const result = await api.evaluate({ voxels: generated.voxels }, (p) =>
        setStatus(`Evaluating… load case ${p.loadCase}/${p.total}`),
      );
      setAchieved(result.achieved);
      setStatus("");
    } catch {
      setStatus("Evaluation failed.");
    } finally {
      setBusy(null);
    }
  }

  // Save the current design (inputs + geometry + achieved + model) as a JSON file.
  function handleSave() {
    if (!generated) return;
    const design = buildSavedDesign({
      model: modelId,
      target: submittedTarget ?? target,
      generated,
      achieved,
    });
    downloadSavedDesign(design);
  }

  // Restore a saved design from a JSON file, repopulating inputs, geometry, and results.
  async function handleFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    e.target.value = ""; // reset so the same file can be chosen again
    if (!file) return;
    try {
      const design = parseSavedDesign(await file.text());
      setTarget(design.target);
      setSubmittedTarget(design.target);
      // Only restore a generated design if the file has the full generate() output.
      const canRestore =
        design.voxels != null &&
        design.filledTarget != null &&
        design.fa != null &&
        design.epsE != null;
      setGenerated(
        canRestore
          ? {
              voxels: design.voxels!,
              filledTarget: design.filledTarget!,
              fa: design.fa!,
              epsE: design.epsE!,
            }
          : null,
      );
      setAchieved(canRestore ? design.achieved : null);
      setStatus("");
    } catch (err) {
      setStatus(`Load failed: ${(err as Error).message}`);
    }
  }

  return (
    <div className="app">
      {/* ---------------- LEFT: action / output spine ---------------- */}
      <div className="col-left">
        <div className="model-chip">
          <span className="dot" />
          <label htmlFor="model-select">Model</label>
          <select
            id="model-select"
            value={modelId}
            onChange={(e) => setModelId(e.target.value)}
            disabled={busy !== null || models.length === 0}
          >
            {models.length === 0 ? (
              <option value="">loading…</option>
            ) : (
              models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label}
                </option>
              ))
            )}
          </select>
        </div>

        <div className="button-row">
          <button
            type="button"
            className="btn btn-primary"
            onClick={handleGenerate}
            disabled={busy !== null || !hasTarget}
          >
            Generate
          </button>
          <button
            type="button"
            className="btn btn-secondary"
            onClick={handleEvaluate}
            disabled={busy !== null || !generated}
          >
            Evaluate
          </button>
        </div>

        {/* progress / status message shows here on Generate / Evaluate */}
        <div className="status-line">{status}</div>

        {/* 3D voxel preview */}
        <VoxelPreview voxels={generated ? generated.voxels : null} />

        <div className="confidence-line">
          <span>
            Confidence (f<sub>a</sub>)
          </span>
          <span className="value">
            {generated ? generated.fa.toFixed(3) : "—"}
          </span>
        </div>

        <div className="button-row">
          <button
            type="button"
            className="btn btn-ghost"
            onClick={handleSave}
            disabled={!generated}
          >
            Save
          </button>
          <button
            type="button"
            className="btn btn-ghost"
            onClick={() => fileInputRef.current?.click()}
          >
            Load
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept="application/json,.json"
            onChange={handleFile}
            style={{ display: "none" }}
          />
        </div>
      </div>

      {/* ---------------- RIGHT: one solid block — inputs (top) + results (bottom) --------- */}
      <div className="col-right">
        <div className="workpanel">
          {/* inputs */}
          <div className="workpanel-section">
            <h2>Stiffness Matrix (C)</h2>
            <StiffnessMatrix target={target} onChange={setChannel} />

            <h2>Volume Fraction (vf)</h2>
            <VolumeFraction value={target[0]} onChange={(v) => setChannel(0, v)} />
          </div>

          {/* results readout (dark, joined flush to the inputs above) */}
          <div className="workpanel-results">
            <h2>Achieved Material Properties</h2>
            <ResultsPanel
              achieved={achieved}
              target={generated ? generated.filledTarget : null}
              autoMask={submittedTarget ? submittedTarget.map((v) => v === null) : null}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
