import { useLayoutEffect, useMemo, useRef } from "react";
import { Canvas } from "@react-three/fiber";
import { Line, OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { gridDimForVoxelCount } from "../api/types";
import "./VoxelPreview.css";

/**
 * VoxelPreview — renders the generated N×N×N geometry as cubes.
 *
 * The N³ voxels arrive flat with X FASTEST (see the voxel note in types.ts), so we decode
 *   x = i % N,  y = (i / N) % N,  z = i / N²
 * where N is recovered from the array length (∛length) — so the preview adapts to any model
 * resolution (10³, 15³, …) with no hardcoded grid size. We draw a cube for each solid (==1)
 * voxel; all solid cubes share ONE instanced mesh, so three.js draws them in a single GPU call.
 *
 * Drag to rotate, scroll to zoom (OrbitControls).
 */

interface Props {
  voxels: number[] | null;
}

const CUBE_SIZE = 1.0; // full size: adjacent voxels touch so struts stay connected (it's a
// metamaterial). Separation between voxels comes from the edge outlines, not gaps.
const EDGE_SIZE = CUBE_SIZE * 1.015; // edges drawn from a slightly LARGER box so the lines sit
// just outside the green faces — avoids z-fighting (the flicker) with coincident surfaces.
const EDGE_WIDTH = 1; // edge thickness in pixels — adjust to taste (fat lines, so it works)
const EDGE_COLOR = "#0d193a"; // near-black green

/** The instanced cubes + merged edge outlines for all solid voxels. */
function Voxels({ voxels }: { voxels: number[] }) {
  const meshRef = useRef<THREE.InstancedMesh>(null);

  // World positions of every solid voxel, centered on the origin. Grid size N and the center
  // offset are derived from the array length, so this works for any resolution.
  const positions = useMemo(() => {
    const gridDim = gridDimForVoxelCount(voxels.length);
    const center = (gridDim - 1) / 2;
    const p: Array<[number, number, number]> = [];
    for (let i = 0; i < voxels.length; i++) {
      if (voxels[i]) {
        const x = i % gridDim;
        const y = Math.floor(i / gridDim) % gridDim;
        const z = Math.floor(i / (gridDim * gridDim));
        p.push([x - center, y - center, z - center]);
      }
    }
    return p;
  }, [voxels]);

  // Write each cube's transform into the instanced mesh's matrix buffer.
  useLayoutEffect(() => {
    const mesh = meshRef.current;
    if (!mesh) return;
    const dummy = new THREE.Object3D();
    positions.forEach((pos, idx) => {
      dummy.position.set(pos[0], pos[1], pos[2]);
      dummy.updateMatrix();
      mesh.setMatrixAt(idx, dummy.matrix);
    });
    mesh.count = positions.length;
    mesh.instanceMatrix.needsUpdate = true;
  }, [positions]);

  // Edge outlines as fat lines: take one cube's 12 edges (via EdgesGeometry) and copy them,
  // translated, for every solid voxel. drei's <Line segments> draws them as screen-space quads
  // (so lineWidth actually works). Points are consecutive segment-endpoint pairs.
  const edgePoints = useMemo(() => {
    const box = new THREE.BoxGeometry(EDGE_SIZE, EDGE_SIZE, EDGE_SIZE);
    const edges = new THREE.EdgesGeometry(box);
    const unit = edges.attributes.position.array; // line-segment vertices for one cube
    const pts: Array<[number, number, number]> = [];
    positions.forEach(([px, py, pz]) => {
      for (let k = 0; k < unit.length; k += 3) {
        pts.push([unit[k] + px, unit[k + 1] + py, unit[k + 2] + pz]);
      }
    });
    box.dispose();
    edges.dispose();
    return pts;
  }, [positions]);

  return (
    <>
      {/* key forces a fresh instance buffer when the solid-count changes */}
      <instancedMesh
        ref={meshRef}
        key={positions.length}
        args={[undefined, undefined, Math.max(1, positions.length)]}
      >
        <boxGeometry args={[CUBE_SIZE, CUBE_SIZE, CUBE_SIZE]} />
        <meshStandardMaterial color="#5d94d7" flatShading roughness={0.6} />
      </instancedMesh>
      {edgePoints.length > 0 && (
        <Line points={edgePoints} segments color={EDGE_COLOR} lineWidth={EDGE_WIDTH} />
      )}
    </>
  );
}

export default function VoxelPreview({ voxels }: Props) {
  if (!voxels) {
    return <div className="preview-box">3D preview</div>;
  }

  // Pull the camera back proportionally to the grid size so a 15³ cell frames the same as a
  // 10³ one (the base position [14,11,16] was tuned for a 10³ grid).
  const scale = gridDimForVoxelCount(voxels.length) / 10;
  const cam: [number, number, number] = [14 * scale, 11 * scale, 16 * scale];

  return (
    <div className="preview-box preview-box--canvas">
      <Canvas camera={{ position: cam, fov: 40 }}>
        <ambientLight intensity={0.55} />
        <directionalLight position={[10, 15, 10]} intensity={1.2} />
        <directionalLight position={[-8, -4, -10]} intensity={0.5} />
        <Voxels voxels={voxels} />
        <OrbitControls enablePan={false} />
      </Canvas>
    </div>
  );
}
