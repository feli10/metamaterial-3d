import { useLayoutEffect, useMemo, useRef } from "react";
import { Canvas } from "@react-three/fiber";
import { Line, OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { GRID_DIM } from "../api/types";
import "./VoxelPreview.css";

/**
 * VoxelPreview — renders the generated 10x10x10 geometry as green cubes.
 *
 * The 1000 voxels arrive flat with X FASTEST (see the voxel note in types.ts), so we decode
 *   x = i % 10,  y = (i / 10) % 10,  z = i / 100
 * and draw a cube for each solid (==1) voxel. All solid cubes share ONE instanced mesh, so
 * three.js draws them in a single GPU call — fast even though there can be hundreds of them.
 *
 * Drag to rotate, scroll to zoom (OrbitControls).
 */

interface Props {
  voxels: number[] | null;
}

const CENTER = (GRID_DIM - 1) / 2; // shift the grid so its center sits at the origin
const CUBE_SIZE = 1.0; // full size: adjacent voxels touch so struts stay connected (it's a
// metamaterial). Separation between voxels comes from the edge outlines, not gaps.
const EDGE_SIZE = CUBE_SIZE * 1.015; // edges drawn from a slightly LARGER box so the lines sit
// just outside the green faces — avoids z-fighting (the flicker) with coincident surfaces.
const EDGE_WIDTH = 1; // edge thickness in pixels — adjust to taste (fat lines, so it works)
const EDGE_COLOR = "#0d193a"; // near-black green

/** The instanced cubes + merged edge outlines for all solid voxels. */
function Voxels({ voxels }: { voxels: number[] }) {
  const meshRef = useRef<THREE.InstancedMesh>(null);

  // World positions of every solid voxel, centered on the origin.
  const positions = useMemo(() => {
    const p: Array<[number, number, number]> = [];
    for (let i = 0; i < voxels.length; i++) {
      if (voxels[i]) {
        const x = i % GRID_DIM;
        const y = Math.floor(i / GRID_DIM) % GRID_DIM;
        const z = Math.floor(i / (GRID_DIM * GRID_DIM));
        p.push([x - CENTER, y - CENTER, z - CENTER]);
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

  return (
    <div className="preview-box preview-box--canvas">
      <Canvas camera={{ position: [14, 11, 16], fov: 40 }}>
        <ambientLight intensity={0.55} />
        <directionalLight position={[10, 15, 10]} intensity={1.2} />
        <directionalLight position={[-8, -4, -10]} intensity={0.5} />
        <Voxels voxels={voxels} />
        <OrbitControls enablePan={false} />
      </Canvas>
    </div>
  );
}
