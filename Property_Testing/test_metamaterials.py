#!/usr/bin/env python3
"""
Test 3D FEM homogenization on various metamaterial structures.

This script demonstrates the solver's capability on:
1. Checkerboard patterns (void-solid alternation)
2. Graded density fields (Young's modulus variation)
3. Layered composites (horizontal/vertical alternation)
4. Octet truss-like structures (diagonal connectivity)
"""

import numpy as np
import sys
sys.path.insert(0, '/sessions/elegant-epic-sagan/mnt/3D_MultiLatticeTO')

from homopy_fem_3d import homogenize_3d, IsotropicMaterial


def create_checkerboard(nelx, nely, nelz, threshold=0.5):
    """Create a 3D checkerboard pattern (alternating void-solid)."""
    density = np.zeros((nelz, nely, nelx), dtype=np.float64)
    for i in range(nelx):
        for j in range(nely):
            for k in range(nelz):
                # Checkerboard pattern: alternating based on coordinate sum
                if (i + j + k) % 2 == 0:
                    density[k, j, i] = 1.0
                else:
                    density[k, j, i] = 0.01  # Hollow (very soft)
    return density


def create_layered_composite(nelx, nely, nelz, direction='z', period=2):
    """Create a layered composite with alternating stiff-soft layers."""
    density = np.zeros((nelz, nely, nelx), dtype=np.float64)
    
    if direction == 'z':
        # Horizontal layers (constant in z)
        for k in range(nelz):
            layer = (k // period) % 2
            density[k, :, :] = 1.0 if layer == 0 else 0.1
    elif direction == 'y':
        # Layers perpendicular to y
        for j in range(nely):
            layer = (j // period) % 2
            density[:, j, :] = 1.0 if layer == 0 else 0.1
    elif direction == 'x':
        # Layers perpendicular to x
        for i in range(nelx):
            layer = (i // period) % 2
            density[:, :, i] = 1.0 if layer == 0 else 0.1
    
    return density


def create_graded_density(nelx, nely, nelz, gradient_dir='z'):
    """Create a density field with continuous gradient."""
    density = np.zeros((nelz, nely, nelx), dtype=np.float64)
    
    if gradient_dir == 'z':
        for k in range(nelz):
            density[k, :, :] = 0.2 + 0.8 * (k / max(1, nelz - 1))
    elif gradient_dir == 'x':
        for i in range(nelx):
            density[:, :, i] = 0.2 + 0.8 * (i / max(1, nelx - 1))
    elif gradient_dir == 'y':
        for j in range(nely):
            density[:, j, :] = 0.2 + 0.8 * (j / max(1, nely - 1))
    
    return density


def create_octet_truss_like(nelx, nely, nelz):
    """Create a sparse truss-like structure (simplified octet)."""
    density = np.zeros((nelz, nely, nelx), dtype=np.float64)
    
    # Only keep material at edges and corners
    for i in range(nelx):
        for j in range(nely):
            for k in range(nelz):
                # Keep material at:
                # - Edges of the domain
                # - Interior on a diagonal pattern
                on_edge = (i == 0 or i == nelx-1 or 
                          j == 0 or j == nely-1 or 
                          k == 0 or k == nelz-1)
                
                # Diagonal connectivity pattern
                interior_pattern = ((i + j + k) % 3 == 0) and (
                    0 < i < nelx-1 and 0 < j < nely-1 and 0 < k < nelz-1
                )
                
                if on_edge or interior_pattern:
                    density[k, j, i] = 1.0
                else:
                    density[k, j, i] = 0.01
    
    return density


def test_metamaterial(structure_name, density_field, nelx, nely, nelz):
    """Run homogenization test on a metamaterial structure."""
    print("\n" + "="*80)
    print(f"METAMATERIAL TEST: {structure_name}")
    print("="*80)
    print(f"Mesh size: {nelx}×{nely}×{nelz}")
    print(f"Density statistics:")
    print(f"  Min: {density_field.min():.4f}")
    print(f"  Max: {density_field.max():.4f}")
    print(f"  Mean: {density_field.mean():.4f}")
    print(f"  Volume fraction: {density_field.mean():.1%}")
    
    # Create material and homogenize
    material = IsotropicMaterial(E=1.0, nu=0.3, name="Base Material")
    
    # Run homogenization with density field
    C_eff = homogenize_3d(nelx, nely, nelz, material, 
                         density_field=density_field, verbose=False)
    
    # Extract effective properties
    C11 = C_eff[0, 0]
    C12 = C_eff[0, 1]
    C44 = C_eff[3, 3]
    
    print(f"\nEffective elastic tensor properties:")
    print(f"  C[0,0]: {C11:.6f}")
    print(f"  C[0,1]: {C12:.6f}")
    print(f"  C[3,3]: {C44:.6f}")
    
    # Estimate effective Young's modulus (assuming isotropic)
    # For isotropic: C11 = λ + 2μ, C44 = μ
    mu_eff = C44
    lambda_eff = C12
    
    if mu_eff > 1e-10:
        E_eff = mu_eff * (3*lambda_eff + 2*mu_eff) / (lambda_eff + mu_eff)
        nu_eff = lambda_eff / (2*(lambda_eff + mu_eff))
    else:
        E_eff = 0.0
        nu_eff = 0.0
    
    print(f"\nEstimated effective isotropic properties:")
    print(f"  E_eff: {E_eff:.6f}")
    print(f"  ν_eff: {nu_eff:.6f}")
    print(f"  μ_eff: {mu_eff:.6f}")
    
    # Tensor symmetry check
    sym_error = np.max(np.abs(C_eff - C_eff.T))
    print(f"\nSymmetry check: {sym_error:.2e}")
    
    # Eigenvalues
    eigvals = np.linalg.eigvalsh(C_eff)
    print(f"Eigenvalues: min={eigvals[0]:.6f}, max={eigvals[-1]:.6f}")
    
    return C_eff


if __name__ == "__main__":
    # Test 1: Checkerboard pattern
    print("\n" + "#"*80)
    print("# CHECKERBOARD PATTERN (Void-Solid Alternation)")
    print("#"*80)
    nelx, nely, nelz = 4, 4, 4
    density_cb = create_checkerboard(nelx, nely, nelz)
    C_cb = test_metamaterial("3D Checkerboard", density_cb, nelx, nely, nelz)
    
    # Test 2: Layered composite (z-direction)
    print("\n" + "#"*80)
    print("# LAYERED COMPOSITE (Horizontal Layers)")
    print("#"*80)
    nelx, nely, nelz = 4, 4, 4
    density_layer_z = create_layered_composite(nelx, nely, nelz, direction='z', period=2)
    C_layer_z = test_metamaterial("Layered (z-direction)", density_layer_z, 
                                  nelx, nely, nelz)
    
    # Test 3: Layered composite (x-direction)
    print("\n" + "#"*80)
    print("# LAYERED COMPOSITE (Vertical Layers - X Direction)")
    print("#"*80)
    density_layer_x = create_layered_composite(nelx, nely, nelz, direction='x', period=2)
    C_layer_x = test_metamaterial("Layered (x-direction)", density_layer_x, 
                                  nelx, nely, nelz)
    
    # Test 4: Graded density (z-direction)
    print("\n" + "#"*80)
    print("# GRADED DENSITY FIELD (Z-Direction Gradient)")
    print("#"*80)
    density_grad_z = create_graded_density(nelx, nely, nelz, gradient_dir='z')
    C_grad_z = test_metamaterial("Graded (z-direction)", density_grad_z, 
                                 nelx, nely, nelz)
    
    # Test 5: Octet truss
    print("\n" + "#"*80)
    print("# OCTET TRUSS-LIKE STRUCTURE")
    print("#"*80)
    density_octet = create_octet_truss_like(nelx, nely, nelz)
    C_octet = test_metamaterial("Octet Truss-like", density_octet, 
                                nelx, nely, nelz)
    
    # Comparison summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Structure':<30} {'C[0,0]':<12} {'C[0,1]':<12} {'C[3,3]':<12}")
    print("-"*80)
    print(f"{'Checkerboard':<30} {C_cb[0,0]:<12.6f} {C_cb[0,1]:<12.6f} {C_cb[3,3]:<12.6f}")
    print(f"{'Layered (z)':<30} {C_layer_z[0,0]:<12.6f} {C_layer_z[0,1]:<12.6f} {C_layer_z[3,3]:<12.6f}")
    print(f"{'Layered (x)':<30} {C_layer_x[0,0]:<12.6f} {C_layer_x[0,1]:<12.6f} {C_layer_x[3,3]:<12.6f}")
    print(f"{'Graded (z)':<30} {C_grad_z[0,0]:<12.6f} {C_grad_z[0,1]:<12.6f} {C_grad_z[3,3]:<12.6f}")
    print(f"{'Octet Truss':<30} {C_octet[0,0]:<12.6f} {C_octet[0,1]:<12.6f} {C_octet[3,3]:<12.6f}")

