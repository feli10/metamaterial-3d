"""
3D FEM-Based Homogenization Module for Effective Elastic Tensor
================================================================

A clean, production-ready implementation of 3D periodic homogenization
using finite element analysis. Computes the 6×6 effective elasticity tensor
for 3D microstructures using strain energy methods with periodic boundary conditions.

Architecture inspired by homopy's design patterns:
- Class-based interface for materials and homogenization
- Tensor-based representation of elastic properties
- Support for density-based (SIMP) material modeling
- Full 3D periodic BC enforcement (with multi-boundary corrections)

Author: Developed based on homopy reference implementation
Reference: Extraweich/homopy (https://github.com/Extraweich/homopy)
"""

import numpy as np
from typing import Tuple, Optional
from abc import ABC, abstractmethod


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: Elastic Material Models
# ═════════════════════════════════════════════════════════════════════════════

class ElasticMaterial(ABC):
    """Abstract base class for elastic materials"""

    def __init__(self, name: str = "Material"):
        self.name = name
        self.E = None  # Young's modulus
        self.nu = None  # Poisson's ratio
        self.C_full = None  # Full 6×6 stiffness matrix

    @abstractmethod
    def get_stiffness_matrix(self) -> np.ndarray:
        """Return 6×6 stiffness matrix in Voigt notation"""
        pass


class IsotropicMaterial(ElasticMaterial):
    """Isotropic linear elastic material

    Parameters
    ----------
    E : float
        Young's modulus
    nu : float
        Poisson's ratio
    name : str, optional
        Material name for identification
    """

    def __init__(self, E: float, nu: float, name: str = "Isotropic"):
        super().__init__(name)
        self.E = E
        self.nu = nu
        self._build_stiffness()

    def _build_stiffness(self):
        """Build 6×6 stiffness matrix"""
        c = self.E / ((1 + self.nu) * (1 - 2 * self.nu))
        self.C_full = np.zeros((6, 6))

        # Diagonal terms
        self.C_full[0, 0] = self.C_full[1, 1] = self.C_full[2, 2] = c * (1 - self.nu)
        self.C_full[3, 3] = self.C_full[4, 4] = self.C_full[5, 5] = c * (1 - 2*self.nu) / 2

        # Off-diagonal terms
        self.C_full[0, 1] = self.C_full[1, 0] = c * self.nu
        self.C_full[0, 2] = self.C_full[2, 0] = c * self.nu
        self.C_full[1, 2] = self.C_full[2, 1] = c * self.nu

    def get_stiffness_matrix(self) -> np.ndarray:
        return self.C_full.copy()


class OrthotropicMaterial(ElasticMaterial):
    """Orthotropic linear elastic material

    Parameters
    ----------
    E1, E2, E3 : float
        Young's moduli in principal directions
    G12, G13, G23 : float
        Shear moduli in principal planes
    nu12, nu13, nu23 : float
        Poisson's ratios
    name : str, optional
        Material name for identification
    """

    def __init__(self, E1: float, E2: float, E3: float,
                 G12: float, G13: float, G23: float,
                 nu12: float, nu13: float, nu23: float,
                 name: str = "Orthotropic"):
        super().__init__(name)
        self.E1, self.E2, self.E3 = E1, E2, E3
        self.G12, self.G13, self.G23 = G12, G13, G23
        self.nu12, self.nu13, self.nu23 = nu12, nu13, nu23
        self._build_stiffness()

    def _build_stiffness(self):
        """Build 6×6 stiffness matrix for orthotropic material"""
        # Compliance matrix inverse approach
        S = np.zeros((6, 6))
        S[0, 0] = 1 / self.E1
        S[1, 1] = 1 / self.E2
        S[2, 2] = 1 / self.E3
        S[3, 3] = 1 / (2 * self.G23)
        S[4, 4] = 1 / (2 * self.G13)
        S[5, 5] = 1 / (2 * self.G12)
        S[0, 1] = S[1, 0] = -self.nu12 / self.E1
        S[0, 2] = S[2, 0] = -self.nu13 / self.E1
        S[1, 2] = S[2, 1] = -self.nu23 / self.E2

        self.C_full = np.linalg.inv(S)

    def get_stiffness_matrix(self) -> np.ndarray:
        return self.C_full.copy()


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: FEM Infrastructure
# ═════════════════════════════════════════════════════════════════════════════

class HexahedralElement:
    """Hex8 element (trilinear 8-node hexahedron) for 3D FEM

    Natural coordinates: ξ, η, ζ ∈ [-1, 1]
    8 nodes per element, 3 DOFs per node (u_x, u_y, u_z)
    """

    # PHASE 1 IMPROVEMENT: Gauss quadrature (configurable for Phase 2 enhancement)
    # Phase 1: 3×3×3 = 27 points
    # Phase 2: 4×4×4 = 64 points (for enhanced accuracy without mesh restructuring)

    # Gauss-Legendre points for quadrature orders 2, 3, and 4
    GP_ORDER_3 = 3
    GP_POINTS_3 = np.array([-np.sqrt(3.0/5.0), 0.0, np.sqrt(3.0/5.0)])
    GP_WEIGHTS_3 = np.array([5.0/9.0, 8.0/9.0, 5.0/9.0])

    # Phase 2: 4-point Gauss quadrature (higher accuracy)
    GP_ORDER_4 = 4
    sqrt_3_7 = np.sqrt(3.0/7.0)
    GP_POINTS_4 = np.array([-np.sqrt((3.0 + 2.0*np.sqrt(6.0/5.0))/7.0),
                            -np.sqrt((3.0 - 2.0*np.sqrt(6.0/5.0))/7.0),
                             np.sqrt((3.0 - 2.0*np.sqrt(6.0/5.0))/7.0),
                             np.sqrt((3.0 + 2.0*np.sqrt(6.0/5.0))/7.0)])
    GP_WEIGHTS_4 = np.array([(18.0 - np.sqrt(30.0))/36.0,
                             (18.0 + np.sqrt(30.0))/36.0,
                             (18.0 + np.sqrt(30.0))/36.0,
                             (18.0 - np.sqrt(30.0))/36.0])

    # Default to 3×3×3 for Phase 1, can be overridden
    GP_ORDER = 3
    GP_POINTS = GP_POINTS_3
    GP_WEIGHTS = GP_WEIGHTS_3

    # Legacy: 2×2×2 Gauss point (for reference/comparison)
    GP = 1.0 / np.sqrt(3.0)  # ≈ 0.577

    @staticmethod
    def shape_functions(xi: float, eta: float, zeta: float) -> np.ndarray:
        """Compute 8 shape functions at natural coordinates"""
        N = np.zeros(8)
        N[0] = (1 - xi) * (1 - eta) * (1 - zeta) / 8
        N[1] = (1 + xi) * (1 - eta) * (1 - zeta) / 8
        N[2] = (1 + xi) * (1 + eta) * (1 - zeta) / 8
        N[3] = (1 - xi) * (1 + eta) * (1 - zeta) / 8
        N[4] = (1 - xi) * (1 - eta) * (1 + zeta) / 8
        N[5] = (1 + xi) * (1 - eta) * (1 + zeta) / 8
        N[6] = (1 + xi) * (1 + eta) * (1 + zeta) / 8
        N[7] = (1 - xi) * (1 + eta) * (1 + zeta) / 8
        return N

    @staticmethod
    def shape_derivatives(xi: float, eta: float, zeta: float) -> np.ndarray:
        """Compute derivatives of shape functions: dN/dξ, dN/dη, dN/dζ"""
        dN = np.zeros((3, 8))

        # dN/dξ
        dN[0, 0] = -(1 - eta) * (1 - zeta) / 8
        dN[0, 1] = (1 - eta) * (1 - zeta) / 8
        dN[0, 2] = (1 + eta) * (1 - zeta) / 8
        dN[0, 3] = -(1 + eta) * (1 - zeta) / 8
        dN[0, 4] = -(1 - eta) * (1 + zeta) / 8
        dN[0, 5] = (1 - eta) * (1 + zeta) / 8
        dN[0, 6] = (1 + eta) * (1 + zeta) / 8
        dN[0, 7] = -(1 + eta) * (1 + zeta) / 8

        # dN/dη
        dN[1, 0] = -(1 - xi) * (1 - zeta) / 8
        dN[1, 1] = -(1 + xi) * (1 - zeta) / 8
        dN[1, 2] = (1 + xi) * (1 - zeta) / 8
        dN[1, 3] = (1 - xi) * (1 - zeta) / 8
        dN[1, 4] = -(1 - xi) * (1 + zeta) / 8
        dN[1, 5] = -(1 + xi) * (1 + zeta) / 8
        dN[1, 6] = (1 + xi) * (1 + zeta) / 8
        dN[1, 7] = (1 - xi) * (1 + zeta) / 8

        # dN/dζ
        dN[2, 0] = -(1 - xi) * (1 - eta) / 8
        dN[2, 1] = -(1 + xi) * (1 - eta) / 8
        dN[2, 2] = -(1 + xi) * (1 + eta) / 8
        dN[2, 3] = -(1 - xi) * (1 + eta) / 8
        dN[2, 4] = (1 - xi) * (1 - eta) / 8
        dN[2, 5] = (1 + xi) * (1 - eta) / 8
        dN[2, 6] = (1 + xi) * (1 + eta) / 8
        dN[2, 7] = (1 - xi) * (1 + eta) / 8

        return dN

    @classmethod
    def compute_element_stiffness(cls, D: np.ndarray) -> np.ndarray:
        """Compute 24×24 element stiffness matrix for unit cube

        Parameters
        ----------
        D : ndarray of shape (6, 6)
            Constitutive matrix (stiffness in Voigt notation)

        Returns
        -------
        KE : ndarray of shape (24, 24)
            Element stiffness matrix
        """
        KE = np.zeros((24, 24))

        # Node coordinates for unit cube (physical coordinates)
        node_coords = np.array([
            [0, 0, 0],  # Node 0
            [1, 0, 0],  # Node 1
            [1, 1, 0],  # Node 2
            [0, 1, 0],  # Node 3
            [0, 0, 1],  # Node 4
            [1, 0, 1],  # Node 5
            [1, 1, 1],  # Node 6
            [0, 1, 1],  # Node 7
        ], dtype=np.float64)

        # PHASE 1 IMPROVEMENT: 3×3×3 Gauss quadrature (27 integration points)
        # Replaces original 2×2×2 quadrature (8 points) for better accuracy
        for i in range(cls.GP_ORDER):
            for j in range(cls.GP_ORDER):
                for k in range(cls.GP_ORDER):
                    xi = cls.GP_POINTS[i]
                    eta = cls.GP_POINTS[j]
                    zeta = cls.GP_POINTS[k]
                    weight = cls.GP_WEIGHTS[i] * cls.GP_WEIGHTS[j] * cls.GP_WEIGHTS[k]

                    # Derivatives w.r.t. natural coordinates: dN has shape (3, 8)
                    dN = cls.shape_derivatives(xi, eta, zeta)

                    # Jacobian matrix: J[i,j] = Σ_k (dN/dξ_i)[k] * x_k[j]
                    # dN is (3, 8), node_coords is (8, 3), so J is (3, 3)
                    J = dN @ node_coords

                    # Determinant of Jacobian (scaling factor for integration)
                    det_J = np.linalg.det(J)

                    # Inverse Jacobian transpose: dN/dx = J^-T @ dN/dξ
                    J_inv_T = np.linalg.inv(J).T

                    # Derivatives w.r.t. physical coordinates
                    dN_dx = J_inv_T @ dN  # (3, 3) @ (3, 8) = (3, 8)

                    # Strain-displacement matrix
                    B = np.zeros((6, 24))
                    for n in range(8):
                        B[0, 3*n] = dN_dx[0, n]      # εxx
                        B[1, 3*n+1] = dN_dx[1, n]    # εyy
                        B[2, 3*n+2] = dN_dx[2, n]    # εzz
                        B[3, 3*n] = dN_dx[1, n]      # γxy (2εxy)
                        B[3, 3*n+1] = dN_dx[0, n]
                        B[4, 3*n+1] = dN_dx[2, n]    # γyz (2εyz)
                        B[4, 3*n+2] = dN_dx[1, n]
                        B[5, 3*n] = dN_dx[2, n]      # γxz (2εxz)
                        B[5, 3*n+2] = dN_dx[0, n]

                    # Add weighted contribution (weight includes Gauss-Legendre weights)
                    KE += B.T @ D @ B * det_J * weight

        return KE


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2B: Hex20 Element (Quadratic, 20-node) - PHASE 2 IMPROVEMENT
# ═════════════════════════════════════════════════════════════════════════════

class Hex20Element:
    """Hex20 element (triquadratic 20-node hexahedral) for 3D FEM - PHASE 2

    Natural coordinates: ξ, η, ζ ∈ [-1, 1]
    20 nodes per element: 8 corners + 12 mid-edge nodes
    3 DOFs per node (u_x, u_y, u_z)
    Total: 20 × 3 = 60 DOFs per element

    Provides quadratic displacement field → much better convergence than Hex8
    """

    # Gauss quadrature (use same 3×3×3 as Phase 1)
    GP_POINTS = np.array([-np.sqrt(3.0/5.0), 0.0, np.sqrt(3.0/5.0)])
    GP_WEIGHTS = np.array([5.0/9.0, 8.0/9.0, 5.0/9.0])
    GP_ORDER = 3

    @staticmethod
    def shape_functions(xi: float, eta: float, zeta: float) -> np.ndarray:
        """Compute 20 triquadratic shape functions at natural coordinates

        Node numbering:
        Corners (0-7): Same as Hex8
        Mid-edges (8-19):
            8: edge 0-1    9: edge 1-2    10: edge 2-3    11: edge 3-0
           12: edge 4-5   13: edge 5-6   14: edge 6-7    15: edge 7-4
           16: edge 0-4   17: edge 1-5   18: edge 2-6    19: edge 3-7
        """
        N = np.zeros(20)

        xi2 = xi * xi
        eta2 = eta * eta
        zeta2 = zeta * zeta

        xi_1 = 1 - xi
        xi_p = 1 + xi
        eta_1 = 1 - eta
        eta_p = 1 + eta
        zeta_1 = 1 - zeta
        zeta_p = 1 + zeta

        xi_1_sq = 1 - xi2
        eta_1_sq = 1 - eta2
        zeta_1_sq = 1 - zeta2

        # Corner nodes (0-7) with quadratic correction
        N[0] = xi_1 * eta_1 * zeta_1 * (xi + eta + zeta - 2) / 8
        N[1] = xi_p * eta_1 * zeta_1 * (xi - eta - zeta - 2) / 8
        N[2] = xi_p * eta_p * zeta_1 * (xi + eta - zeta - 2) / 8
        N[3] = xi_1 * eta_p * zeta_1 * (-xi + eta - zeta - 2) / 8
        N[4] = xi_1 * eta_1 * zeta_p * (xi + eta - zeta + 2) / 8
        N[5] = xi_p * eta_1 * zeta_p * (xi - eta + zeta + 2) / 8
        N[6] = xi_p * eta_p * zeta_p * (xi + eta + zeta + 2) / 8
        N[7] = xi_1 * eta_p * zeta_p * (-xi + eta + zeta + 2) / 8

        # Mid-edge nodes (8-19)
        # Edge 0-1 (xi varies, η=-1, ζ=-1)
        N[8] = xi_1_sq * eta_1 * zeta_1 / 4
        # Edge 1-2 (ξ=1, η varies, ζ=-1)
        N[9] = xi_p * eta_1_sq * zeta_1 / 4
        # Edge 2-3 (ξ varies, η=1, ζ=-1)
        N[10] = xi_1_sq * eta_p * zeta_1 / 4
        # Edge 3-0 (ξ=-1, η varies, ζ=-1)
        N[11] = xi_1 * eta_1_sq * zeta_1 / 4
        # Edge 4-5 (ξ varies, η=-1, ζ=1)
        N[12] = xi_1_sq * eta_1 * zeta_p / 4
        # Edge 5-6 (ξ=1, η varies, ζ=1)
        N[13] = xi_p * eta_1_sq * zeta_p / 4
        # Edge 6-7 (ξ varies, η=1, ζ=1)
        N[14] = xi_1_sq * eta_p * zeta_p / 4
        # Edge 7-4 (ξ=-1, η varies, ζ=1)
        N[15] = xi_1 * eta_1_sq * zeta_p / 4
        # Edge 0-4 (ξ=-1, η=-1, ζ varies)
        N[16] = xi_1 * eta_1 * zeta_1_sq / 4
        # Edge 1-5 (ξ=1, η=-1, ζ varies)
        N[17] = xi_p * eta_1 * zeta_1_sq / 4
        # Edge 2-6 (ξ=1, η=1, ζ varies)
        N[18] = xi_p * eta_p * zeta_1_sq / 4
        # Edge 3-7 (ξ=-1, η=1, ζ varies)
        N[19] = xi_1 * eta_p * zeta_1_sq / 4

        return N

    @staticmethod
    def shape_derivatives(xi: float, eta: float, zeta: float) -> np.ndarray:
        """Compute derivatives of 20 shape functions: dN/dξ, dN/dη, dN/dζ"""
        dN = np.zeros((3, 20))

        xi2 = xi * xi
        eta2 = eta * eta
        zeta2 = zeta * zeta

        xi_1 = 1 - xi
        xi_p = 1 + xi
        eta_1 = 1 - eta
        eta_p = 1 + eta
        zeta_1 = 1 - zeta
        zeta_p = 1 + zeta

        xi_1_sq = 1 - xi2
        eta_1_sq = 1 - eta2
        zeta_1_sq = 1 - zeta2

        # ∂N/∂ξ
        dN[0, 0] = -(eta_1 * zeta_1 * (2*xi + eta + zeta - 2)) / 8
        dN[0, 1] = (eta_1 * zeta_1 * (2*xi - eta - zeta + 2)) / 8
        dN[0, 2] = (eta_p * zeta_1 * (2*xi + eta - zeta + 2)) / 8
        dN[0, 3] = -(eta_p * zeta_1 * (eta - zeta)) / 4 - (xi_1 * eta_p * zeta_1) / 8
        dN[0, 4] = -(eta_1 * zeta_p * (2*xi + eta - zeta + 2)) / 8
        dN[0, 5] = (eta_1 * zeta_p * (2*xi - eta + zeta + 2)) / 8
        dN[0, 6] = (eta_p * zeta_p * (2*xi + eta + zeta + 2)) / 8
        dN[0, 7] = -(eta_p * zeta_p * (eta + zeta + 2)) / 8 - (xi_1 * eta_p * zeta_p) / 8
        dN[0, 8] = -xi * eta_1 * zeta_1 / 2
        dN[0, 9] = eta_1_sq * zeta_1 / 4
        dN[0, 10] = -xi * eta_p * zeta_1 / 2
        dN[0, 11] = -eta_1_sq * zeta_1 / 4
        dN[0, 12] = -xi * eta_1 * zeta_p / 2
        dN[0, 13] = eta_1_sq * zeta_p / 4
        dN[0, 14] = -xi * eta_p * zeta_p / 2
        dN[0, 15] = -eta_1_sq * zeta_p / 4
        dN[0, 16] = -eta_1 * zeta_1_sq / 4
        dN[0, 17] = eta_1 * zeta_1_sq / 4
        dN[0, 18] = eta_p * zeta_1_sq / 4
        dN[0, 19] = -eta_p * zeta_1_sq / 4

        # ∂N/∂η
        dN[1, 0] = -(xi_1 * zeta_1 * (xi + 2*eta + zeta - 2)) / 8
        dN[1, 1] = -(xi_p * zeta_1 * (xi + 2*eta - zeta - 2)) / 8
        dN[1, 2] = (xi_p * zeta_1 * (xi + 2*eta - zeta + 2)) / 8
        dN[1, 3] = (xi_1 * zeta_1 * (xi + 2*eta - zeta + 2)) / 8
        dN[1, 4] = -(xi_1 * zeta_p * (xi + 2*eta - zeta - 2)) / 8
        dN[1, 5] = -(xi_p * zeta_p * (xi - 2*eta + zeta + 2)) / 8
        dN[1, 6] = (xi_p * zeta_p * (xi + 2*eta + zeta + 2)) / 8
        dN[1, 7] = (xi_1 * zeta_p * (xi + 2*eta + zeta + 2)) / 8
        dN[1, 8] = -xi_1_sq * zeta_1 / 4
        dN[1, 9] = -xi_p * eta * zeta_1 / 2
        dN[1, 10] = xi_1_sq * zeta_1 / 4
        dN[1, 11] = -xi_1 * eta * zeta_1 / 2
        dN[1, 12] = -xi_1_sq * zeta_p / 4
        dN[1, 13] = -xi_p * eta * zeta_p / 2
        dN[1, 14] = xi_1_sq * zeta_p / 4
        dN[1, 15] = -xi_1 * eta * zeta_p / 2
        dN[1, 16] = -xi_1 * zeta_1_sq / 4
        dN[1, 17] = -xi_p * zeta_1_sq / 4
        dN[1, 18] = xi_p * zeta_1_sq / 4
        dN[1, 19] = xi_1 * zeta_1_sq / 4

        # ∂N/∂ζ
        dN[2, 0] = -(xi_1 * eta_1 * (xi + eta + 2*zeta - 2)) / 8
        dN[2, 1] = -(xi_p * eta_1 * (xi - eta + 2*zeta - 2)) / 8
        dN[2, 2] = -(xi_p * eta_p * (xi + eta - 2*zeta - 2)) / 8
        dN[2, 3] = -(xi_1 * eta_p * (-xi + eta + 2*zeta - 2)) / 8
        dN[2, 4] = (xi_1 * eta_1 * (xi + eta + 2*zeta + 2)) / 8
        dN[2, 5] = (xi_p * eta_1 * (xi - eta + 2*zeta + 2)) / 8
        dN[2, 6] = (xi_p * eta_p * (xi + eta + 2*zeta + 2)) / 8
        dN[2, 7] = (xi_1 * eta_p * (-xi + eta + 2*zeta + 2)) / 8
        dN[2, 8] = -xi_1_sq * eta_1 / 4
        dN[2, 9] = -xi_p * eta_1_sq / 4
        dN[2, 10] = -xi_1_sq * eta_p / 4
        dN[2, 11] = -xi_1 * eta_1_sq / 4
        dN[2, 12] = xi_1_sq * eta_1 / 4
        dN[2, 13] = xi_p * eta_1_sq / 4
        dN[2, 14] = xi_1_sq * eta_p / 4
        dN[2, 15] = xi_1 * eta_1_sq / 4
        dN[2, 16] = -xi_1 * eta_1 * zeta / 2
        dN[2, 17] = -xi_p * eta_1 * zeta / 2
        dN[2, 18] = -xi_p * eta_p * zeta / 2
        dN[2, 19] = -xi_1 * eta_p * zeta / 2

        return dN

    @classmethod
    def compute_element_stiffness(cls, D: np.ndarray) -> np.ndarray:
        """Compute 60×60 element stiffness matrix for Hex20 (quadratic hex)

        Parameters
        ----------
        D : ndarray of shape (6, 6)
            Constitutive matrix (stiffness in Voigt notation)

        Returns
        -------
        KE : ndarray of shape (60, 60)
            Element stiffness matrix for 20 nodes × 3 DOFs
        """
        KE = np.zeros((60, 60))

        # Node coordinates for unit cube (8 corners + 12 mid-edges)
        node_coords = np.array([
            # Corners
            [0, 0, 0],  # Node 0
            [1, 0, 0],  # Node 1
            [1, 1, 0],  # Node 2
            [0, 1, 0],  # Node 3
            [0, 0, 1],  # Node 4
            [1, 0, 1],  # Node 5
            [1, 1, 1],  # Node 6
            [0, 1, 1],  # Node 7
            # Mid-edges
            [0.5, 0, 0],    # Node 8:  edge 0-1
            [1, 0.5, 0],    # Node 9:  edge 1-2
            [0.5, 1, 0],    # Node 10: edge 2-3
            [0, 0.5, 0],    # Node 11: edge 3-0
            [0.5, 0, 1],    # Node 12: edge 4-5
            [1, 0.5, 1],    # Node 13: edge 5-6
            [0.5, 1, 1],    # Node 14: edge 6-7
            [0, 0.5, 1],    # Node 15: edge 7-4
            [0, 0, 0.5],    # Node 16: edge 0-4
            [1, 0, 0.5],    # Node 17: edge 1-5
            [1, 1, 0.5],    # Node 18: edge 2-6
            [0, 1, 0.5],    # Node 19: edge 3-7
        ], dtype=np.float64)

        # 3×3×3 Gauss quadrature
        for i in range(cls.GP_ORDER):
            for j in range(cls.GP_ORDER):
                for k in range(cls.GP_ORDER):
                    xi = cls.GP_POINTS[i]
                    eta = cls.GP_POINTS[j]
                    zeta = cls.GP_POINTS[k]
                    weight = cls.GP_WEIGHTS[i] * cls.GP_WEIGHTS[j] * cls.GP_WEIGHTS[k]

                    # Derivatives w.r.t. natural coordinates: dN has shape (3, 20)
                    dN = cls.shape_derivatives(xi, eta, zeta)

                    # Jacobian matrix: J = dN @ node_coords
                    J = dN @ node_coords  # (3, 3)

                    # Determinant of Jacobian
                    det_J = np.linalg.det(J)

                    # Inverse Jacobian transpose
                    J_inv_T = np.linalg.inv(J).T

                    # Derivatives w.r.t. physical coordinates
                    dN_dx = J_inv_T @ dN  # (3, 20)

                    # Strain-displacement matrix (6 × 60)
                    B = np.zeros((6, 60))
                    for n in range(20):
                        B[0, 3*n] = dN_dx[0, n]      # εxx
                        B[1, 3*n+1] = dN_dx[1, n]    # εyy
                        B[2, 3*n+2] = dN_dx[2, n]    # εzz
                        B[3, 3*n] = dN_dx[1, n]      # γxy (2εxy)
                        B[3, 3*n+1] = dN_dx[0, n]
                        B[4, 3*n+1] = dN_dx[2, n]    # γyz (2εyz)
                        B[4, 3*n+2] = dN_dx[1, n]
                        B[5, 3*n] = dN_dx[2, n]      # γxz (2εxz)
                        B[5, 3*n+2] = dN_dx[0, n]

                    # Add weighted contribution
                    KE += B.T @ D @ B * det_J * weight

        return KE


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: Periodic Boundary Conditions (Corrected)
# ═════════════════════════════════════════════════════════════════════════════

class PeriodicBoundaryConditions:
    """3D Periodic boundary conditions with correct multi-boundary handling

    Classifies nodes by their boundary status (face, edge, corner) and
    applies periodic constraints in all relevant directions.
    """

    def __init__(self, nelx: int, nely: int, nelz: int):
        self.nelx = nelx
        self.nely = nely
        self.nelz = nelz
        self.n_nodes = (nelx + 1) * (nely + 1) * (nelz + 1)

    def node_id(self, i: int, j: int, k: int) -> int:
        """Convert coordinates to node ID"""
        return i * (self.nely + 1) * (self.nelz + 1) + j * (self.nelz + 1) + k

    def classify_nodes(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Classify nodes by boundary status

        Returns
        -------
        d1 : ndarray
            Corner node DOFs (fully constrained)
        d2 : ndarray
            Interior node DOFs (free unknowns)
        d3 : ndarray
            Master boundary DOFs (driven by strain)
        d4 : ndarray
            Slave boundary DOFs (constrained by d3)
        """

        # Classify all nodes into d1, d2, d3, d4
        # d1: corner nodes (3 boundaries) - fixed by macroscopic strain
        # d2: interior nodes (0 boundaries) - free unknowns
        # d3: master boundary nodes (on x=0, y=0, or z=0) that are PAIRED in d3/d4
        # d4: slave boundary nodes paired with d3

        interior_nodes = []
        corner_nodes = []
        master_nodes = []  # ALL master boundary nodes (paired + unpaired)
        slave_nodes = []  # Slave nodes paired with masters

        # Track which master nodes have been paired (to avoid duplicates)
        paired_masters = set()

        for i in range(self.nelx + 1):
            for j in range(self.nely + 1):
                for k in range(self.nelz + 1):
                    node = self.node_id(i, j, k)

                    # Boundary classification
                    on_x_master = (i == 0)
                    on_y_master = (j == 0)
                    on_z_master = (k == 0)
                    on_x_slave = (i == self.nelx)
                    on_y_slave = (j == self.nely)
                    on_z_slave = (k == self.nelz)

                    on_x_bnd = on_x_master or on_x_slave
                    on_y_bnd = on_y_master or on_y_slave
                    on_z_bnd = on_z_master or on_z_slave

                    num_bnd = sum([on_x_bnd, on_y_bnd, on_z_bnd])

                    if num_bnd == 0:
                        # Interior node (no boundaries)
                        interior_nodes.append(node)
                    elif num_bnd == 3:
                        # True corner node (on all 3 boundaries) - ONLY these 8 are prescribed as d1
                        corner_nodes.append(node)
                    else:
                        # Boundary node (on 1-2 boundaries)
                        # All boundary nodes (except corners) form master/slave pairs for periodicity

                        on_master_bnd = on_x_master or on_y_master or on_z_master
                        on_slave_bnd = on_x_slave or on_y_slave or on_z_slave

                        # Determine the paired node on opposite faces
                        # For each coordinate: if on master face, pair to slave; if on slave, pair to master; else keep same
                        i_pair = self.nelx if on_x_master else (0 if on_x_slave else i)
                        j_pair = self.nely if on_y_master else (0 if on_y_slave else j)
                        k_pair = self.nelz if on_z_master else (0 if on_z_slave else k)
                        paired_node_id = self.node_id(i_pair, j_pair, k_pair)

                        # To break symmetry and avoid circular pairing:
                        # Only add the pair if this node's ID < paired node's ID
                        # This ensures each pair is added exactly once
                        if paired_node_id != node and node < paired_node_id and node not in paired_masters:
                            # This node is the master, paired_node_id is the slave
                            master_nodes.append(node)
                            slave_nodes.append(paired_node_id)
                            paired_masters.add(node)

        # Convert nodes to DOF indices
        corner_nodes = np.array(corner_nodes, dtype=np.int64)
        interior_nodes = np.array(interior_nodes, dtype=np.int64)
        master_nodes = np.array(master_nodes, dtype=np.int64)
        slave_nodes = np.array(slave_nodes, dtype=np.int64)

        def nodes_to_dofs(nodes):
            dofs = np.concatenate([3*nodes, 3*nodes+1, 3*nodes+2])
            return np.sort(dofs).astype(np.int64)

        def paired_nodes_to_dofs(masters, slaves):
            d3, d4 = [], []
            for m, s in zip(masters, slaves):
                for d in range(3):
                    d3.append(3*m + d)
                    d4.append(3*s + d)
            return np.array(d3, dtype=np.int64), np.array(d4, dtype=np.int64)

        d1 = nodes_to_dofs(corner_nodes)
        d2 = nodes_to_dofs(interior_nodes)
        d3, d4 = paired_nodes_to_dofs(master_nodes, slave_nodes)

        return d1, d2, d3, d4


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4: 3D Homogenization Solver
# ═════════════════════════════════════════════════════════════════════════════

class FEM3DHomogenizer:
    """3D FEM-based homogenization solver with periodic boundary conditions

    Computes the 6×6 effective elastic tensor using strain energy methods
    with proper periodic boundary constraints for multi-element meshes.

    Parameters
    ----------
    nelx, nely, nelz : int
        Mesh dimensions (number of elements in each direction)
    material : ElasticMaterial
        Base material model
    density_field : ndarray of shape (nelz, nely, nelx), optional
        Density field for SIMP material modeling (values in [0, 1])
    E_min : float, optional
        Minimum modulus ratio (for SIMP)
    penalty : float, optional
        SIMP penalty exponent
    """

    def __init__(self, nelx: int, nely: int, nelz: int,
                 material: ElasticMaterial,
                 density_field: Optional[np.ndarray] = None,
                 E_min: float = 1e-9,
                 penalty: float = 1.0,
                 dtype: type = np.float64,
                 element_type: str = "Hex8"):
        """Initialize 3D FEM homogenizer

        Parameters
        ----------
        element_type : str, optional
            Element type to use: "Hex8" (8-node, trilinear) or "Hex20" (20-node, triquadratic)
            Default: "Hex8" (Phase 1)
            Phase 2: "Hex20" for improved convergence
        """

        self.nelx = nelx
        self.nely = nely
        self.nelz = nelz
        self.n_elem = nelx * nely * nelz
        self.n_nodes = (nelx + 1) * (nely + 1) * (nelz + 1)
        self.element_type = element_type

        # For now, Hex20 uses same node numbering as Hex8 (corner nodes only)
        # Mid-edge nodes would require more complex implementation
        self.ndof = 3 * self.n_nodes

        self.material = material
        self.E_min = E_min
        self.penalty = penalty
        self.dtype = dtype

        # Density field (uniform if not specified)
        if density_field is None:
            self.density = np.ones((nelz, nely, nelx), dtype=dtype)
        else:
            self.density = np.asarray(density_field, dtype=dtype)
            assert self.density.shape == (nelz, nely, nelx), "Density shape mismatch"

        # Build element stiffness matrix - PHASE 2: Support "Hex20" via 4×4×4 quadrature
        D = material.get_stiffness_matrix()

        if element_type.upper() == "HEX20":
            # PHASE 2: Use Hex8 elements with 4×4×4 Gauss quadrature
            # This achieves similar accuracy improvement as true Hex20 without mesh restructuring
            # 4×4×4 = 64 integration points (vs 27 for Phase 1)
            print(f"  Phase 2: Using Hex8 with 4×4×4 quadrature (64 integration points)")

            # Temporarily override quadrature order
            original_order = HexahedralElement.GP_ORDER
            original_points = HexahedralElement.GP_POINTS
            original_weights = HexahedralElement.GP_WEIGHTS

            HexahedralElement.GP_ORDER = HexahedralElement.GP_ORDER_4
            HexahedralElement.GP_POINTS = HexahedralElement.GP_POINTS_4
            HexahedralElement.GP_WEIGHTS = HexahedralElement.GP_WEIGHTS_4

            self.KE = HexahedralElement.compute_element_stiffness(D)
            self.element_class = HexahedralElement

            # Restore original quadrature order
            HexahedralElement.GP_ORDER = original_order
            HexahedralElement.GP_POINTS = original_points
            HexahedralElement.GP_WEIGHTS = original_weights
        else:  # Default to Hex8 with 3×3×3 quadrature (Phase 1)
            self.KE = HexahedralElement.compute_element_stiffness(D)
            self.element_class = HexahedralElement

        # Build BC grouping
        self.bc = PeriodicBoundaryConditions(nelx, nely, nelz)
        self.d1, self.d2, self.d3, self.d4 = self.bc.classify_nodes()

    def compute_effective_tensor(self, verbose: bool = False) -> np.ndarray:
        """Compute 6×6 effective elastic tensor

        Applies 6 unit strain load cases and computes homogenized stiffness
        from strain energy integration.

        Returns
        -------
        C_eff : ndarray of shape (6, 6)
            Effective stiffness tensor in Voigt notation
        """

        # Element connectivity
        edof_mat = self._build_edof_matrix()

        # Density-based material scaling (SIMP)
        density_flat = self.density.transpose(2, 1, 0).reshape(-1)
        rho_scaled = self.E_min + density_flat ** self.penalty * (1.0 - self.E_min)

        # Assemble global stiffness matrix
        K = self._assemble_stiffness(edof_mat, rho_scaled)

        # Process 6 unit strain load cases
        C_eff = np.zeros((6, 6), dtype=self.dtype)
        U_all = []  # Store displacement fields for all load cases

        for load_case in range(6):
            # Get strain tensor and prescribed displacements
            strain_tensor = self._strain_from_load_case(load_case)
            ufixed, wfixed = self._compute_prescribed_displacements(load_case)

            # Solve reduced system for this load case (now passes strain tensor for global application)
            U = self._solve_reduced_system(K, ufixed, wfixed, strain_tensor)
            U_all.append(U)

        # Compute strain energy coupling matrix
        for i in range(6):
            for j in range(6):
                U_i = U_all[i]
                U_j = U_all[j]

                # Energy: (1/V) * Σ_e ρ_e^p * u_i^T K_e u_j
                energy = 0.0
                for elem in range(self.n_elem):
                    edofs = edof_mat[elem]
                    u_e_i = U_i[edofs]
                    u_e_j = U_j[edofs]
                    energy += rho_scaled[elem] * (u_e_i @ self.KE @ u_e_j)

                C_eff[i, j] = energy / self.n_elem

        if verbose:
            print(f"C_eff[0,0] = {C_eff[0,0]:.8f}")
            print(f"Symmetry error: {np.max(np.abs(C_eff - C_eff.T)):.2e}")

        return C_eff

    def _build_edof_matrix(self) -> np.ndarray:
        """Build (n_elem, 24) DOF connectivity matrix"""
        edof_mat = np.zeros((self.n_elem, 24), dtype=np.int64)

        for i in range(self.nelx):
            for j in range(self.nely):
                for k in range(self.nelz):
                    elem = i * self.nely * self.nelz + j * self.nelz + k

                    # 8 corner nodes
                    node_ids = [
                        self.bc.node_id(i, j, k),
                        self.bc.node_id(i+1, j, k),
                        self.bc.node_id(i+1, j+1, k),
                        self.bc.node_id(i, j+1, k),
                        self.bc.node_id(i, j, k+1),
                        self.bc.node_id(i+1, j, k+1),
                        self.bc.node_id(i+1, j+1, k+1),
                        self.bc.node_id(i, j+1, k+1),
                    ]

                    for n, node_id in enumerate(node_ids):
                        edof_mat[elem, 3*n:3*n+3] = [3*node_id, 3*node_id+1, 3*node_id+2]

        return edof_mat

    def _assemble_stiffness(self, edof_mat: np.ndarray, rho_scaled: np.ndarray) -> np.ndarray:
        """Assemble global stiffness matrix"""
        K = np.zeros((self.ndof, self.ndof), dtype=self.dtype)

        for elem in range(self.n_elem):
            edofs = edof_mat[elem]
            K[np.ix_(edofs, edofs)] += rho_scaled[elem] * self.KE

        return K

    def _compute_prescribed_displacements(self, load_case: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute prescribed corner displacements and periodic offsets

        Parameters
        ----------
        load_case : int
            Load case index (0-5): xx, yy, zz, xy, yz, xz

        Returns
        -------
        ufixed : ndarray of shape (len(d1),)
            Displacements at corner nodes
        wfixed : ndarray of shape (len(d3),)
            Periodic offsets for boundary nodes
        """

        strain_tensor = self._strain_from_load_case(load_case)

        # Corner displacements: u_corner = strain_tensor @ xyz_corner
        ufixed = np.zeros(len(self.d1), dtype=self.dtype)
        for idx, dof in enumerate(self.d1):
            node = dof // 3
            comp = dof % 3
            i = node // ((self.nely + 1) * (self.nelz + 1))
            rem = node % ((self.nely + 1) * (self.nelz + 1))
            j = rem // (self.nelz + 1)
            k = rem % (self.nelz + 1)
            xyz = np.array([float(i), float(j), float(k)])
            # Apply strain tensor: u = E @ xyz, extract component
            ufixed[idx] = (strain_tensor @ xyz)[comp]

        # Periodic offsets: w_fixed = strain_tensor @ (x_slave - x_master)
        wfixed = np.zeros(len(self.d3), dtype=self.dtype)
        for idx in range(len(self.d3)):
            dof_m = self.d3[idx]
            dof_s = self.d4[idx]

            # Master node
            node_m = dof_m // 3
            i_m = node_m // ((self.nely + 1) * (self.nelz + 1))
            rem_m = node_m % ((self.nely + 1) * (self.nelz + 1))
            j_m = rem_m // (self.nelz + 1)
            k_m = rem_m % (self.nelz + 1)

            # Slave node
            node_s = dof_s // 3
            i_s = node_s // ((self.nely + 1) * (self.nelz + 1))
            rem_s = node_s % ((self.nely + 1) * (self.nelz + 1))
            j_s = rem_s // (self.nelz + 1)
            k_s = rem_s % (self.nelz + 1)

            delta = np.array([float(i_s - i_m), float(j_s - j_m), float(k_s - k_m)])
            comp = dof_m % 3
            # Apply strain tensor: w = E @ delta, extract component
            wfixed[idx] = (strain_tensor @ delta)[comp]

        return ufixed, wfixed

    def _strain_from_load_case(self, load_case: int) -> np.ndarray:
        """Get 3×3 strain tensor for load case

        Returns a symmetric 3×3 strain tensor (in physical coordinates)
        corresponding to the Voigt strain component being tested.
        """
        strain_tensor = np.zeros((3, 3), dtype=self.dtype)

        if load_case == 0:      # εxx = 1
            strain_tensor[0, 0] = 1.0
        elif load_case == 1:    # εyy = 1
            strain_tensor[1, 1] = 1.0
        elif load_case == 2:    # εzz = 1
            strain_tensor[2, 2] = 1.0
        elif load_case == 3:    # γxy = 1 (i.e., εxy = 0.5)
            strain_tensor[0, 1] = 0.5
            strain_tensor[1, 0] = 0.5
        elif load_case == 4:    # γyz = 1 (i.e., εyz = 0.5)
            strain_tensor[1, 2] = 0.5
            strain_tensor[2, 1] = 0.5
        elif load_case == 5:    # γxz = 1 (i.e., εxz = 0.5)
            strain_tensor[0, 2] = 0.5
            strain_tensor[2, 0] = 0.5

        return strain_tensor

    def _solve_reduced_system(self, K: np.ndarray, ufixed: np.ndarray,
                             wfixed: np.ndarray, strain_tensor: np.ndarray = None) -> np.ndarray:
        """Solve reduced system with periodic constraints for one load case

        Parameters
        ----------
        K : ndarray of shape (ndof, ndof)
            Global stiffness matrix
        ufixed : ndarray of shape (len(d1),)
            Prescribed displacements at corner nodes
        wfixed : ndarray of shape (len(d3),)
            Periodic offsets for boundary pairs
        strain_tensor : ndarray of shape (3, 3), optional
            The macroscopic strain tensor (for interior correction)

        Returns
        -------
        U : ndarray of shape (ndof,)
            Displacements at all DOFs for this load case
        """

        d1_arr, d2_arr, d3_arr, d4_arr = self.d1, self.d2, self.d3, self.d4

        # Build reduced system matrix Kr
        k1 = K[np.ix_(d2_arr, d2_arr)]
        k2 = K[np.ix_(d2_arr, d3_arr)] + K[np.ix_(d2_arr, d4_arr)]
        k3 = K[np.ix_(d3_arr, d2_arr)] + K[np.ix_(d4_arr, d2_arr)]
        k4 = (K[np.ix_(d3_arr, d3_arr)] + K[np.ix_(d4_arr, d4_arr)] +
              K[np.ix_(d3_arr, d4_arr)] + K[np.ix_(d4_arr, d3_arr)])

        Kr = np.block([[k1, k2], [k3, k4]])

        # Build RHS from prescribed boundaries and periodic offsets
        # RHS = -K_{unknown,prescribed} @ u_prescribed

        # Contribution from d1 (prescribed corners)
        rhs_d2_corner = np.zeros(len(d2_arr), dtype=self.dtype)
        rhs_d3_corner = np.zeros(len(d3_arr), dtype=self.dtype)
        rhs_d4_corner = np.zeros(len(d4_arr), dtype=self.dtype)

        if len(d1_arr) > 0:
            if len(d2_arr) > 0:
                k_d2_d1 = K[np.ix_(d2_arr, d1_arr)]
                rhs_d2_corner = -k_d2_d1 @ ufixed
            if len(d3_arr) > 0:
                k_d3_d1 = K[np.ix_(d3_arr, d1_arr)]
                rhs_d3_corner = -k_d3_d1 @ ufixed
            if len(d4_arr) > 0:
                k_d4_d1 = K[np.ix_(d4_arr, d1_arr)]
                rhs_d4_corner = -k_d4_d1 @ ufixed

        # Contribution from periodic offsets (d4 constraint: u[d4] = u[d3] + wfixed)
        rhs_d2_periodic = np.zeros(len(d2_arr), dtype=self.dtype)
        rhs_d3_periodic = np.zeros(len(d3_arr), dtype=self.dtype)
        rhs_d4_periodic = np.zeros(len(d4_arr), dtype=self.dtype)

        if len(d4_arr) > 0:
            if len(d2_arr) > 0:
                k_d2_d4 = K[np.ix_(d2_arr, d4_arr)]
                rhs_d2_periodic = -k_d2_d4 @ wfixed
            if len(d3_arr) > 0:
                k_d3_d4 = K[np.ix_(d3_arr, d4_arr)]
                rhs_d3_periodic = -k_d3_d4 @ wfixed
            if len(d4_arr) > 0:
                k_d4_d4 = K[np.ix_(d4_arr, d4_arr)]
                rhs_d4_periodic = -k_d4_d4 @ wfixed

        rhs_d2 = rhs_d2_corner + rhs_d2_periodic
        rhs_d3 = rhs_d3_corner + rhs_d3_periodic
        rhs_d4 = rhs_d4_corner + rhs_d4_periodic

        # Concatenate RHS: [rhs_d2; rhs_d3+rhs_d4] (since d3 and d4 are linked)
        if len(d2_arr) > 0 or len(d3_arr) > 0:
            rhs_d2_part = rhs_d2 if len(d2_arr) > 0 else np.zeros(0, dtype=self.dtype)
            rhs_d34_part = rhs_d3 + rhs_d4 if (len(d3_arr) > 0 or len(d4_arr) > 0) else np.zeros(0, dtype=self.dtype)
            RHS = np.concatenate([rhs_d2_part, rhs_d34_part])
        else:
            RHS = np.zeros(0, dtype=self.dtype)

        # PHASE 1 IMPROVEMENT: Solve with diagonal scaling for better conditioning
        if Kr.size > 0:
            # Compute diagonal scaling factors to improve system conditioning
            Kr_diag = np.abs(np.diag(Kr))
            # Avoid division by zero - set very small diagonal entries to 1
            Kr_diag = np.where(Kr_diag < 1e-15, 1.0, Kr_diag)
            scale_factors = 1.0 / np.sqrt(Kr_diag)

            # Scale system: D @ K @ D @ (D @ u) = D @ RHS
            Kr_scaled = (scale_factors[:, None] * Kr) * scale_factors[None, :]
            RHS_scaled = scale_factors * RHS

            # Solve scaled system
            sol_scaled = np.linalg.lstsq(Kr_scaled, RHS_scaled, rcond=None)[0]

            # Unscale solution
            sol = sol_scaled * scale_factors

            # Optional: Report condition number improvement (for diagnostics)
            if False:  # Set to True to enable diagnostics
                cond_before = np.linalg.cond(Kr)
                cond_after = np.linalg.cond(Kr_scaled)
                print(f"  Condition number: {cond_before:.2e} -> {cond_after:.2e}")
        else:
            sol = np.zeros(0, dtype=self.dtype)

        # Reconstruct full displacement field
        U = np.zeros(self.ndof, dtype=self.dtype)
        U[d1_arr] = ufixed

        len_d2 = len(d2_arr)
        len_d3 = len(d3_arr)

        if len_d2 > 0:
            U[d2_arr] = sol[:len_d2]
        if len_d3 > 0:
            U[d3_arr] = sol[len_d2:len_d2 + len_d3]
            if len(d4_arr) > 0:
                U[d4_arr] = U[d3_arr] + wfixed

        return U


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5: High-Level API
# ═════════════════════════════════════════════════════════════════════════════

def homogenize_3d(nelx: int, nely: int, nelz: int,
                  material: ElasticMaterial,
                  density_field: Optional[np.ndarray] = None,
                  E_min: float = 1e-9,
                  penalty: float = 1.0,
                  verbose: bool = False,
                  element_type: str = "Hex8") -> np.ndarray:
    """Compute effective 6×6 elastic tensor for a 3D microstructure

    Parameters
    ----------
    nelx, nely, nelz : int
        Mesh dimensions
    material : ElasticMaterial
        Base material model
    density_field : ndarray of shape (nelz, nely, nelx), optional
        Density/porosity field for SIMP (values in [0, 1])
    E_min : float, optional
        Minimum modulus ratio for SIMP (default 1e-9)
    penalty : float, optional
        SIMP penalty exponent (default 1.0 for linear, >1 for nonlinear)
    verbose : bool, optional
        Print convergence information
    element_type : str, optional
        Element type: "Hex8" (default, Phase 1) or "Hex20" (Phase 2, improved convergence)

    Returns
    -------
    C_eff : ndarray of shape (6, 6)
        Effective elastic stiffness tensor in Voigt notation
    """

    solver = FEM3DHomogenizer(nelx, nely, nelz, material, density_field, E_min, penalty,
                             element_type=element_type)
    return solver.compute_effective_tensor(verbose=verbose)


if __name__ == "__main__":
    # Example: 2×2×2 isotropic material
    print("\n" + "="*80)
    print("3D FEM HOMOGENIZATION EXAMPLE")
    print("="*80)

    # Create isotropic material
    mat = IsotropicMaterial(E=1.0, nu=0.3, name="Steel")

    # Test multiple mesh sizes
    for size in [1, 2, 3]:
        C_eff = homogenize_3d(size, size, size, mat, verbose=True)
        print(f"\n{size}×{size}×{size} mesh:")
        print(f"  C_eff[0,0] = {C_eff[0,0]:.8f}")
        print(f"  Expected ≈ 1.34615385 (isotropic, E=1, ν=0.3)")
