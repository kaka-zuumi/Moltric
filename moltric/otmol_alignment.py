#!/usr/bin/env python3
from typing import List, Tuple, Union, Optional
import numpy as np
import ot
from scipy.spatial import distance_matrix
from scipy.sparse import csr_array
from scipy.sparse.csgraph import floyd_warshall
import os


def otmol_alignment(
        X_A, 
        X_B, 
        T_A, 
        T_B, 
        B_A: np.ndarray = None,
        B_B: np.ndarray = None,
        method: str = 'fGW', 
        alpha_list: list = None, 
        molecule_sizes: List[int] = None,
        reflection: bool = False,
        cst_D: float = 0.,
        minimize_mismatched_edges: bool = False,
        save_path: str = None,
        return_BCI: bool = False,
        ) -> Tuple[np.ndarray, float, float]:
    """Compute alignment between two molecules or (molecular complexes) with optimal transport.

    Parameters
    ----------
    X_A : numpy.ndarray
        Coordinates of molecule A.
    X_B : numpy.ndarray
        Coordinates of molecule B.
    T_A : array_like
        Atom labels of molecule A.
    T_B : array_like
        Atom labels of molecule B.
    method : list of str
        Optimal transport method to use, by default ['fgw', 'emd'].
    alpha_list : list
        List of alpha values to try for fGW or fsGW solver, by default None.
    molecule_sizes : List[int], optional
        Sizes of molecules, by default None. 
        It is only used when two structures contain multiple molecules,
        and molecules are ordered in the same way.
    reflection : bool, optional
        Whether to allow reflection in the Kabsch algorithm, by default False.
    cst_D : float, optional
        D = (1-cst_D)*Euclidean + cst_D*Geodesic, by default 0. If the user wants to reduce bond inconsistency, set cst_D to a value close to 1.
    minimize_mismatched_edges : bool, optional
        Whether to prioritize minimizing mismatched edges in the alignment, by default False.
    save_path : str, optional
        Path to save the aligned molecule, by default None. The atoms in the aligned molecule will be reordered.
    return_BCI: bool, optional
        Whether to return the BCI value (in range [0, 1]), by default False. Only use when minimize_mismatched_edges is False.

    Returns
    -------
    assignment : numpy.ndarray
        Optimal assignment between molecules.
    rmsd : float
        Best RMSD value.
    alpha : float
        Best alpha value.
    BCI : float
        BCI value. If minimize_mismatched_edges or return_BCI is True, the BCI value will be returned.
        A mismatched edge is an edge that is present in A but not in B.
        BCI is defined as the number of mismatched edges divided by the total number of edges in A.
    """
    if molecule_sizes is not None:
        T_B_original = T_B.copy()
        T_A, T_B = add_molecule_indices(T_A, T_B, molecule_sizes)
    C = cost_matrix(T_A = T_A, T_B = T_B, k = np.inf)

    C_finite = C.copy()
    C_finite[C_finite == np.inf] = 1e12
    if cst_D < 1e-5:
        D_A = distance_matrix(X_A, X_A)
        D_B = distance_matrix(X_B, X_B)
        D_A, D_B = D_A/D_A.max(), D_B/D_A.max()
    elif B_A is not None and B_B is not None:
        Euc_A, Euc_B = distance_matrix(X_A, X_A), distance_matrix(X_B, X_B)
        Geo_A, Geo_B = geodesic_distance(X_A, B_A), geodesic_distance(X_B, B_B)
        Euc_A, Euc_B = Euc_A/Euc_A.max(), Euc_B/Euc_A.max()
        if Geo_A.max() == np.inf:
            Geo_A[Geo_A == np.inf] = Euc_A[Geo_A == np.inf]
            Geo_B[Geo_B == np.inf] = Euc_B[Geo_B == np.inf]
        Geo_A, Geo_B = Geo_A/Geo_A.max(), Geo_B/Geo_A.max()
        D_A = (1-cst_D)*Euc_A + cst_D*Geo_A
        D_B = (1-cst_D)*Euc_B + cst_D*Geo_B
    rmsd_best = 1e10
    mismatched_bond_best = 1e10
    assignment_list = []
    assignment_set = set()
    assignment_best = None
    alpha_best = None
    _alpha_list = []
    X_B_aligned_best = None
    for alpha in alpha_list:
        # Fused Gromov-Wasserstein
        P = ot.gromov.fused_gromov_wasserstein(C_finite, D_A, D_B, alpha=alpha, symmetric=True)
        assignment = np.argmax(P, axis=1)
        if is_permutation(T_A=T_A, T_B=T_B, perm=assignment, case='single') and tuple(assignment) not in assignment_set:
            assignment_list.append(assignment)
            _alpha_list.append(alpha) # stores the alpha value for each assignment
            assignment_set.add(tuple(assignment))

    if minimize_mismatched_edges:  
        n = len(T_A)
        for i, assignment in enumerate(assignment_list):
            mismatched_bond = mismatched_bond_counter(B_A, B_B, assignment, n, n)
            if mismatched_bond < mismatched_bond_best:
                rmsd_best = 1e10 # reset rmsd_best
                mismatched_bond_best = mismatched_bond
                X_B_aligned, _, _ = kabsch(X_A, X_B, permutation_to_matrix(assignment), reflection)
                rmsd = root_mean_square_deviation(X_A, X_B_aligned[assignment])
                if rmsd < rmsd_best:
                    rmsd_best = rmsd
                    assignment_best = assignment
                    alpha_best = _alpha_list[i]
                    X_B_aligned_best = X_B_aligned[assignment]
            if mismatched_bond == mismatched_bond_best:
                X_B_aligned, _, _ = kabsch(X_A, X_B, permutation_to_matrix(assignment), reflection)
                rmsd = root_mean_square_deviation(X_A, X_B_aligned[assignment])
                if rmsd < rmsd_best:
                    rmsd_best = rmsd
                    assignment_best = assignment     
                    alpha_best = _alpha_list[i]
                    X_B_aligned_best = X_B_aligned[assignment]
        if assignment_best is None:
            print('No valid assignment found') 
            return None, None, None, None
        BCI = mismatched_bond_counter(B_A, B_B, assignment_best, n, n, only_A_bonds=True)[0]/np.sum(B_A)*2
        return assignment_best, rmsd_best, alpha_best, BCI
    else:
        for i, assignment in enumerate(assignment_list):
            X_B_aligned, _, _ = kabsch(X_A, X_B, permutation_to_matrix(assignment), reflection)
            rmsd = root_mean_square_deviation(X_A, X_B_aligned[assignment])
            if rmsd < rmsd_best:
                rmsd_best = rmsd
                assignment_best = assignment
                alpha_best = _alpha_list[i]
                X_B_aligned_best = X_B_aligned[assignment]
        if assignment_best is None:
            print('No valid assignment found')
            return None, None, None
        if return_BCI:
            BCI = mismatched_bond_counter(B_A, B_B, assignment_best, len(T_A), len(T_B), only_A_bonds=True)[0]/np.sum(B_A)*2
            return assignment_best, rmsd_best, alpha_best, BCI
        else:
            return assignment_best, rmsd_best, alpha_best









def kabsch(
    X1: np.ndarray,
    X2: np.ndarray,
    P: np.ndarray,
    reflection: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kabsch algorithm. 
    Perform rigid body rotation (including reflection if reflection is set to True), 
    and translation to align molecules.

    Parameters
    ----------
    X1 : numpy.ndarray
        Coordinates of molecule 1 (reference) as an n x 3 array.
    X2 : numpy.ndarray
        Coordinates of molecule 2 (to be aligned) as an m x 3 array.
    P : numpy.ndarray
        A permutation matrix describing the atom assignment between molecules, 
        where matrix[i, j] = 1 if assignment[i] = j.
    reflection : bool, optional
        Whether to allow reflection, by default False. 

    Returns
    -------
    X2_aligned : numpy.ndarray
        Aligned coordinates of molecule 2.
    R : numpy.ndarray
        Rotation matrix.
    t : numpy.ndarray
        Translation vector.
    """
    total_weight = P.sum()
    
    # Compute weights for each point
    w1 = P.sum(axis=1)  # weights for X1
    w2 = P.sum(axis=0)  # weights for X2
    
    # Compute weighted centroids
    mu1 = np.sum(X1 * w1[:, None], axis=0) / total_weight
    mu2 = np.sum(X2 * w2[:, None], axis=0) / total_weight
    
    # Center the point clouds
    X1_centered = X1 - mu1
    X2_centered = X2 - mu2
    
    # vectorized calculation: H = X1_centered^T P X2_centered 
    H = X1_centered.T @ P @ X2_centered
    
    # Compute SVD of H
    U, _, Vt = np.linalg.svd(H)
    
    if not reflection and np.linalg.det(U @ Vt) < 0: # Ensure R is a proper rotation matrix (det(R)=1)
        Vt[-1, :] *= -1
        R = U @ Vt
    else:
        R = U @ Vt
        
    # Compute the translation
    t = mu1 - R @ mu2
    
    # Transform X2
    X2_aligned = (R @ X2.T).T + t
    
    return X2_aligned, R, t


def perturbation_before_gw(
        X_A: np.ndarray, 
        X_B: np.ndarray, 
        p_list: list = [1], 
        n_trials: int = 100, 
        scale: float = 0.1, 
        ) -> List[np.ndarray]:
    """Find various suboptimal transport plans between clusters of atoms.

    When calculating the distance matrix, Gaussian noise is added to the coordinates
    to generate various suboptimal transport plans.
    We first do a GW, then do a Kantorovich (ot.emd) on the aligned coordinates from GW.

    Parameters
    ----------
    X_A : numpy.ndarray
        Coordinates of cluster A.
    X_B : numpy.ndarray
        Coordinates of cluster B.
    p_list : list, 
        Power of the distance matrix, by default [1].
    n_trials : int, optional
        Number of trials to run, by default 100.
    scale : float, optional
        Standard deviation of the Gaussian noise, by default 0.1.

    Returns
    -------
    list_perms : List[numpy.ndarray]
        List of permutations.
    """
    unique_perms = set()
    list_perms = []
    # It seems that when the number of atoms is 2, the GW algorithm always returns the permutation [0,1] regardless of the input.
    # So we handle this case separately.
    if len(X_A) == 2:
        list_perms = [np.array([0,1]), np.array([1,0])]
        return list_perms
  
    for i in range(n_trials):
        X_A_perturbed, X_B_perturbed = add_perturbation(X_A, scale, random_state = i), add_perturbation(X_B, scale, random_state = i)
        Euc_A, Euc_B = distance_matrix(X_A_perturbed, X_A_perturbed), distance_matrix(X_B_perturbed, X_B_perturbed)
        for p in p_list:
            D_A = Euc_A**p
            D_B = Euc_B**p
            D_A, D_B = D_A/D_A.max(), D_B/D_A.max()
            P = ot.gromov.gromov_wasserstein(D_A, D_B, symmetric=True)
            perm = np.argmax(P, axis=1)
            if not is_permutation(perm=perm):
                continue
            X_B_aligned, _, _ = kabsch(X_A, X_B, permutation_to_matrix(perm), reflection=False)
            D_ot = distance_matrix(X_A, X_B_aligned)**2
            P = ot.emd([], [], D_ot/D_ot.max())
            perm = np.argmax(P, axis=1)
            if not is_permutation(perm=perm):
                continue
            perm_tuple = tuple(perm)
            if perm_tuple not in unique_perms:
                unique_perms.add(perm_tuple)
                list_perms.append(perm)                
    return list_perms




#####################################################################################
#####################################################################################

# Utility Functions


def root_mean_square_deviation(
        X: np.ndarray, 
        Y: np.ndarray
        ) -> float:
    """Compute the Root Mean Square Deviation (RMSD) between two sets of points.

    Parameters
    ----------
    X : np.ndarray
        First set of points as an ``n`` x ``3`` array.
    Y : np.ndarray
        Second set of points as an ``n`` x ``3`` array.

    Returns
    -------
    float
        The RMSD between the two sets of points.
    """
    return np.sqrt(np.mean(np.sum((X - Y) ** 2, axis=1)))


def cost_matrix(
        X_A: np.ndarray = None, 
        X_B: np.ndarray = None, 
        T_A: np.ndarray = None, 
        T_B: np.ndarray = None, 
        k: float = 1e12, 
        ) -> np.ndarray:
    """Create a cost matrix for T_A and T_B.

    T_A and T_B are the atom labels.
    If X_A and X_B are provided, creates a cost matrix where the cost of atoms having the same label
    is the Euclidean distance between the atoms and the cost of atoms having different labels is a constant k 
    (can be set to infinity).
    If X_A and X_B are not provided, creates a cost matrix where the cost of atoms having the same label
    is 0 and the cost of atoms having different labels is a constant k (can be set to infinity).
    If multiple_molecules_block_size is provided, creates a block diagonal matrix where off-diagonal blocks
    are filled with k.

    Parameters
    ----------
    X_A : numpy.ndarray
        Array of coordinates for molecule A.
    X_B : numpy.ndarray
        Array of coordinates for molecule B.
    T_A : numpy.ndarray
        Array of atom labels for molecule A.
    T_B : numpy.ndarray
        Array of atom labels for molecule B.
    k : float, optional
        Cost of mismatching atoms, by default 1e11.

    Returns
    -------
    numpy.ndarray
        Cost matrix as numpy array.
    """
    n = len(T_A)
    m = len(T_B)
    C = np.full((n, m), k)
    
    for i in range(n):
        for j in range(m):
            if T_A[i] == T_B[j]:
                if X_A is not None and X_B is not None:
                    C[i, j] = np.linalg.norm(X_A[i] - X_B[j])
                else:
                    C[i, j] = 0
    return C


def compare_labels(
        list1, 
        list2
        ) -> List[Tuple[int, str, str]]:
    """Compare two arrays of atom labels and return the indices and labels where they differ.

    Parameters
    ----------
    list1 : array_like
        Array of atom labels.
    list2 : array_like
        Array of atom labels.

    Returns
    -------
    List[Tuple[int, str, str]]
        A list of tuples, where each tuple contains the index and the differing labels
        in the format (index, label_from_list1, label_from_list2).
    """
    differences = []
    # Compare elements up to the length of the shorter list
    for i in range(len(list1)):
        if list1[i] != list2[i]:
            differences.append((i, list1[i], list2[i]))

    return differences


def parse_molecule_pairs(
        file_path: str, 
        mol_type: str = 'water cluster'
        ) -> List[List[str]]:
    """Parses list file in ArbAlign data folder.

    Parameters:
    ----------
    file_path : str
        Path to the file containing molecule pairs.

    Returns:
    -------
    list of lists
        A list where each element is a pair [molA, molB].
    """
    molecule_pairs = []
    with open(file_path, 'r') as file:
        if mol_type == 'water cluster' or mol_type == 'S1':
            for line in file:
                line = line.strip()  # Remove any leading/trailing whitespace
                if line:  # Skip empty lines
                    # Some lines are like "molA_molB_2", and some are like "molA_molB"
                    molecule_pairs.append([line.split('_')[0], line.split('_')[1]])
        if mol_type == 'FGG':
            next(file)  # Skip the first line
            for line in file:
                line = line.strip()
                if line:
                    molA, molB = line.split('-')
                    molecule_pairs.append([molA, molB])
    return molecule_pairs


def permutation_to_matrix(permutation, n: int = None, m: int = None) -> np.ndarray:
    """Convert a permutation list to a permutation matrix.

    Parameters
    ----------
    permutation : array_like
        A list or array representing the permutation.
        For example, [2, 0, 1] means index 0 maps to 2, index 1 maps to 0, etc.
    n : int, optional
        The number of rows in the matrix.
    m : int, optional
        The number of columns in the matrix.

    Returns
    -------
    numpy.ndarray
        A permutation matrix where matrix[i, j] = 1 if permutation[i] = j, 0 otherwise.
    """
    if n is None and m is None:
        n = len(permutation)
        matrix = np.zeros((n, n), dtype=int)
    else:
        matrix = np.zeros((n, m), dtype=int)

    for i, j in enumerate(permutation):
        matrix[i, j] = 1
    return matrix


def is_permutation(
        T_A: np.ndarray = None,
        T_B: np.ndarray = None,
        perm: np.ndarray = None, 
        case: str = None, 
        n_atoms: int = None
        ) -> bool:
    """Check if the given array is a permutation, with optional special cases.
    If T_A and T_B are provided, check if T_A == T_B[perm].
    When case = 'single', this function can also be used for a matching between A and B 
    when the number of atoms in A is less than the number of atoms in B.

    Parameters
    ----------
    T_A : numpy.ndarray
        Array like. A 1D array of atom labels.
    T_B : numpy.ndarray
        Array like. A 1D array of atom labels.
    perm : numpy.ndarray
        Array like. A 1D array of integers.
    case : str, optional
        Special case to check:
            - None: Basic permutation check (all elements unique).
            - 'single': labels are matched.
            - 'molecule cluster': molecule cluster permutation (groups of n_atoms).
    n_atoms : int, optional
        Only used when case is 'molecule cluster'. The number of atoms in a molecule.

    Returns
    -------
    bool
        True if the array satisfies the permutation condition, False otherwise.
    """
    if case is None:
        return len(np.unique(perm)) == len(perm)
    if case == 'single':
        return len(np.unique(perm)) == len(perm) and np.array_equal(T_A, T_B[perm])
    if case == 'molecule cluster':
        return is_molecule_cluster_permutation(T_A = T_A, T_B = T_B, perm = perm, n_atoms = n_atoms)


def is_molecule_cluster_permutation(
        T_A: np.ndarray = None, 
        T_B: np.ndarray = None, 
        perm: np.ndarray = None, 
        n_atoms: int = 3
        ) -> bool:
    """Check if the given array perm is a permutation that satisfies the condition:
    After grouping every n_atoms = k integers, sorting each group in ascending order
    results in [min(group), min(group)+1, ..., min(group)+k-1], and min(group) is a multiple of k.

    Parameters
    ----------
    T_A : array like
        A 1D numpy array of atom labels.
    T_B : array like
        A 1D numpy array of atom labels.
    perm : array like
        A 1D numpy array of integers.
    n_atoms : int
        The number of atoms in a molecule.

    Returns
    -------
    bool
        True if all groups satisfy the condition, False otherwise.
    """
    # check label consistency
    if T_A is not None and T_B is not None:
        if not np.array_equal(T_A, T_B[perm]):
            return False
    # basic check
    if len(np.unique(perm)) != len(perm):
        return False
    n = len(perm)
    if n % n_atoms != 0:
        raise ValueError("The length of the array must be a multiple of {}.".format(n_atoms))
    
    blocks = [perm[i:i+n_atoms] for i in range(0, n, n_atoms)]  # Group every n_atoms integers
    for block in blocks:
        sorted_block = sorted(block)
        if sorted_block[0] % n_atoms != 0:
            return False
        if not np.array_equal(sorted_block, np.arange(sorted_block[0], sorted_block[0] + n_atoms)):
            return False
    return True


def add_perturbation(
        X: np.ndarray, 
        noise_scale: float = None, 
        random_state: int = None
        ) -> np.ndarray:
    """Add Gaussian noise to coordinates.
    
    Parameters
    ----------
    X : numpy.ndarray
        The 3D coordinates as an n x 3 array.
    noise_scale : float, optional
        The standard deviation of the Gaussian noise to add.
        If None, it will be set to 1/10 of the shortest distance between coordinates.
    random_state : int or numpy.random.RandomState
        Seed for the random number generator or RandomState instance.
        
    Returns
    -------
    numpy.ndarray
        The perturbed coordinates.
    """
    if noise_scale is None:
        # Calculate pairwise distances between all points
        distances = distance_matrix(X, X)
        # Set diagonal to infinity to ignore self-distances
        np.fill_diagonal(distances, np.inf)
        # Find minimum distance and set noise scale to 1/10 of that
        min_distance = np.min(distances)
        noise_scale = min_distance / 10.0
    
    # Create random state
    rng = np.random.RandomState(random_state)
    noise = rng.normal(0, noise_scale, X.shape)
    return X + noise


def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    """Normalize a matrix by dividing by its maximal non-infinity value.
    
    Parameters
    ----------
    matrix : numpy.ndarray
        Input matrix that may contain infinity values.
        
    Returns
    -------
    numpy.ndarray
        Normalized matrix where non-infinity values are divided by the maximum non-infinity value.
        Infinity values remain unchanged.
    """
    # Create a mask for non-infinity values
    finite_mask = np.isfinite(matrix)
    
    if not np.any(finite_mask):
        return matrix  # Return original matrix if all values are infinity
        
    # Find maximum non-infinity value
    max_val = np.max(matrix[finite_mask])
    
    if max_val == 0:
        return matrix  # Return original matrix if max value is 0
        
    # Create normalized matrix
    normalized = matrix.copy()
    normalized[finite_mask] = normalized[finite_mask] / max_val
    
    return normalized


def resolve_sinkhorn_conflicts(P: np.ndarray) -> List[np.ndarray]:
    """Resolve conflicts in a Sinkhorn transport plan P.

    Parameters
    ----------
    P : numpy.ndarray
        The Sinkhorn transport plan to resolve conflicts in.

    Returns
    -------
    list of numpy.ndarray
        A list of resolved Sinkhorn transport plans.
    """
    n = P.shape[0]
    assignment = np.full(n, -1)
    used = np.zeros(n, dtype=bool)
    
    # First pass: assign unambiguous cases
    for i in range(n):
        row = P[i]
        max_val = np.max(row)
        max_indices = np.where(row == max_val)[0]
        if len(max_indices) == 1 and not used[max_indices[0]]:
            assignment[i] = max_indices[0]
            used[max_indices[0]] = True
    
    if np.sum(assignment == -1) > 2:
        return None
    else:
        res = []
        unassigned = np.where(assignment == -1)[0]
        unused = np.where(~used)[0]

        tmp = assignment.copy()
        tmp[unassigned] = unused
        res.append(tmp)

        tmp = assignment.copy()
        tmp[unassigned] = unused[::-1]
        res.append(tmp)

        return res


def add_molecule_indices(
        T_A: List[str],
        T_B: List[str],
        molecule_sizes: List[int]
        ) -> Tuple[List[str], List[str]]:
    """Add indices to labels based on molecule sizes.
    
    For each molecule, adds an index to all labels in that molecule's range.
    For example, if molecule_sizes = [3, 2], then:
    - Labels 0-2 get index 0
    - Labels 3-4 get index 1
    
    Parameters
    ----------
    T_A : List[str]
        List of labels for molecule A
    T_B : List[str]
        List of labels for molecule B
    molecule_sizes : List[int]
        List of sizes for each molecule. Sum must equal len(T_A) = len(T_B)
        
    Returns
    -------
    T_A_with_indices : np.ndarray
        Array of labels for molecule A with indices
    T_B_with_indices : np.ndarray
        Array of labels for molecule B with indices
        
    Raises
    ------
    ValueError
        If sum(molecule_sizes) != len(T_A) or len(T_A) != len(T_B)
    """
    if sum(molecule_sizes) != len(T_A) or len(T_A) != len(T_B):
        raise ValueError("sum(molecule_sizes) must equal len(T_A) = len(T_B)")
        
    T_A_with_indices = []
    T_B_with_indices = []
    
    current_pos = 0
    for i, size in enumerate(molecule_sizes):
        # Add index to all labels in this molecule's range
        for j in range(size):
            T_A_with_indices.append(f"{T_A[current_pos + j]}_{i}")
            T_B_with_indices.append(f"{T_B[current_pos + j]}_{i}")
        current_pos += size
        
    return np.array(T_A_with_indices, dtype=str), np.array(T_B_with_indices, dtype=str)


def mismatched_bond_counter(B_A, B_B, assignment, n, m, only_A_bonds=False):
    """
    Count the number of mismatched bonds between two sets of bonds.
    If only_A_bonds is True, only count the number of bonds in A that are not in B.

    Parameters
    ----------
    B_A : numpy.ndarray
        Array of shape (n, n) containing the bond information for molecule A.
    B_B : numpy.ndarray
        Array of shape (m, m) containing the bond information for molecule B.
    assignment : numpy.ndarray
        Array of shape (n,) containing the assignment of atoms in B to atoms in A.
    n : int
        The number of atoms in molecule A.
    m : int
        The number of atoms in molecule B.
    only_A_bonds : bool, optional
        If True, only count the number of bonds in A that are not in B.

    Returns
    -------
    int
        The number of mismatched bonds. 
        When only_A_bonds is True, the number of bonds in A that are not in B.
    list
        When only_A_bonds is True, a list of tuples, where each tuple contains the indices of the mismatched bonds.
    """
    i, j = np.triu_indices(n, k=1)  # k=1 to exclude diagonal
    # Create assignment matrix        
    P = np.zeros((n, m), dtype=int)
    P[np.arange(n), assignment] = 1
    # Compare bonds using matrix operations
    B_B_permuted = P @ B_B @ P.T
    if only_A_bonds:
        mask1 = B_A[i, j] == 1 
        mask2 = B_B_permuted[i, j] == 0
        return np.sum(mask1 & mask2), list(zip(i[mask1 & mask2], j[mask1 & mask2]))
    else:
        return np.sum(B_A[i, j] != B_B_permuted[i, j])



#####################################################################################
#####################################################################################

# Distance Processing


ATOMIC_NAME = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    15: "P",
    16: "S",
    17: "Cl",
    35: "Br"
}

ATOMIC_COLOR = {
    "H": "silver",
    "C": "black",
    "N": "blue",
    "O": "red",
    "F": "green",
    "P": "orange",
    "S": "yellow",
    "Cl": "limegreen",
    "Br": "salmon"
}

ATOMIC_SIZE = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Br": 1.20
}

ATOMIC_PROPERTIES = {
    "H": {"en": 2.20, "vdw": 1.10, "cov": 0.32},   # Hydrogen (H)
    "C": {"en": 2.55, "vdw": 1.70, "cov": 0.76},   # Carbon (C)
    "N": {"en": 3.04, "vdw": 1.55, "cov": 0.71},   # Nitrogen (N)
    "O": {"en": 3.44, "vdw": 1.52, "cov": 0.66},   # Oxygen (O)
    "F": {"en": 3.98, "vdw": 1.47, "cov": 0.64},   # Fluorine (F)
    "P": {"en": 2.19, "vdw": 1.80, "cov": 1.06},  # Phosphorus (P)
    "S": {"en": 2.58, "vdw": 1.80, "cov": 1.02},  # Sulfur (S)
    "Cl": {"en": 3.16, "vdw": 1.75, "cov": 0.99},  # Chlorine (Cl)
    "Br": {"en": 2.96, "vdw": 1.85, "cov": 1.14},  # Bromine (Br)
}


def geodesic_distance(
    X,
    B
):
    """
    return the geodesic distance between all pairs of atoms in the molecule.

    Parameters
    ----------
    X: numpy.ndarray
        Coordinates of the molecule.
    B: numpy.ndarray
        Adjacency matrix of the graph.

    Returns
    -------
    numpy.ndarray
        Geodesic distance between all pairs of atoms in the molecule.
    """
    dists = distance_matrix(X, X)   
    graph = np.where(B, dists, 0)
    graph = csr_array(graph)
    geodesic = floyd_warshall(graph, directed=False)
    return geodesic


