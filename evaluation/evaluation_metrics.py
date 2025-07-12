import torch
import numpy as np
import warnings
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from tqdm.auto import tqdm

# Global flag for EMD warning to avoid repeated messages
_EMD_NOT_IMPL_WARNED = False


def emd_approx(sample: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Approximates Earth Mover's Distance (EMD).
    Currently, this function returns zeros and prints a warning as EMD is not
    fully implemented due to GPU compatibility. Users can implement their own
    EMD logic here if needed.

    Args:
        sample (torch.Tensor): A batch of sample point clouds.
                                Shape: (batch_size, num_points, point_dim)
        ref (torch.Tensor): A batch of reference point clouds.
                            Shape: (batch_size, num_points, point_dim)

    Returns:
        torch.Tensor: Approximated EMD values (currently zeros).
                      Shape: (batch_size,)
    """
    global _EMD_NOT_IMPL_WARNED
    emd = torch.zeros(sample.size(0), device=sample.device)
    if not _EMD_NOT_IMPL_WARNED:
        _EMD_NOT_IMPL_WARNED = True
        print("\n\n[WARNING]")
        print("  * EMD is not fully implemented due to potential GPU compatibility issues.")
        print("  * All EMD values are currently set to zero by default.")
        print("  * You may implement your own EMD in the `emd_approx` function.")
        print("\n")
    return emd


def distChamfer(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the Chamfer Distance between two sets of point clouds.

    Args:
        a (torch.Tensor): First set of point clouds.
                          Shape: (batch_size, num_points_a, point_dim)
        b (torch.Tensor): Second set of point clouds.
                          Shape: (batch_size, num_points_b, point_dim)

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing two tensors:
                                           - The minimum squared distances from points in 'a' to 'b'.
                                             Shape: (batch_size, num_points_a)
                                           - The minimum squared distances from points in 'b' to 'a'.
                                             Shape: (batch_size, num_points_b)
    """
    x, y = a, b
    bs, num_points_x, _ = x.size()

    # Calculate squared Euclidean distances using matrix multiplication
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))

    # Extract diagonal elements for squared norms
    diag_ind = torch.arange(0, num_points_x, device=a.device).long()
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy) # This seems incorrect if y has a different num_points_y

    # Corrected ry for variable number of points in y
    num_points_y = y.size(1)
    diag_ind_y = torch.arange(0, num_points_y, device=b.device).long()
    ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(yy)

    # Compute pairwise squared distances
    # P_ij = ||x_i - y_j||^2 = ||x_i||^2 + ||y_j||^2 - 2 * x_i @ y_j^T
    P = rx.transpose(2, 1) + ry - 2 * zz

    # Find the minimum distance from each point in 'a' to 'b' and vice-versa
    min_dist_a_to_b = P.min(1)[0]
    min_dist_b_to_a = P.min(2)[0]
    return min_dist_a_to_b, min_dist_b_to_a


def EMD_CD(sample_pcs: torch.Tensor, ref_pcs: torch.Tensor, batch_size: int, reduced: bool = True) -> dict:
    """
    Computes Chamfer Distance (CD) and approximated Earth Mover's Distance (EMD)
    between two sets of point clouds.

    Args:
        sample_pcs (torch.Tensor): Sample point clouds. Shape: (N_sample, num_points, point_dim)
        ref_pcs (torch.Tensor): Reference point clouds. Shape: (N_ref, num_points, point_dim)
        batch_size (int): Batch size for processing.
        reduced (bool, optional): If True, returns the mean of the metrics.
                                  If False, returns all individual metric values. Defaults to True.

    Returns:
        dict: A dictionary containing 'MMD-CD' and 'MMD-EMD' (mean or all values).
    """
    n_sample = sample_pcs.shape[0]
    n_ref = ref_pcs.shape[0]
    assert n_sample == n_ref, f"Number of sample and reference point clouds must be equal. REF:{n_ref} SMP:{n_sample}"

    cd_lst = []
    emd_lst = []

    for b_start in tqdm(range(0, n_sample, batch_size), desc="EMD-CD"):
        b_end = min(n_sample, b_start + batch_size)
        sample_batch = sample_pcs[b_start:b_end]
        ref_batch = ref_pcs[b_start:b_end]

        dl, dr = distChamfer(sample_batch, ref_batch)
        cd_lst.append(dl.mean(dim=1) + dr.mean(dim=1))

        emd_batch = emd_approx(sample_batch, ref_batch)
        emd_lst.append(emd_batch)

    cd_combined = torch.cat(cd_lst)
    emd_combined = torch.cat(emd_lst)

    results = {
        "MMD-CD": cd_combined.mean() if reduced else cd_combined,
        "MMD-EMD": emd_combined.mean() if reduced else emd_combined,
    }
    return results


def _pairwise_EMD_CD_(sample_pcs: torch.Tensor, ref_pcs: torch.Tensor, batch_size: int, verbose: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes pairwise Chamfer Distance (CD) and approximated Earth Mover's Distance (EMD)
    between all sample and reference point clouds.

    Args:
        sample_pcs (torch.Tensor): Sample point clouds. Shape: (N_sample, num_points, point_dim)
        ref_pcs (torch.Tensor): Reference point clouds. Shape: (N_ref, num_points, point_dim)
        batch_size (int): Batch size for processing reference point clouds for each sample.
        verbose (bool, optional): If True, displays a progress bar. Defaults to True.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing two tensors:
                                           - all_cd: Pairwise Chamfer Distances. Shape: (N_sample, N_ref)
                                           - all_emd: Pairwise approximated EMDs. Shape: (N_sample, N_ref)
    """
    n_sample = sample_pcs.shape[0]
    n_ref = ref_pcs.shape[0]
    all_cd = []
    all_emd = []

    iterator = range(n_sample)
    if verbose:
        iterator = tqdm(iterator, desc="Pairwise EMD-CD")

    for sample_idx in iterator:
        sample_batch = sample_pcs[sample_idx]

        cd_for_current_sample = []
        emd_for_current_sample = []

        for ref_b_start in range(0, n_ref, batch_size):
            ref_b_end = min(n_ref, ref_b_start + batch_size)
            ref_batch = ref_pcs[ref_b_start:ref_b_end]

            current_batch_size_ref = ref_batch.size(0)
            point_dim = ref_batch.size(2)

            # Expand the single sample point cloud to match the batch size of reference points
            sample_batch_expanded = sample_batch.view(1, -1, point_dim).expand(
                current_batch_size_ref, -1, -1
            ).contiguous()

            dl, dr = distChamfer(sample_batch_expanded, ref_batch)
            cd = (dl.mean(dim=1) + dr.mean(dim=1)).view(1, -1)
            cd_for_current_sample.append(cd)

            emd_batch = emd_approx(sample_batch_expanded, ref_batch)
            emd_for_current_sample.append(emd_batch.view(1, -1))

        all_cd.append(torch.cat(cd_for_current_sample, dim=1))
        all_emd.append(torch.cat(emd_for_current_sample, dim=1))

    all_cd = torch.cat(all_cd, dim=0)  # N_sample, N_ref
    all_emd = torch.cat(all_emd, dim=0)  # N_sample, N_ref

    return all_cd, all_emd


def knn(Mxx: torch.Tensor, Mxy: torch.Tensor, Myy: torch.Tensor, k: int, sqrt: bool = False) -> dict:
    """
    Performs k-Nearest Neighbors classification and returns accuracy metrics.
    This is typically used for 1-NN classification in FID-like metrics.

    Args:
        Mxx (torch.Tensor): Pairwise distances within the first set (e.g., ref-ref).
                            Shape: (N_ref, N_ref)
        Mxy (torch.Tensor): Pairwise distances between the first and second set (e.g., ref-sample).
                            Shape: (N_ref, N_sample)
        Myy (torch.Tensor): Pairwise distances within the second set (e.g., sample-sample).
                            Shape: (N_sample, N_sample)
        k (int): Number of nearest neighbors to consider.
        sqrt (bool, optional): If True, applies square root to distances. Defaults to False.

    Returns:
        dict: A dictionary containing precision, recall, accuracy, true positives,
              false positives, false negatives, and true negatives.
    """
    n0 = Mxx.size(0)
    n1 = Myy.size(0)
    
    # Create labels: 1 for first set (reference), 0 for second set (sample)
    labels = torch.cat((torch.ones(n0), torch.zeros(n1))).to(Mxx.device)

    # Combine distance matrices into a single matrix for KNN
    M = torch.cat(
        [torch.cat((Mxx, Mxy), 1), torch.cat((Mxy.transpose(0, 1), Myy), 1)], 0
    )
    
    if sqrt:
        M = M.abs().sqrt()

    # Set diagonal elements to infinity to exclude self-loops for neighbors
    INFINITY = float("inf")
    M_plus_diag_inf = M + torch.diag(INFINITY * torch.ones(n0 + n1, device=Mxx.device))
    
    # Find k nearest neighbors for each point
    _, idx = M_plus_diag_inf.topk(k, dim=0, largest=False) # Changed to largest=False for smallest distances

    # Count how many of the k neighbors belong to the same class as the point itself
    # A neighbor is considered 'same class' if its label matches the 'predicted' label based on neighbors
    # For KNN classification, we check if more than k/2 neighbors are from the *same* class as the point being classified.
    # Here, `label.index_select(0, idx[i])` gets the labels of the neighbors.
    # Summing them up and comparing to k/2 essentially counts how many neighbors are '1' (from the first set).
    # If count >= k/2, it predicts '1', otherwise '0'.
    
    count = torch.zeros(n0 + n1, device=Mxx.device)
    for i in range(k):
        count += labels.index_select(0, idx[i])
        
    # Predict based on majority vote (more than k/2 neighbors from the "1" class)
    predictions = (count >= (float(k) / 2)).float()

    # Calculate statistics
    s = {
        "tp": (predictions * labels).sum(),  # True Positives: Predicted 1, Actual 1
        "fp": (predictions * (1 - labels)).sum(),  # False Positives: Predicted 1, Actual 0
        "fn": ((1 - predictions) * labels).sum(),  # False Negatives: Predicted 0, Actual 1
        "tn": ((1 - predictions) * (1 - labels)).sum(),  # True Negatives: Predicted 0, Actual 0
    }

    # Calculate precision, recall, and accuracy
    s.update(
        {
            "precision": s["tp"] / (s["tp"] + s["fp"] + 1e-10),
            "recall": s["tp"] / (s["tp"] + s["fn"] + 1e-10),
            "acc_t": s["tp"] / (s["tp"] + s["fn"] + 1e-10),  # Accuracy for true labels (class 1)
            "acc_f": s["tn"] / (s["tn"] + s["fp"] + 1e-10),  # Accuracy for false labels (class 0)
            "acc": torch.eq(labels, predictions).float().mean(),  # Overall accuracy
        }
    )
    return s


def lgan_mmd_cov(all_dist: torch.Tensor) -> dict:
    """
    Calculates metrics similar to Maximum Mean Discrepancy (MMD) and Coverage (COV)
    based on pairwise distances, inspired by point cloud generation evaluation.

    Args:
        all_dist (torch.Tensor): A matrix of pairwise distances, e.g., M_rs_cd.t() from samples to refs.
                                 Shape: (N_sample, N_ref)

    Returns:
        dict: A dictionary containing 'lgan_mmd', 'lgan_cov', and 'lgan_mmd_smp'.
    """
    n_sample, n_ref = all_dist.size(0), all_dist.size(1)

    # Minimum distance from each sample to any reference point
    min_val_from_smp, min_idx = torch.min(all_dist, dim=1) # min_idx tells which ref point is closest

    # Minimum distance from each reference point to any sample point
    min_val_from_ref, _ = torch.min(all_dist, dim=0)

    # MMD: Mean of the minimum distances from reference points to sample points
    mmd = min_val_from_ref.mean()

    # MMD (Sample): Mean of the minimum distances from sample points to reference points
    mmd_smp = min_val_from_smp.mean()

    # Coverage: Proportion of unique reference points that are the closest to at least one sample point
    unique_min_indices = min_idx.unique()
    cov = float(unique_min_indices.size(0)) / float(n_ref)
    cov = torch.tensor(cov, device=all_dist.device) # Ensure it's a tensor on the same device

    # For debugging/information
    print(f"Unique min indices (sample to ref): {unique_min_indices.numel()} out of {n_ref}")
    print(f"MMD: {mmd.item():.4f}, MMD (Sample): {mmd_smp.item():.4f}, Coverage: {cov.item():.4f}")

    return {
        "lgan_mmd": mmd,
        "lgan_cov": cov,
        "lgan_mmd_smp": mmd_smp,
    }


def lgan_mmd_cov_match(all_dist: torch.Tensor) -> tuple[dict, torch.Tensor]:
    """
    Calculates MMD, Coverage, and returns the indices of the closest reference points
    for each sample point. This function is similar to `lgan_mmd_cov` but also
    returns the matching indices.

    Args:
        all_dist (torch.Tensor): A matrix of pairwise distances. Shape: (N_sample, N_ref)

    Returns:
        tuple[dict, torch.Tensor]: A tuple containing:
                                   - dict: 'lgan_mmd', 'lgan_cov', and 'lgan_mmd_smp'.
                                   - torch.Tensor: Indices of the closest reference points for each sample.
                                                   Shape: (N_sample,)
    """
    n_sample, n_ref = all_dist.size(0), all_dist.size(1)

    min_val_from_smp, min_idx = torch.min(all_dist, dim=1)
    min_val_from_ref, _ = torch.min(all_dist, dim=0)

    mmd = min_val_from_ref.mean()
    mmd_smp = min_val_from_smp.mean()

    cov = float(min_idx.unique().size(0)) / float(n_ref)
    cov = torch.tensor(cov, device=all_dist.device)

    results = {
        "lgan_mmd": mmd,
        "lgan_cov": cov,
        "lgan_mmd_smp": mmd_smp,
    }
    return results, min_idx.view(-1)


def compute_all_metrics(sample_pcs: torch.Tensor, ref_pcs: torch.Tensor, batch_size: int) -> dict:
    """
    Computes a comprehensive set of metrics for evaluating point cloud generation models,
    including Chamfer Distance-based MMD/Coverage and 1-NN classification accuracy.

    Args:
        sample_pcs (torch.Tensor): Generated/sample point clouds.
                                   Shape: (N_sample, num_points, point_dim)
        ref_pcs (torch.Tensor): Reference/ground truth point clouds.
                                Shape: (N_ref, num_points, point_dim)
        batch_size (int): Batch size for computations.

    Returns:
        dict: A dictionary containing all computed metrics.
    """
    results = {}

    print(f"Shape of sample_pcs: {sample_pcs.shape}")
    print(f"Shape of ref_pcs: {ref_pcs.shape}")
    print(f"Batch size: {batch_size}")

    # --- Pairwise EMD and CD (Sample to Reference) ---
    print("\nComputing Pairwise EMD-CD (Reference to Sample)")
    # M_rs_cd: Ref to Sample CD distances (N_ref x N_sample)
    # M_rs_emd: Ref to Sample EMD distances (N_ref x N_sample)
    M_rs_cd, M_rs_emd = _pairwise_EMD_CD_(ref_pcs, sample_pcs, batch_size)

    # Calculate MMD and Coverage based on CD from Reference to Sample
    res_cd_ref_to_sample = lgan_mmd_cov(M_rs_cd)
    results.update({f"{k}-CD": v for k, v in res_cd_ref_to_sample.items()})

    print("MMD-CD and Coverage Results (Reference to Sample):")
    for k, v in results.items():
        print(f"  [{k}] {v.item():.8f}")

    # --- Pairwise CD (Reference-Reference and Sample-Sample) for 1-NN ---
    print("\nComputing Pairwise CD (Reference-Reference and Sample-Sample) for 1-NN")
    M_rr_cd, _ = _pairwise_EMD_CD_(ref_pcs, ref_pcs, batch_size, verbose=False)
    M_ss_cd, _ = _pairwise_EMD_CD_(sample_pcs, sample_pcs, batch_size, verbose=False)

    # --- 1-NN Classification based on CD ---
    print("\nComputing 1-NN Classification Accuracy (CD)")
    one_nn_cd_res = knn(M_rr_cd, M_rs_cd, M_ss_cd, k=1, sqrt=False)
    results.update(
        {f"1-NN-CD-{k}": v for k, v in one_nn_cd_res.items() if "acc" in k or "precision" in k or "recall" in k}
    )
    print("1-NN CD Results:")
    for k, v in one_nn_cd_res.items():
        if "acc" in k or "precision" in k or "recall" in k:
            print(f"  [{k}] {v.item():.8f}")

    return results


## Jensen-Shannon Divergence (JSD) Metrics
"""
These functions are designed to compute the Jensen-Shannon Divergence between point cloud sets, primarily based on their occupancy grids.
"""

def unit_cube_grid_point_cloud(resolution: int, clip_sphere: bool = False) -> tuple[np.ndarray, float]:
    """
    Returns the center coordinates of each cell of a 3D grid with `resolution^3` cells,
    placed within the unit cube (from -0.5 to 0.5 in each dimension).

    Args:
        resolution (int): The resolution of the grid (number of cells along each dimension).
        clip_sphere (bool, optional): If True, only keeps cells whose centers are within
                                      a unit sphere (radius 0.5) centered at the origin.
                                      Defaults to False.

    Returns:
        tuple[np.ndarray, float]: A tuple containing:
                                  - np.ndarray: Coordinates of grid cell centers.
                                                Shape: (num_cells, 3)
                                  - float: The spacing between grid points.
    """
    # Create a grid where each dimension ranges from -0.5 to 0.5
    spacing = 1.0 / float(resolution - 1) if resolution > 1 else 0.0 # Handle resolution=1 case
    
    # Use broadcasting with `np.linspace` for efficiency
    x_coords = np.linspace(-0.5, 0.5, resolution, dtype=np.float32)
    y_coords = np.linspace(-0.5, 0.5, resolution, dtype=np.float32)
    z_coords = np.linspace(-0.5, 0.5, resolution, dtype=np.float32)

    # Create a meshgrid
    grid_x, grid_y, grid_z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    grid = np.stack([grid_x, grid_y, grid_z], axis=-1) # Shape: (res, res, res, 3)

    if clip_sphere:
        grid = grid.reshape(-1, 3)
        grid = grid[norm(grid, axis=1) <= 0.5 + 1e-6] # Add small epsilon for floating point issues

    return grid.reshape(-1, 3), spacing


def jsd_between_point_cloud_sets(sample_pcs: np.ndarray, ref_pcs: np.ndarray, resolution: int = 28) -> float:
    """
    Computes the Jensen-Shannon Divergence (JSD) between two sets of point clouds.
    This metric quantifies the similarity between the occupancy patterns of the point clouds
    when projected onto a 3D grid.

    Args:
        sample_pcs (np.ndarray): Sample point clouds. Shape: (num_samples, num_points, 3)
        ref_pcs (np.ndarray): Reference point clouds. Shape: (num_references, num_points, 3)
        resolution (int, optional): Grid resolution for occupancy calculation. Defaults to 28.

    Returns:
        float: The Jensen-Shannon Divergence value.
    """
    # Note: `in_unit_sphere` is effectively False due to its original usage with `entropy_of_occupancy_grid`
    # and its default behavior. The warning inside `entropy_of_occupancy_grid` addresses bounds.
    in_unit_sphere = False 

    _, sample_grid_counters = entropy_of_occupancy_grid(sample_pcs, resolution, in_unit_sphere)
    _, ref_grid_counters = entropy_of_occupancy_grid(ref_pcs, resolution, in_unit_sphere)
    
    return jensen_shannon_divergence(sample_grid_counters, ref_grid_counters)


def entropy_of_occupancy_grid(
    pclouds: np.ndarray, grid_resolution: int, in_sphere: bool = False, verbose: bool = True
) -> tuple[float, np.ndarray]:
    """
    Estimates the entropy of occupancy-grid activation patterns given a collection of point clouds.

    Args:
        pclouds (np.ndarray): Collection of point clouds. Shape: (num_pclouds, points_per_pcloud, 3)
        grid_resolution (int): Size of the occupancy grid along one dimension.
        in_sphere (bool, optional): If True, checks if point clouds are within a unit sphere.
                                    Defaults to False.
        verbose (bool, optional): If True, prints warnings. Defaults to True.

    Returns:
        tuple[float, np.ndarray]: A tuple containing:
                                  - float: The average entropy of activated grid cells.
                                  - np.ndarray: The count of how many times each grid cell was occupied.
    """
    epsilon = 1e-4
    bound = 0.5 + epsilon

    # Check if point clouds are within the unit cube
    if np.max(np.abs(pclouds)) > bound and verbose:
        warnings.warn("Point-clouds are not entirely within the unit cube (-0.5 to 0.5).")

    # Check if point clouds are within the unit sphere (if specified)
    if in_sphere and np.max(norm(pclouds, axis=2)) > bound and verbose:
        warnings.warn("Point-clouds are not entirely within the unit sphere (radius 0.5).")

    grid_coordinates, _ = unit_cube_grid_point_cloud(grid_resolution, in_sphere)
    grid_coordinates = grid_coordinates.reshape(-1, 3) # Ensure 2D for NearestNeighbors

    # Initialize counters for grid cell occupancy
    grid_counters = np.zeros(len(grid_coordinates), dtype=int)
    grid_bernoulli_rvars = np.zeros(len(grid_coordinates), dtype=int) # Counts how many PCs activate a cell

    # Use NearestNeighbors to find which grid cell each point falls into
    nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(grid_coordinates)

    for pc in tqdm(pclouds, desc="JSD"):
        # Find the closest grid cell for each point in the current point cloud
        _, indices = nn.kneighbors(pc)
        indices = np.squeeze(indices) # Convert to 1D array of indices

        # Increment counter for each point's closest grid cell
        for i in indices:
            grid_counters[i] += 1
        
        # Increment Bernoulli random variable counter only once per activated cell per point cloud
        # This counts how many *distinct* point clouds activate a given cell
        unique_indices = np.unique(indices)
        for i in unique_indices:
            grid_bernoulli_rvars[i] += 1

    # Calculate average entropy
    acc_entropy = 0.0
    n = float(len(pclouds)) # Total number of point clouds
    num_grid_cells = float(len(grid_counters))

    for g in grid_bernoulli_rvars:
        if g > 0:
            p = float(g) / n
            acc_entropy += entropy([p, 1.0 - p], base=2) # Using base 2 for bits

    return acc_entropy / num_grid_cells, grid_counters


def jensen_shannon_divergence(P: np.ndarray, Q: np.ndarray) -> float:
    """
    Computes the Jensen-Shannon Divergence (JSD) between two probability distributions.
    The JSD is a symmetric and smoothed version of the Kullback-Leibler Divergence (KLD).

    Args:
        P (np.ndarray): First probability distribution (or counts that can be normalized).
        Q (np.ndarray): Second probability distribution (or counts that can be normalized).

    Returns:
        float: The Jensen-Shannon Divergence value.

    Raises:
        ValueError: If input arrays contain negative values or have different sizes.
    """
    if np.any(P < 0) or np.any(Q < 0):
        raise ValueError("Input arrays must not contain negative values.")
    if P.shape != Q.shape:
        raise ValueError("Input arrays must have the same shape.")

    # Normalize to ensure they are valid probability distributions
    P_norm = P / np.sum(P)
    Q_norm = Q / np.sum(Q)

    # Compute the average distribution M
    M = 0.5 * (P_norm + Q_norm)

    # Calculate JSD using the formula: JSD(P||Q) = 0.5 * (KLD(P||M) + KLD(Q||M))
    # Using scipy.stats.entropy for KLD (which is equivalent to relative entropy)
    jsd_value = 0.5 * (entropy(P_norm, M, base=2) + entropy(Q_norm, M, base=2))

    # Original code had a redundant internal _jsdiv check, removed for clarity.
    # The scipy.stats.entropy with two arguments directly calculates KL divergence.

    return jsd_value


if __name__ == "__main__":
    # Example usage for Chamfer Distance and EMD (approximated)
    print("--- Running EMD_CD Example ---")
    # Generate some dummy data on CPU for demonstration
    a_cpu = torch.randn([16, 2048, 3])
    b_cpu = torch.randn([16, 2048, 3])

    # If CUDA is available, move tensors to GPU
    if torch.cuda.is_available():
        print("CUDA is available. Running on GPU.")
        a = a_cpu.cuda()
        b = b_cpu.cuda()
    else:
        print("CUDA not available. Running on CPU.")
        a = a_cpu
        b = b_cpu

    cd_emd_results = EMD_CD(a, b, batch_size=8)
    print("\nEMD_CD Results:")
    for metric_name, value in cd_emd_results.items():
        print(f"  {metric_name}: {value.item():.6f}")

    # Example usage for compute_all_metrics
    print("\n--- Running compute_all_metrics Example ---")
    # For compute_all_metrics, let's use slightly different shapes for clarity
    # N_sample = 32, N_ref = 32
    sample_point_clouds = torch.randn([32, 1024, 3]).to(a.device)
    ref_point_clouds = torch.randn([32, 1024, 3]).to(a.device)

    all_metrics_results = compute_all_metrics(sample_point_clouds, ref_point_clouds, batch_size=16)
    print("\nAll Metrics Results:")
    for metric_name, value in all_metrics_results.items():
        print(f"  {metric_name}: {value.item():.6f}")

    # Example usage for JSD metrics
    print("\n--- Running JSD Example ---")
    # Generate dummy NumPy data for JSD
    np_sample_pcs = np.random.rand(100, 500, 3) - 0.5 # 100 point clouds, 500 points each, in unit cube
    np_ref_pcs = np.random.rand(120, 500, 3) - 0.5

    jsd_val = jsd_between_point_cloud_sets(np_sample_pcs, np_ref_pcs, resolution=16)
    print(f"\nJensen-Shannon Divergence: {jsd_val:.6f}")