import numpy as np

from array_utils import get_ula_response, get_upa_response


def single_reflection_si_ula(N_t, N_r, theta):
    A_t = get_ula_response(N_t, theta).flatten()
    A_r = get_ula_response(N_r, theta).flatten()
    H = np.outer(A_r, A_t.conj())
    return H

def draw_ula_spherical_wave_channel(rx_dim, tx_dim, distance_lambda):
    """
    Generate a spherical wave MIMO channel between two parallel uniform linear arrays.

    TX array lies along Y-axis at x=0.
    RX array lies along Y-axis at x=distance_lambda.
    Half-wavelength element spacing.

    Args:
        rx_dim:           Number of RX elements
        tx_dim:           Number of TX elements
        distance_lambda:  TX-to-RX separation in wavelengths

    Returns:
        h: Complex channel matrix of shape (rx_dim, tx_dim),
           normalized so ||h||_F^2 = rx_dim * tx_dim
    """
    tx_pos = np.arange(tx_dim) / 2          # shape (tx_dim,)
    rx_pos = np.arange(rx_dim) / 2          # shape (rx_dim,)

    # Euclidean distance between each RX-TX pair
    dy = rx_pos[:, np.newaxis] - tx_pos[np.newaxis, :]   # shape (rx_dim, tx_dim)
    r = np.sqrt(distance_lambda**2 + dy**2)

    h = np.exp(-1j * 2 * np.pi * r) / r

    rho = np.sqrt(rx_dim * tx_dim) / np.linalg.norm(h, 'fro')
    h = h * rho

    return h


def draw_static_si_channel_ula(N_t, N_r, sep, n_reflectors, kappa):

    H_reflectors = []
    for i in range(n_reflectors):
        theta = np.random.uniform(-np.pi/2, np.pi/2)
        alpha = np.random.uniform(0,1) * np.exp(1j * np.random.uniform(0, 2*np.pi))
        H_reflectors.append(alpha * single_reflection_si_ula(N_t,N_r,theta))

    H_reflectors = np.mean(np.stack(H_reflectors, axis=0), axis=0)  # shape (N_r, N_t)
    H_reflectors = np.sqrt(N_t*N_r) * H_reflectors / np.linalg.norm(H_reflectors)

    # spherical wave
    H_sw = draw_ula_spherical_wave_channel(N_r, N_t, sep)

    H = kappa * H_reflectors + (1 - kappa) * H_sw
    return H


def single_reflection_si_upa(N_t, M_t, N_r, M_r, theta, phi):
    A_t = get_upa_response(N_t, M_t, theta, phi).flatten()
    A_r = get_upa_response(N_r, M_r, theta, phi).flatten()
    H = np.outer(A_r, A_t.conj())
    return H


def draw_upa_spherical_wave_channel(rx_rows, rx_cols, tx_rows, tx_cols, distance_lambda):
    """
    Generate a spherical wave MIMO channel between two uniform planar arrays.

    Both arrays lie in the YZ-plane; propagation is along the X-axis.
    Element spacing is half-wavelength in both Y and Z directions.

    Args:
        rx_rows:          Number of RX elements along Z-axis
        rx_cols:          Number of RX elements along Y-axis
        tx_rows:          Number of TX elements along Z-axis
        tx_cols:          Number of TX elements along Y-axis
        distance_lambda:  TX-to-RX separation in wavelengths

    Returns:
        h: Complex channel matrix of shape (rx_rows*rx_cols, tx_rows*tx_cols),
           normalized so ||h||_F^2 = rx_rows*rx_cols*tx_rows*tx_cols
    """
    # --- Build element positions (in wavelengths) ---
    # TX array centred at x=0, RX array centred at x=distance_lambda
    # Half-wavelength spacing in Y and Z

    tx_y = np.arange(tx_cols) / 2                   # shape (tx_cols,)
    tx_z = np.arange(tx_rows) / 2                   # shape (tx_rows,)
    tx_yy, tx_zz = np.meshgrid(tx_y, tx_z)          # shape (tx_rows, tx_cols)
    tx_pos = np.stack([
        np.zeros_like(tx_yy),                        # x = 0
        tx_yy,                                       # y
        tx_zz                                        # z
    ], axis=-1).reshape(-1, 3)                       # shape (tx_rows*tx_cols, 3)

    rx_y = np.arange(rx_cols) / 2                   # shape (rx_cols,)
    rx_z = np.arange(rx_rows) / 2                   # shape (rx_rows,)
    rx_yy, rx_zz = np.meshgrid(rx_y, rx_z)          # shape (rx_rows, rx_cols)
    rx_pos = np.stack([
        np.full_like(rx_yy, distance_lambda),        # x = distance_lambda
        rx_yy,                                       # y
        rx_zz                                        # z
    ], axis=-1).reshape(-1, 3)                       # shape (rx_rows*rx_cols, 3)

    # --- Compute pairwise 3D distances ---
    # diff[i, j] = rx_pos[i] - tx_pos[j], shape (N_rx, N_tx, 3)
    diff = rx_pos[:, np.newaxis, :] - tx_pos[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=-1)               # shape (N_rx, N_tx)

    # --- Spherical wave channel (free-space Green's function) ---
    h = np.exp(-1j * 2 * np.pi * r) / r

    # --- Frobenius-norm power normalisation (matches your MATLAB convention) ---
    n_rx = rx_rows * rx_cols
    n_tx = tx_rows * tx_cols
    rho = np.sqrt(n_rx * n_tx) / np.linalg.norm(h, 'fro')
    h = h * rho

    return h

