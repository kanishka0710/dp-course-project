import numpy as np

def get_ula_response(N, az, d_over_lambda=0.5):
    n = np.arange(N)
    a = np.exp(-1j * np.pi * 2 * d_over_lambda * n[:, None] * np.sin(az))
    return a

def get_ula_gain(N, w, az, d_over_lambda=0.5):
    a = get_ula_response(N, az, d_over_lambda)
    g = np.conj(w).T @ a
    return g

def get_upa_response(My, Mz, az, el, dy_over_lambda=0.5, dz_over_lambda=0.5):
    """
    Returns steering matrix for a My x Mz UPA (y/z plane).
    Element (m, n) occupies index n*My + m in the output rows.
    az, el: azimuth and elevation in radians, shape (K,) or scalar.
    Returns shape (My*Mz, K).
    """
    m = np.arange(My)
    n = np.arange(Mz)
    # (My, K): phase along y-axis depends on az and el
    a_y = np.exp(-1j * 2 * np.pi * dy_over_lambda * m[:, None] * np.sin(az) * np.cos(el))
    # (Mz, K): phase along z-axis depends only on el
    a_z = np.exp(-1j * 2 * np.pi * dz_over_lambda * n[:, None] * np.sin(el))
    # Kronecker product per angle: (Mz, My, K) -> (My*Mz, K)
    return (a_z[:, None, :] * a_y[None, :, :]).reshape(My * Mz, -1)

def get_upa_gain(My, Mz, w, az, el, dy_over_lambda=0.5, dz_over_lambda=0.5):
    a = get_upa_response(My, Mz, az, el, dy_over_lambda, dz_over_lambda)
    g = np.conj(w).T @ a
    return g

def get_dft_codebook_ula(N, B):
    k = np.arange(B).reshape(B, 1)   # beam index
    n = np.arange(N).reshape(1, N)   # element index
    w = (1/np.sqrt(N)) * np.exp(-1j * 2*np.pi * k * n / B)
    return w

def get_dft_codebook_upa(My, Mz, By, Bz):
    """
    Returns a (By*Bz, My*Mz) UPA DFT codebook as the Kronecker product of two
    ULA codebooks. Beam (ky, kz) is at row ky*Bz + kz; element (m, n) is at
    column n*My + m, consistent with get_upa_response.
    """
    W_y = get_dft_codebook_ula(My, By)   # (By, My)
    W_z = get_dft_codebook_ula(Mz, Bz)   # (Bz, Mz)
    # (By, 1, 1, My) * (1, Bz, Mz, 1) -> (By, Bz, Mz, My) -> (By*Bz, Mz*My)
    return (W_y[:, None, None, :] * W_z[None, :, :, None]).reshape(By * Bz, Mz * My)
