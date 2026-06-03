from array_utils import *
from channel import *
from scipy.linalg import null_space
import math
import matplotlib.pyplot as plt

import pickle


def main():
    print("Hello from dp-course-project!")
    # N_t = M_t = N_r = M_r = 8
    # H = single_reflection_si_upa(N_t, M_t, N_r, M_r, 0, 0)
    # U, S, Vh = np.linalg.svd(H, full_matrices=False)

    N = 16
    N_r = N_t = N
    theta_t = 30
    theta_r = -20

    # LOS channels for UL/DL 
    h_t = get_ula_response(N, theta_t*np.pi/180).flatten()
    h_r = get_ula_response(N, theta_r*np.pi/180).flatten()

    f = h_t.conj()
    w = h_r.conj()

    f = f / np.linalg.norm(f)
    w = w / np.linalg.norm(w)

    H_SI = draw_static_si_channel_ula(N_t, N_r, 10, 32, 1)

    h_t_eff = w @ H_SI

    B = null_space(h_t_eff.reshape(1,-1))
    P = B @ np.linalg.pinv(B.conj().T @ B) @ B.conj().T

    f_bfc = P @ f

    print(np.abs(w @ H_SI @ f)**2)
    print(np.abs(w @ H_SI @ f_bfc)**2)

    H_SI_new = H_SI + single_reflection_si_ula(N_t, N_r, np.pi/7)
    print(np.abs(w @ H_SI_new @ f_bfc)**2)

    ul_power = np.abs(h_r @ w)**2
    dl_power = np.abs(h_t @ f_bfc)**2


    si_powers = []

    ch_hist = []
    
    for t in range(300):
        x = t - 150  # center the pass near the origin
        y = 10
        z = 0
        r = math.sqrt(x**2 + y**2 + z**2)       # radial distance
        theta = math.acos(z / r)                  # polar angle from z-axis (inclination)
        phi = math.atan2(y, x)                    # azimuthal angle in x-y plane

        H_dynamic = 1e6 *  r**-4 * single_reflection_si_ula(N_t, N_r, phi)

        H_SI_new = H_SI + H_dynamic

        H_SI_new = N * H_SI_new / np.linalg.norm(H_SI_new)

        print(np.linalg.norm(H_SI_new))

        si_powers.append(np.abs(w @ H_SI_new @ f_bfc)**2)

        ch_hist.append(H_SI_new)

    si_powers = np.array(si_powers)
    plt.plot(10*np.log10(si_powers))
    plt.show()

    plt.plot(np.log2(1 + ul_power/si_powers) + np.log2(1 + dl_power))
    plt.xlabel("time (ms)")
    plt.ylabel("SSE (bits/s/Hz)")
    # plt.title("Changing SSE as one object passes a full-duplex base station")
    plt.savefig("sse.pdf")
    plt.show()

    with open("H_one_reflection.pkl", "wb") as file:
        pickle.dump(ch_hist, file)

    breakpoint()


if __name__ == "__main__":
    main()
