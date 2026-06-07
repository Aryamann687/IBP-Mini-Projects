import matplotlib.pyplot as plt
plt.set_cmap('gray')

from time import time
import numpy as np
import cv2
from multiprocessing import Pool, cpu_count


# -----------------------------------
# Standard DFT
# -----------------------------------
def stdFT(x):

    N = len(x)
    X = np.zeros((N), complex)

    for u in range(N):
        for k in range(N):

            wk = np.e ** (-2j * np.pi * u * k / N)

            X[u] += x[k] * wk

    return X


# -----------------------------------
# Optimized Radix-2 FFT
# -----------------------------------
def fft_radix2_optimized(x):

    def bit_reverse_indices(n):

        bits = int(np.log2(n))

        indices = np.arange(n)

        reversed_indices = np.array([
            int('{:0{width}b}'.format(i, width=bits)[::-1], 2)
            for i in indices
        ])

        return reversed_indices

    N = len(x)

    if N & (N - 1) != 0:
        raise ValueError("Length must be power of 2")

    # Bit reversal
    x = np.asarray(x, dtype=complex)

    x = x[bit_reverse_indices(N)]

    # Butterfly stages
    m = 2

    while m <= N:

        W_m = np.exp(-2j * np.pi / m)

        half_m = m // 2

        for k in range(0, N, m):

            W = 1.0

            for j in range(half_m):

                t = W * x[k + j + half_m]

                u = x[k + j]

                x[k + j] = u + t

                x[k + j + half_m] = u - t

                W *= W_m

        m *= 2

    return x


# -----------------------------------
# Sequential 2D FFT
# -----------------------------------
def fft2_custom(img, fn):

    img = np.asarray(img, dtype=np.complex128)

    # FFT along rows
    F = np.apply_along_axis(fn, axis=1, arr=img)

    # FFT along columns
    F = np.apply_along_axis(fn, axis=0, arr=F)

    return F


# -----------------------------------
# Parallel 2D FFT
# -----------------------------------
def fft2_parallel(img):

    img = np.asarray(img, dtype=np.complex128)

    # Parallel FFT on rows
    with Pool(cpu_count()) as p:

        Fr = p.map(fft_radix2_optimized, img)

    Fr = np.array(Fr)

    # Parallel FFT on columns
    with Pool(cpu_count()) as p:

        F = p.map(fft_radix2_optimized, Fr.T)

    # Correct transpose
    F = np.array(F).T

    return F


# -----------------------------------
# Main Program
# -----------------------------------
if __name__ == '__main__':

    # Load image
    img = cv2.imread('landscape.jpeg', 0)

    if img is None:

        print("Image not found!")

        exit()

    # Resize image
    img = cv2.resize(img, (512, 512))

    # -----------------------------------
    # Sequential FFT
    # -----------------------------------
    startTime = time()

    imfft_seq = fft2_custom(img, fft_radix2_optimized)

    endTime = time()

    print(f'Sequential FFT Time: {endTime - startTime:.4f} seconds')

    # -----------------------------------
    # Parallel FFT
    # -----------------------------------
    startTime = time()

    imfft_parallel = fft2_parallel(img)

    endTime = time()

    print(f'Parallel FFT Time: {endTime - startTime:.4f} seconds')

    # -----------------------------------
    # Inverse FFT Reconstruction
    # -----------------------------------
    imgBack = np.fft.ifft2(imfft_parallel)

    plt.imshow(imgBack.real)

    plt.title('Image after inverse FFT')

    plt.show(block=True)
