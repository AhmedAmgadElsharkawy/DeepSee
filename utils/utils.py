import numpy as np
def gaussian_kernel(size=3, sigma=1):
        kernel = np.zeros((size, size), dtype=np.float32)
        center = size // 2
        sum = 0

        # Gaussian values
        for i in range(size):
            for j in range(size):
                x, y = i - center, j - center
                kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2)) * 1 / (2 * np.pi * sigma**2)
                sum += kernel[i, j]

        return kernel / sum


def pad_image(image, pad_size,pading_type='constant', pad_value=0):
    if pading_type=='constant':

        if image.ndim == 2:
            padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size)), mode='constant', constant_values=pad_value)
        elif image.ndim == 3:
            padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='constant', constant_values=pad_value)
        else:
            raise ValueError("Unsupported image shape")
    elif pading_type=="reflect":
        padded_image = np.pad(image, pad_size, mode="reflect")


    return padded_image