import colorsys
import io

import numpy as np
from PIL import Image as pil_image
from sklearn.metrics import pairwise_distances_argmin


def load_img(bytes_io, mode='HSV'):
    return pil_image.open(bytes_io).convert(mode)


def save_img(img_array, format_='PNG'):
    image = pil_image.fromarray(
        np.uint8(img_array),
        'RGB'
    )
    with io.BytesIO() as bytes_io:
        image.save(bytes_io, format=format_)
        return bytes_io.getvalue()


def hex_to_rgb(hex_):
    hex_ = hex_.lstrip('#')
    hlen = len(hex_)
    return tuple(int(
        hex_[i:i+int(hlen/3)], 16
    ) for i in range(0, hlen, int(hlen/3)))


def rgb_to_hsv(rgb):
    max_rgb = 255.0
    r, g, b = tuple(rgb)
    return colorsys.rgb_to_hsv(
        r / max_rgb,
        g / max_rgb,
        b / max_rgb
    )


class ColorDesignFormatter:
    def __init__(
        self,
        color_design_format,
        metric='euclidean',
        mode='HSV'
    ):
        self.hsv_format_array = self._preprocess_color_design_format_as_hsv(
            color_design_format
        )
        self.rgb_format_array = self._preprocess_color_design_format_as_rgb(
            color_design_format
        )
        self.metric = metric
        self.mode = mode

    def _preprocess_img_array(self, img_array):
        w, h, d = tuple(img_array.shape)
        reshaped_array = np.reshape(img_array, (w * h, d))
        regularized_array = np.copy(reshaped_array) / 255.0
        if self.mode == 'HSV':
            return self._preprocess_hsv_array(regularized_array)
        else:
            return regularized_array - 0.5

    def _preprocess_color_design_format_as_rgb(self, format_):
        return np.array(
            [hex_to_rgb(f) for f in format_],
            dtype=np.uint8
        )
    
    def _preprocess_color_design_format_as_hsv(self, format_):
        rgb_array = self._preprocess_color_design_format_as_rgb(
            format_
        )
        hsv_array = np.array(
            [rgb_to_hsv(rgb) for rgb in rgb_array],
            dtype=np.float64
        )
        return self._preprocess_hsv_array(hsv_array)

    def _preprocess_hsv_array(self, hsv_array):
        array = np.empty((hsv_array.shape[0], 4))

        array[:, 2:4] = hsv_array[:, 2:4] * 2 - 1

        rad_hue_array = np.deg2rad(hsv_array[:, 0] * 360)
        array[:, 0] = np.sin(rad_hue_array)
        array[:, 1] = np.cos(rad_hue_array)
        return array

    def format(self, img):
        raw_img_array = np.array(img, dtype=np.float64)
        w, h, d = tuple(raw_img_array.shape)
        assert d == 3
        img_array = self._preprocess_img_array(raw_img_array)

        format_array = self.hsv_format_array if (
            self.mode == 'HSV'
        ) else (self.rgb_format_array / 255.0) - 0.5
        color_labels = pairwise_distances_argmin(
            format_array,
            img_array,
            metric=self.metric,
            axis=0
        )

        return self._recreate_image(
            color_labels, w, h, d
        )

    def _recreate_image(self, labels, w, h, d):
        """Recreate the (compressed) image from the code book & labels"""
        image = np.zeros((w, h, d))
        label_idx = 0
        for i in range(w):
            for j in range(h):
                image[i][j] = self.rgb_format_array[labels[label_idx]]
                label_idx += 1
        return image
