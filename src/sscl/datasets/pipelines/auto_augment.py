#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
"""
from mmcls.datasets.pipelines import auto_augment
# from mmcls.datasets.pipelines.auto_augment import Rotate, random_negative
from mmcls.datasets.builder import PIPELINES
import numpy as np
import cv2
from mmcv.image.geometric import cv2_border_modes, cv2_interp_codes

"""
mmcls里面这个rotate调用的imrotate一直报warning，所以重写一下。
"""
@PIPELINES.register_module(force=True)
class Rotate(auto_augment.Rotate):
    def imrotate_without_warning(
        self,
        img: np.ndarray,
        angle: float,
        center = None,
        scale: float = 1.0,
        border_value: int = 0,
        interpolation: str = 'bilinear',
        auto_bound: bool = False,
        border_mode: str = 'constant'
    ) -> np.ndarray:

        if center is not None and auto_bound:
            raise ValueError('`auto_bound` conflicts with `center`')
        h, w = img.shape[:2]
        if center is None:
            center = ((w - 1) * 0.5, (h - 1) * 0.5)
        assert isinstance(center, tuple)

        matrix = cv2.getRotationMatrix2D(center, -angle, scale)
        if auto_bound:
            cos = np.abs(matrix[0, 0])
            sin = np.abs(matrix[0, 1])
            new_w = h * sin + w * cos
            new_h = h * cos + w * sin
            matrix[0, 2] += (new_w - w) * 0.5
            matrix[1, 2] += (new_h - h) * 0.5
            w = int(np.round(new_w))
            h = int(np.round(new_h))
        rotated = cv2.warpAffine(
            img,
            matrix, (w, h),
            flags=cv2_interp_codes[interpolation],
            borderMode=cv2_border_modes[border_mode],
            borderValue=border_value)
        return rotated

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        angle = auto_augment.random_negative(self.angle, self.random_negative_prob)
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img_rotated = self.imrotate_without_warning(
                img,
                angle,
                center=self.center,
                scale=self.scale,
                border_value=self.pad_val,
                interpolation=self.interpolation)
            results[key] = img_rotated.astype(img.dtype)
        return results



if __name__ == "__main__":
    exit(0)
