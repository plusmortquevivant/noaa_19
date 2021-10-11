"""Provides a change of coordinates"""
from typing import Union, Sequence
import numpy as np
from .types import Epoch, Matrix33
from .precessions import Precession, PrecessionGOST
from .nutations import Nutation1980
from .rotation import Rotation

def matrix_multiple_batch(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Вычисление построчного произведения

    :param A: массив матриц(N матриц размерности k * l)
    :param B: массив матриц(N матриц размерности l * m)
    :return: Для каждой пары матриц результат A * B. N матриц размера k * m
    """
    return np.einsum('ijk,ikm->ijm', A, B)

class EarthRotatingCS:
    """provides a change of coordinates"""

    def __init__(self, rotation: Rotation, precession: Precession, nutation: Nutation1980):
        self.rotation = rotation
        self.precession = precession
        self.nutation = nutation

    def get_matrix(self, epoch: Union[Epoch, np.ndarray]) -> Matrix33:
        """
        Create rotation matrix
        :param epoch: date
        :return: matrix
        """
        if isinstance(epoch, Sequence):
            epoch = np.array(epoch)
        rot = self.rotation.get_matrix(epoch)
        prec = self.precession.get_matrix(epoch)
        nut = self.nutation.get_matrix(epoch)
        if isinstance(epoch, Epoch):
            res = rot @ nut @ prec
        else:
            res = matrix_multiple_batch(matrix_multiple_batch(rot, nut), prec)
        return res


def build_default_rotating_cs():
    """build default rotating cs"""
    nut = Nutation1980()
    prec = PrecessionGOST()
    rot = Rotation(nut)
    return EarthRotatingCS(rot, prec, nut)
