"""
Нахождение трехмерной точки по 2 углам с системе отсчета GEO
"""
import numpy as np
from coordinate_system.earth_models import EarthEllipsoidModel


def get_cartesian_geo_from_geo_angles(
        latitude,  # широта
        longitude,  # долгота
        model: EarthEllipsoidModel,  # модель Земли
) -> np.ndarray:
    """
    Функция, переводящая географические координаты в декартовы в системе GEO

    :param latitudes: широта в диапазоне [- Pi /2; Pi /2] -
    от южного полюса (-Pi / 2) к северному (Pi / 2)
    :type latitude: float

    :param longitudes: долгота в диапазоне [Pi; Pi] -
    0 - Гринвический меридиан. Восточные долготы - положительные.
    :type longitude: float

    :param model: Модель земной формы - Spherical, wsg84, ...
    :type model: Model

    :return: декартовы координаты в системе GEO -
    Oz - от южного полюса к северному
    Ox - от центра к точке latitude = 0 longitude = 0
    (пересечение Гринвического меридиана и плоскости экватора)
    Oy - до правой тройки

    """
    x_cor = model.eq_rad * np.cos(latitude) * np.cos(longitude)
    y_cor = model.eq_rad * np.cos(latitude) * np.sin(longitude)
    z_cor = model.pol_rad * np.sin(latitude)
    return np.transpose(np.array([x_cor, y_cor, z_cor]))
