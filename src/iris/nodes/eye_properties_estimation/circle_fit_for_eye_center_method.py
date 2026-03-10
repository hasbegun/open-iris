from typing import Tuple

import numpy as np
from pydantic import Field

from iris.io.class_configs import Algorithm
from iris.io.dataclasses import EyeCenters, GeometryPolygons
from iris.io.errors import EyeCentersEstimationError


class CircleFitEyeCenterMethod(Algorithm):
    """Estimate pupil and iris centers using a robust circle-fitting approach.

    This algorithm estimates the center of a pupil or iris polygon using:

    1. an initial least-squares circle fit to all polygon points,
    2. trimming of outlier points based on radial residuals,
    3. a second circle fit on the retained inliers.

    LIMITATIONS:
    This method assumes that the pupil and iris contours are reasonably well
    approximated by circles in image space. It is therefore most appropriate
    when off-gaze and strong perspective distortion have already been filtered out.
    """

    class Parameters(Algorithm.Parameters):
        """Default parameters for circle-fit eye center algorithm."""

        inlier_ratio: float = Field(..., gt=0.0, lt=1.0)

    __parameters_type__ = Parameters

    def __init__(
        self,
        inlier_ratio: float = 0.9,
    ) -> None:
        """Assign parameters.

        Args:
            inlier_ratio (float, optional): Fraction of points retained after the
                initial fit based on the smallest radial residuals. Defaults to 0.9.
        """
        super().__init__(inlier_ratio=inlier_ratio)

    def run(self, geometries: GeometryPolygons) -> EyeCenters:
        """Estimate pupil and iris centers.

        Args:
            geometries (GeometryPolygons): Pupil and iris geometry polygons.

        Returns:
            EyeCenters: Estimated pupil and iris center coordinates.
        """
        pupil_center_x, pupil_center_y = self._calculate_circle_fit_center(geometries.pupil_array.astype(np.float64))
        iris_center_x, iris_center_y = self._calculate_circle_fit_center(geometries.iris_array.astype(np.float64))

        return EyeCenters(
            pupil_x=pupil_center_x,
            pupil_y=pupil_center_y,
            iris_x=iris_center_x,
            iris_y=iris_center_y,
        )

    def _calculate_circle_fit_center(self, polygon: np.ndarray) -> Tuple[float, float]:
        """Estimate the center of a polygon using fit -> trim outliers -> refit.

        Args:
            polygon (np.ndarray): Polygon points of shape (N, 2) representing a
                contour that is approximately circular.

        Raises:
            EyeCentersEstimationError: Raised if the polygon is invalid or if a
                valid circle cannot be fit.

        Returns:
            Tuple[float, float]: Estimated center coordinates (x, y).
        """
        if polygon.ndim != 2 or polygon.shape[1] != 2 or polygon.shape[0] < 3:
            raise EyeCentersEstimationError("Polygon must have shape (N, 2) with at least 3 points")

        pts = polygon.astype(np.float64, copy=False)

        # Initial least-squares circle fit
        x = pts[:, 0]
        y = pts[:, 1]

        A = np.column_stack([x, y, np.ones_like(x)])
        b = -(x * x + y * y)

        coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        a, b_, c = coeffs
        cx = -a / 2.0
        cy = -b_ / 2.0

        r_sq = cx * cx + cy * cy - c
        if r_sq <= 0:
            raise EyeCentersEstimationError("Failed to fit a valid circle to the polygon")

        r = np.sqrt(r_sq)
        center = np.array([cx, cy], dtype=np.float64)

        # Trim outliers using radial residuals
        d = np.linalg.norm(pts - center, axis=1)
        residuals = np.abs(d - r)

        num_keep = max(int(len(pts) * self.params.inlier_ratio), 3)

        keep_idx = np.argsort(residuals)[:num_keep]
        inliers = pts[keep_idx]

        # Refit circle on inliers
        x = inliers[:, 0]
        y = inliers[:, 1]

        A = np.column_stack([x, y, np.ones_like(x)])
        b = -(x * x + y * y)

        coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        a, b_, c = coeffs
        cx = -a / 2.0
        cy = -b_ / 2.0

        r_sq = cx * cx + cy * cy - c
        if r_sq <= 0:
            raise EyeCentersEstimationError("Failed to refit a valid circle to the inlier polygon points")

        return float(cx), float(cy)
