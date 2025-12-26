"""
Resolution-agnostic scaling utilities for multi-resolution video support.
"""


class ScaleAdapter:
    """
    Adapts pixel measurements from reference resolution to actual video size.

    All hardcoded thresholds and sizes are defined for 1920x1080 (Full HD).
    This class automatically scales them for videos of different resolutions.

    Attributes:
        scale_x: Horizontal scaling factor
        scale_y: Vertical scaling factor
        scale: Average scaling factor for distances
        width: Actual video width
        height: Actual video height
    """

    def __init__(self, actual_width, actual_height, reference_width=1920, reference_height=1080):
        """
        Initialize the scale adapter.

        Args:
            actual_width: Width of the actual video
            actual_height: Height of the actual video
            reference_width: Reference width for parameter tuning
            reference_height: Reference height for parameter tuning
        """
        self.scale_x = actual_width / reference_width
        self.scale_y = actual_height / reference_height
        self.scale = (self.scale_x + self.scale_y) / 2
        self.width = actual_width
        self.height = actual_height
        print(f"ScaleAdapter: {actual_width}x{actual_height} → scale={self.scale:.3f}")

    def scale_distance(self, distance):
        """
        Scale a distance measurement.

        Args:
            distance: Distance in reference resolution pixels

        Returns:
            float: Scaled distance in actual resolution pixels
        """
        return distance * self.scale

    def scale_point(self, point):
        """
        Scale a 2D point coordinate.

        Args:
            point: Tuple (x, y) in reference resolution

        Returns:
            tuple: Scaled (x, y) coordinates as integers
        """
        return (int(point[0] * self.scale_x), int(point[1] * self.scale_y))

    def scale_bbox(self, bbox):
        """
        Scale a bounding box.

        Args:
            bbox: Box coordinates [x1, y1, x2, y2] in reference resolution

        Returns:
            list: Scaled box coordinates as floats
        """
        return [
            bbox[0] * self.scale_x,
            bbox[1] * self.scale_y,
            bbox[2] * self.scale_x,
            bbox[3] * self.scale_y
        ]

    def scale_font(self, base_size=0.6):
        """
        Scale a font size for OpenCV text rendering.

        Args:
            base_size: Font scale in reference resolution

        Returns:
            float: Scaled font size (minimum 0.3)
        """
        return max(0.3, base_size * self.scale)

    def scale_thickness(self, base_thickness=2):
        """
        Scale a line thickness for OpenCV drawing.

        Args:
            base_thickness: Thickness in reference resolution

        Returns:
            int: Scaled thickness (minimum 1)
        """
        return max(1, int(base_thickness * self.scale))
