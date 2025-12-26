"""
Utility functions for geometry, collision detection, and math operations.
"""
import math
from typing import Tuple, List, Optional
import numpy as np


def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def normalize_vector(v: Tuple[float, float]) -> Tuple[float, float]:
    """Normalize a 2D vector."""
    length = math.sqrt(v[0] ** 2 + v[1] ** 2)
    if length < 1e-6:
        return (0.0, 0.0)
    return (v[0] / length, v[1] / length)


def point_in_rect(point: Tuple[float, float], rect: Tuple[int, int, int, int], 
                  margin: float = 0) -> bool:
    """Check if a point is inside a rectangle (x, y, width, height) with optional margin."""
    x, y, w, h = rect
    return (x - margin <= point[0] <= x + w + margin and 
            y - margin <= point[1] <= y + h + margin)


def circle_rect_collision(circle_center: Tuple[float, float], radius: float,
                          rect: Tuple[int, int, int, int]) -> bool:
    """Check if a circle collides with a rectangle."""
    cx, cy = circle_center
    rx, ry, rw, rh = rect
    
    # Find the closest point on the rectangle to the circle center
    closest_x = max(rx, min(cx, rx + rw))
    closest_y = max(ry, min(cy, ry + rh))
    
    # Check if the closest point is within the circle
    dist = distance((cx, cy), (closest_x, closest_y))
    return dist < radius


def get_rect_center(rect: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """Get the center point of a rectangle."""
    x, y, w, h = rect
    return (x + w / 2, y + h / 2)


def get_rect_edge_point(rect: Tuple[int, int, int, int], 
                        side: str) -> Tuple[float, float]:
    """Get the center point of a specific edge of a rectangle.
    
    Args:
        rect: (x, y, width, height)
        side: 'top', 'bottom', 'left', or 'right'
    """
    x, y, w, h = rect
    if side == 'top':
        return (x + w / 2, y)
    elif side == 'bottom':
        return (x + w / 2, y + h)
    elif side == 'left':
        return (x, y + h / 2)
    elif side == 'right':
        return (x + w, y + h / 2)
    else:
        return get_rect_center(rect)


def get_circle_edge_point(center: Tuple[float, float], radius: float,
                          angle: float) -> Tuple[float, float]:
    """Get a point on the edge of a circle at a given angle (in radians)."""
    return (center[0] + radius * math.cos(angle),
            center[1] + radius * math.sin(angle))


def angle_between_points(p1: Tuple[float, float], 
                         p2: Tuple[float, float]) -> float:
    """Get the angle from p1 to p2 in radians."""
    return math.atan2(p2[1] - p1[1], p2[0] - p1[0])


def move_towards(current: Tuple[float, float], target: Tuple[float, float],
                 speed: float) -> Tuple[float, float]:
    """Move from current position towards target at given speed.
    
    Returns new position. If distance is less than speed, returns target.
    """
    dist = distance(current, target)
    if dist <= speed:
        return target
    
    direction = normalize_vector((target[0] - current[0], target[1] - current[1]))
    return (current[0] + direction[0] * speed,
            current[1] + direction[1] * speed)


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max."""
    return max(min_val, min(max_val, value))


def random_point_outside_rects(rects: List[Tuple[int, int, int, int]],
                               bounds: Tuple[int, int, int, int],
                               radius: float,
                               rng: np.random.Generator,
                               max_attempts: int = 100) -> Optional[Tuple[float, float]]:
    """Generate a random point that doesn't collide with any rectangles.
    
    Args:
        rects: List of rectangles to avoid
        bounds: (x, y, width, height) of the valid area
        radius: Radius of the circle to place
        rng: Random number generator
        max_attempts: Maximum attempts before giving up
    
    Returns:
        A valid point or None if no valid point found
    """
    bx, by, bw, bh = bounds
    
    for _ in range(max_attempts):
        x = rng.uniform(bx + radius, bx + bw - radius)
        y = rng.uniform(by + radius, by + bh - radius)
        
        valid = True
        for rect in rects:
            if circle_rect_collision((x, y), radius + 5, rect):  # 5px margin
                valid = False
                break
        
        if valid:
            return (x, y)
    
    return None


def find_position_near_rect(rect: Tuple[int, int, int, int],
                            side: str,
                            offset: float,
                            bounds: Tuple[int, int, int, int],
                            other_rects: List[Tuple[int, int, int, int]],
                            radius: float) -> Tuple[float, float]:
    """Find a valid position near a rectangle edge.
    
    Args:
        rect: Target rectangle
        side: 'top', 'bottom', 'left', 'right'
        offset: Distance from the edge
        bounds: Scene bounds
        other_rects: Other rectangles to avoid
        radius: Person radius
    
    Returns:
        A position near the specified edge
    """
    x, y, w, h = rect
    bx, by, bw, bh = bounds
    
    if side == 'top':
        px = x + w / 2
        py = y - offset
    elif side == 'bottom':
        px = x + w / 2
        py = y + h + offset
    elif side == 'left':
        px = x - offset
        py = y + h / 2
    else:  # right
        px = x + w + offset
        py = y + h / 2
    
    # Clamp to bounds
    px = clamp(px, bx + radius, bx + bw - radius)
    py = clamp(py, by + radius, by + bh - radius)
    
    return (px, py)


def get_bounding_box(center: Tuple[float, float], 
                     radius: float) -> Tuple[float, float, float, float]:
    """Get bounding box for a circle.
    
    Returns:
        (x_center, y_center, width, height) - YOLO format (not normalized)
    """
    return (center[0], center[1], radius * 2, radius * 2)


def get_combined_bounding_box(centers: List[Tuple[float, float]], 
                              radius: float,
                              padding: float = 5) -> Tuple[float, float, float, float]:
    """Get bounding box that encompasses multiple circles.
    
    Returns:
        (x_center, y_center, width, height) - YOLO format (not normalized)
    """
    if not centers:
        return (0, 0, 0, 0)
    
    min_x = min(c[0] for c in centers) - radius - padding
    max_x = max(c[0] for c in centers) + radius + padding
    min_y = min(c[1] for c in centers) - radius - padding
    max_y = max(c[1] for c in centers) + radius + padding
    
    width = max_x - min_x
    height = max_y - min_y
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    return (center_x, center_y, width, height)


def normalize_bbox(bbox: Tuple[float, float, float, float],
                   img_size: int) -> Tuple[float, float, float, float]:
    """Normalize bounding box coordinates to 0-1 range for YOLO format."""
    x, y, w, h = bbox
    return (x / img_size, y / img_size, w / img_size, h / img_size)
