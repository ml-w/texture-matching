import numpy as np
import cv2
from typing import Optional, Tuple


def visualize_bbox(im: np.ndarray, 
                   coords: np.ndarray, 
                   box_size: int, 
                   thickness: Optional[int] = 1,
                   zoom_factor: Optional[int] = 32, 
                   box_color: Optional[Tuple[int, int, int]] = (255, 0, 0)) -> np.ndarray:
    """Draws rectangles on an image for the purpose of visualizing bounding boxes.

    Given an image and coordinates, this function scales the image by a specified
    zoom factor and draws rectangles at the given coordinates. The size, thickness,
    and color of the boxes can be customized. This is useful for visualizing
    object detection outputs or any other bounding boxes on an image.

    Args:
        im (np.ndarray):
            The input image on which to draw the bounding boxes.
        coords (np.ndarray):
            An array of coordinates where each row represents the top-left corner
            of a rectangle to be drawn.
        box_size (int):
            The size of the side of the square bounding box to draw.
        thickness (int, optional):
            The thickness of the lines used to draw the box edges. Defaults to 1.
        zoom_factor (int, optional):
            The factor by which to scale the image and bounding box coordinates.
            Defaults to 32.
        box_color (Tuple[int, int, int], optional):
            A tuple representing the color of the bounding box in BGR (blue, green, red)
            format. Defaults to red (255, 0, 0).

    Returns:
        np.ndarray:
            The output image with rectangles drawn on it, scaled by the specified
            zoom factor.

    Raises:
        ArithmeticError: If the zoom factor is less than 10, which would render the
            output image messy and indistinguishable.

    .. notes::
        - The input coordinates should be sorted beforehand if required, as this
          function does not sort the coordinates.
        - To avoid overlap of boxes and to make them distinct, the function may
          adjust the boxes' size and color as it draws them.
        - The function assumes that the input image and the coordinates are in
          pixel units and scales them both by the zoom factor.
    """
    # if zoom factor < 10, you see nothing meaningful
    if zoom_factor < 10:
        raise ArithmeticError("Zoom factor < 10 will renders output messy.")
    
    # make sure coordinates are sorted
    x = coords.sum(axis=1).argsort()
    coords = coords[x]
    
    # resize based on zoom factor
    im_out = cv2.resize(im, np.array([im.shape[1], im.shape[0]]) * zoom_factor, interpolation=cv2.INTER_NEAREST)

    # draw rectangles
    for _coords in coords:
        _coords = _coords * zoom_factor
        _box_size = box_size * zoom_factor - 1 # -1 to avoid crossing to another cell
        _thickness = thickness
        _box_color = np.array(box_color, dtype=np.uint8)

        # check if there's already a box drawn
        while all(np.isclose(im_out[_coords[1], _coords[0]], np.array(_box_color))):
            # shrink the box by 2pixels to try avoid overlap
            _coords += 2
            _box_size -= 2 * 2
            # make color a bit darker
            _box_color[_box_color != 0] -= 50
            
        im_out = cv2.rectangle(im_out, _coords, _coords + _box_size, _box_color.tolist(), _thickness)
    return im_out


def draw_dots_at_coords(im, coords, zoom_factor):
    """Draws dots on an image at specified coordinates.

    This function scales the coordinates by the specified zoom factor and draws a
    dot at each scaled coordinate on the image. This can be used for marking points
    of interest such as feature points or object centroids on an image.

    Args:
        im (np.ndarray):
            The input image on which to draw the dots.
        coords (np.ndarray):
            An array where each row contains the (x, y) coordinates of a point
            where a dot is to be drawn.
        zoom_factor (int):
            The factor by which to scale the coordinates of the dots.
            
    Returns:
        np.ndarray:
            The output image with dots drawn on it at the scaled coordinates.
    
    .. notes::
        - The dots are drawn in green color with a marker size of 5 and a line
          thickness of 2.
        - The function assumes that the input image and the coordinates are in
          pixel units and scales the coordinates by the zoom factor before drawing.
    """
    for c in coords:
        _c = np.array(c) * zoom_factor
        cv2.drawMarker(im, [_c[0], _c[1]], [0, 255, 0], markerSize=5, thickness=2)
    return im

    
def visualize_bbox_plt(im: np.ndarray, 
                       coords: np.ndarray, 
                       box_size: int, 
                       thickness: Optional[int] = 1,
                       zoom_factor: Optional[int] = 32, 
                       box_color: Optional[Tuple[int, int, int]] = (255, 0, 0)) -> None:
    """Displays an image with bounding boxes overlay using matplotlib.

    This function takes an image and coordinates, and displays the image with
    square bounding boxes drawn at the specified coordinates. The box size, 
    thickness, and color can be customized. The image is displayed using matplotlib's
    plotting capabilities.

    Args:
        im (np.ndarray): 
            The input image on which to draw the bounding boxes.
        coords (np.ndarray): 
            An array of coordinates where each row represents the top-left corner
            of a bounding box to be drawn.
        box_size (int): 
            The size of the side of the square bounding box to draw.
        thickness (int, optional): 
            The thickness of the lines used to draw the box edges. Defaults to 1.
        zoom_factor (int, optional): 
            The factor by which to scale the image before drawing the bounding boxes.
            Defaults to 32. Note that in this function, this parameter is not used
            and is included for consistency with related functions.
        box_color (Tuple[int, int, int], optional): 
            A tuple representing the color of the bounding box in RGB (red, green, blue)
            format (matplotlib uses RGB instead of OpenCV's BGR). Defaults to red (255, 0, 0).

    .. notes::
        - Matplotlib should be installed in the environment where this function is being used.
        - This function is intended for interactive use to visually check bounding boxes and
          is not suitable for saving or processing images at scale.
        - The zoom_factor parameter is not utilized in the current implementation of the function
          and is provided for API consistency with similar functions that do use it.
        - The function does not return anything as it is meant for immediate visualization.
    """
    import matplotlib.pytplot as plt
    import matplotlib.patches as patches
    
    fig, ax = plt.subplots(1, 1, dpi=350)
    ax.imshow(im)
    for _coords in coords:
        _coords = _coords 
        rect = patches.Rectangle(_coords - 0.5, box_size, box_size, linewidth=thickness, edgecolor=box_color, facecolor='none')
        ax.add_patch(rect)
    plt.show()
    