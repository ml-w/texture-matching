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
            The input image on which to draw the bounding boxes. The input should has
            three channels RGB, ranged from 0 to 255.
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

    .. note::
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
                       box_color: Optional[Tuple[int, int, int, int]] = (1., 0., 0., 1.),
                       dpi: Optional[int] = 300) -> None:
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
        box_color (Tuple[int, int, int], optional):
            The color of the box edges. A tuple representing the RGBA values in the range [0,1].
            Defaults to red (1, 0, 0, 1).
        dpi (int, optional):
            The resolution of the figure in dots-per-inch. Defaults to 300.

    .. note:
        - `im` must be a numpy array. If it contains an alpha channel (is RGBA), the values should
          be in the range [0, 1]. If the image is RGB, grayscale, or otherwise, there are no specific
          limitations on the value range.
        - The function assumes `coords` provides the top-left corner of each bounding box. If you need
          to specify the center, adjust the coordinates before passing them to the function.
        - The parameter `box_color` should be a tuple of three elements if specifying an RGB color.
          Each element should be a float in the range [0, 1] corresponding to the red, green, and blue
          components of the color.
        - The display is not persistent; it is closed after the `plt.show()` call completes. To save
          the figure, you will need to call `plt.savefig()` before `plt.show()`.

    Raises:
        ValueError: If `coords` contains non-integer values or is not of shape Nx2.
        TypeError: If `im` is not a numpy.ndarray.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    fig, ax = plt.subplots(1, 1, dpi=dpi)
    ax.imshow(im)
    for _coords in coords:
        _coords = _coords
        # coords is the center of the box
        rect = patches.Rectangle(_coords - 0.5, box_size, box_size,
                                 linewidth=thickness, edgecolor=box_color, facecolor='none')
        ax.add_patch(rect)
    plt.show()
    