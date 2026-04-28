import cv2
import numpy as np
from typing import List, Optional, Tuple


Point = Tuple[float, float]


class BrushTool:
    """
    Outil de sélection au pinceau (style surligneur).

    - clic gauche + glisser : peinture fluide de la zone
    - touche 'r'            : reset complet
    - ENTER / SPACE         : valider
    - ESC                   : annuler
    """

    def __init__(self, image_display, window_name: str = "Brush Tool", brush_radius: int = 25):
        self.image = image_display
        self.window_name = window_name
        self.brush_radius = int(brush_radius)

        h, w = self.image.shape[:2]
        self.mask = np.zeros((h, w), dtype="uint8")

        self.drawing = False
        self.last_point: Optional[Tuple[int, int]] = None
        self.cancelled = False
        self.finished = False

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.last_point = (x, y)
            cv2.circle(self.mask, (x, y), self.brush_radius, 1, -1)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            if self.last_point is not None:
                # trait continu entre l'ancienne et la nouvelle position
                cv2.line(
                    self.mask,
                    self.last_point,
                    (x, y),
                    1,
                    thickness=self.brush_radius * 2,
                )
            self.last_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.last_point = None

    def _draw(self):
        # overlay visuel de la zone peinte (style surligneur rouge)
        overlay = self.image.copy()
        mask_bool = self.mask.astype(bool)
        if mask_bool.any():
            color = np.array([0, 0, 255], dtype="uint8")
            overlay[mask_bool] = (0.4 * overlay[mask_bool] + 0.6 * color).astype("uint8")
        cv2.imshow(self.window_name, overlay)

    def run(self) -> Tuple[Optional[np.ndarray], List[Point]]:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        print("Mode pinceau :")
        print("- Clic gauche + glisser : dessiner la zone (surligneur)")
        print("- Touche 'r'            : reset")
        print("- ENTER / SPACE         : valider")
        print("- ESC                   : annuler")

        while True:
            self._draw()
            key = cv2.waitKey(20) & 0xFF

            if key == 27:  # ESC
                self.cancelled = True
                break
            elif key in (13, 32):  # ENTER ou SPACE
                self.finished = True
                break
            elif key in (ord("r"), ord("R")):
                self.mask[...] = 0

        cv2.destroyWindow(self.window_name)

        if not self.finished or self.cancelled:
            return None, []

        return self.mask, []

