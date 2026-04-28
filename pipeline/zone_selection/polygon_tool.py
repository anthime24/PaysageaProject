import cv2
import numpy as np
from typing import List, Tuple, Optional


Point = Tuple[float, float]


class PolygonTool:
    """
    Outil de sélection polygonale basé sur OpenCV.

    - clic gauche : ajoute un point
    - touche 'c'   : ferme le polygone (relie au premier point)
    - touche 'r'   : reset
    - ENTER / SPACE: valider et retourner le masque
    - ESC          : annuler (retourne (None, []))
    """

    def __init__(self, image_display, window_name: str = "Polygon Tool"):
        self.image = image_display
        self.window_name = window_name

        self.points: List[Point] = []
        self.finished = False
        self.cancelled = False

        h, w = self.image.shape[:2]
        self.mask = np.zeros((h, w), dtype="uint8")

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((float(x), float(y)))

    def _draw(self):
        canvas = self.image.copy()

        # dessiner les segments
        if len(self.points) > 1:
            for i in range(len(self.points) - 1):
                p1 = tuple(map(int, self.points[i]))
                p2 = tuple(map(int, self.points[i + 1]))
                cv2.line(canvas, p1, p2, (0, 255, 0), 2)

        # dessiner les points
        for p in self.points:
            cv2.circle(canvas, tuple(map(int, p)), 4, (0, 0, 255), -1)

        cv2.imshow(self.window_name, canvas)

    def _finalize_mask(self):
        if len(self.points) < 3:
            self.mask[...] = 0
            return

        pts = np.array(self.points, dtype="float32")
        pts_int = np.round(pts).astype("int32")
        cv2.fillPoly(self.mask, [pts_int], color=1)

    def run(self) -> Tuple[Optional[np.ndarray], List[Point]]:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        print("Mode polygon :")
        print("- Clic gauche : ajouter un point")
        print("- Touche 'c'  : fermer le polygone")
        print("- Touche 'r'  : reset")
        print("- ENTER / SPACE : valider")
        print("- ESC          : annuler")

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
                self.points = []
                self.mask[...] = 0
            elif key in (ord("c"), ord("C")):
                # fermer le polygone en reliant dernier -> premier
                if len(self.points) >= 3:
                    self.points.append(self.points[0])

        cv2.destroyWindow(self.window_name)

        if not self.finished or self.cancelled:
            return None, []

        self._finalize_mask()
        return self.mask, self.points

