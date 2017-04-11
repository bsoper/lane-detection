import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

class Polydrawer:
    def draw(self, img, left_fit, right_fit, Minv):
        color_warp = np.zeros_like(img).astype(np.uint8)

        fity = np.linspace(0, img.shape[0] - 1, img.shape[0])
        left_fitx = left_fit[0] * fity ** 2 + left_fit[1] * fity + left_fit[2]
        right_fitx = right_fit[0] * fity ** 2 + right_fit[1] * fity + right_fit[2]

        # Third Order
        #left_fitx = left_fit[0] * fity ** 3 + left_fit[1] * fity ** 2 + left_fit[2] * fity + left_fit[3]
        #right_fitx = right_fit[0] * fity ** 3 + right_fit[1] * fity ** 2 + right_fit[2] * fity + right_fit[3]

        # First Order
        #left_fitx = left_fit[0] * fity + left_fit[1]
        #right_fitx = right_fit[0] * fity + right_fit[1]

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, fity]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, fity])))])
        pts = np.hstack((pts_left, pts_right))
        pts = np.array(pts, dtype=np.int32)

        cv2.fillPoly(color_warp, pts, (0, 255, 0))
        misc.imsave('output_images/color_warp.jpg', color_warp)

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

        return result
