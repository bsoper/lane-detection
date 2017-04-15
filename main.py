import cv2
import sys
from matplotlib import pyplot as plt
from moviepy.video.io.VideoFileClip import VideoFileClip
from scipy import misc
import numpy as np

from polydrawer import Polydrawer
from polyfitter import Polyfitter
from thresholder import Thresholder
from undistorter import Undistorter
from warper import Warper
from analyze_lane_type import LaneTypeAnalysis

from image_filtering import filter_image, verify_image_filter

undistorter = Undistorter()
thresholder = Thresholder()
warper = Warper()
polyfitter = Polyfitter()
polydrawer = Polydrawer()
lane_type_analyzer = LaneTypeAnalysis()

image_filter = None


def main(video_name='other_video'):
    # video = 'harder_challenge_video'
    # video = 'challenge_video'
    if video_name.endswith('.mp4'):
        video_name = video_name.rsplit('.', 1)[0]

    white_output = '{}_done_2.mp4'.format(video_name)
    clip1 = VideoFileClip('{}.mp4'.format(video_name))#.subclip(0, 5)
    white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)


def process_image(base):
    
    fig = plt.figure(figsize=(10, 8))
    i = 1
    undistorted = undistorter.undistort(base)
    misc.imsave('output_images/undistorted.jpg', undistorted)
        # i = show_image(fig, i, undistorted, 'Undistorted', 'gray')

    try:
        img = thresholder.threshold(undistorted)
        misc.imsave('output_images/thresholded.jpg', img)
        # i = show_image(fig, i, img, 'Thresholded', 'gray')

        img = warper.warp(img)
        kernel = np.ones((np.ceil(img.shape[1]/40),np.ceil(img.shape[1]/40)),np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

        global image_filter
        image_filter = verify_image_filter(image_filter)
        img, image_filter = filter_image(img, image_filter)

        misc.imsave('output_images/warped.jpg', img)
        warp_color = warper.warp(undistorted)
        warp_color[(img == 0)] = 0
        misc.imsave('output_images/warped_color.jpg', warp_color)
        #return warp_color

        # i = show_image(fig, i, img, 'Warped', 'gray')
        left_lane, right_lane, left_centers, right_centers = \
            lane_type_analyzer.get_lane_type('output_images/warped.jpg', 'output_images/warped_color.jpg')
        left_fit, right_fit = polyfitter.polyfit(img)
        #left_fit, right_fit = generate_fits(left_centers, right_centers, img)

        img = polydrawer.draw(undistorted, left_fit, right_fit, warper.Minv)
        misc.imsave('output_images/final.jpg', img)
        lane_type_analyzer.update_polyfit_coeff(left_fit, right_fit)

        # show_image(fig, i, img, 'Final')

        # plt.show()
        # plt.get_current_fig_manager().frame.Maximize(True)

        #lane_curve, car_pos = polyfitter.measure_curvature(img)

        # if car_pos > 0:
        #     car_pos_text = '{}m right of center'.format(car_pos)
        # else:
        #     car_pos_text = '{}m left of center'.format(abs(car_pos))

        # cv2.putText(img, "Lane curve: {}m".format(lane_curve.round()), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
        #             color=(255, 255, 255), thickness=2)
        # cv2.putText(img, "Car is {}".format(car_pos_text), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 255, 255),
        #             thickness=2)

        # Add lane information to image
        img = add_lane_text(left_lane, right_lane, img)

        # show_image(fig, i, img, 'Final')
        # plt.imshow(img)
        # plt.show()
        set_src(left_fit,right_fit,img.shape[0])
        return img
    except:
        if (lane_type_analyzer.left_fit != None and lane_type_analyzer.right_fit != None):
            undistorted = polydrawer.draw(undistorted, lane_type_analyzer.left_fit, lane_type_analyzer.right_fit, warper.Minv)
        img = add_lane_text(lane_type_analyzer.last_left, lane_type_analyzer.last_right, undistorted)
        cv2.putText(img, "EXCEPTION IN PROCESSING", (450, 340), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    color=(255, 0, 0), thickness=2)
        return img

def show_image(fig, i, img, title, cmap=None):
    a = fig.add_subplot(2, 2, i)
    plt.imshow(img, cmap)
    a.set_title(title)
    return i + 1


def generate_fits(left_centers, right_centers, img):
    height, width = img.shape
    leftx = [p[0] for p in left_centers]
    lefty = [p[1] for p in left_centers]
    rightx = [p[0] + width / 2 for p in right_centers]
    righty = [p[1] for p in right_centers]
    #print (leftx)
    #print (righty)
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    return left_fit, right_fit


def add_lane_text(left_lane, right_lane, img):
    height, width, depth = img.shape
    cv2.putText(img, str(left_lane), (10, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
    cv2.putText(img, str(right_lane), (width - 320, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)

    return img

def set_src(left,right,height):
    # Initialize
    fity = np.float32([0, height])
    dst = np.copy(warper.dst)
    change = False

    # Polyfit method:
    left = left[::-1]
    right = right[::-1]
    left_fit = np.zeros(4)
    left_fit[0:left.size] = left
    right_fit = np.zeros(4)
    right_fit[0:right.size] = right
    left_x = left_fit[3] * fity ** 3 + left_fit[2] * fity ** 2 + left_fit[1] * fity + left_fit[0]
    right_x = right_fit[3] * fity ** 3 + right_fit[2] * fity ** 2 + right_fit[1] * fity + right_fit[0]

    # Splinefit method
    #    left_x = left(fity)
    #    right_x = right(fity)
    # Bottom points adjusting:
    if np.isfinite(left_x[1]) and warper.ratio[0, 0] * 500 > abs(left_x[1] - dst[3, 0]) > warper.ratio[0, 0] * 50:
        dst[3, 0] = left_x[1]
        change = True
    if np.isfinite(right_x[1]) and warper.ratio[0, 0] * 500 > abs(right_x[1] - dst[2, 0]) > warper.ratio[0, 0] * 50:
        dst[2, 0] = right_x[1]
        change = True

    # Top points adjusting:
    if np.isfinite(left_x[0]) and warper.ratio[0, 0] * 150 < abs(left_x[0] - dst[0, 0]) < warper.ratio[0, 0] * 520:
        dst[0, 0] = left_x[0]
        change = True
    if np.isfinite(right_x[0]) and warper.ratio[0, 0] * 150 < abs(right_x[0] - dst[1, 0]) < warper.ratio[0, 0] * 520:
        dst[1, 0] = right_x[0]
        change = True

    # Src point update
    if change:
        src_n = cv2.perspectiveTransform(np.asarray([dst], dtype=np.float32), np.asarray(warper.Minv, dtype=np.float32))
        src_n = np.squeeze(np.asarray(src_n, dtype=np.int16))
        trap_area = ((src_n[1, 0] - src_n[0, 0]) + (src_n[2, 0] - src_n[3, 0])) / 2.0 * (src_n[2, 1] - src_n[1, 1])
        if abs(trap_area - warper.trap_area) < 0.5 * warper.trap_area:
            print('changed src ')
            print(np.array_str(src_n))
            warper.src_n = np.copy(src_n)    


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
