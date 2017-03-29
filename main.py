import cv2
import sys
from matplotlib import pyplot as plt
from moviepy.video.io.VideoFileClip import VideoFileClip
from scipy import misc

from polydrawer import Polydrawer
from polyfitter import Polyfitter
from thresholder import Thresholder
from undistorter import Undistorter
from warper import Warper
from lane import Lane
from dotted_detect import detect_dotted
from detect_color import detect_color

undistorter = Undistorter()
thresholder = Thresholder()
warper = Warper()
polyfitter = Polyfitter()
polydrawer = Polydrawer()


def main(video_name='other_video'):
    # video = 'harder_challenge_video'
    # video = 'challenge_video'
    if video_name.endswith('.mp4'):
        video_name = video_name.split('.')[0]

    white_output = '{}_done_2.mp4'.format(video_name)
    clip1 = VideoFileClip('{}.mp4'.format(video_name)).subclip(20, 25)
    warper.set_transforms(clip1.size)
    white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)


def process_image(base):
    
    fig = plt.figure(figsize=(10, 8))
    i = 1
    undistorted = undistorter.undistort(base)
    misc.imsave('output_images/undistorted.jpg', undistorted)
        # i = show_image(fig, i, undistorted, 'Undistorted', 'gray')

    #try:
    img = thresholder.threshold(undistorted)
    misc.imsave('output_images/thresholded.jpg', img)
    # i = show_image(fig, i, img, 'Thresholded', 'gray')

    img = warper.warp(img)
    misc.imsave('output_images/warped.jpg', img)
    warp_color = warper.warp(undistorted)
    warp_color[(img == 0)] = 0
    misc.imsave('output_images/warped_color.jpg', warp_color)

    # i = show_image(fig, i, img, 'Warped', 'gray')

    left_lane, right_lane = detect_dotted('output_images/warped.jpg')
    left_lane.color, right_lane.color = detect_color('output_images/warped_color.jpg')
    left_fit, right_fit = polyfitter.polyfit(img)

    img = polydrawer.draw(undistorted, left_fit, right_fit, warper.Minv)
    misc.imsave('output_images/final.jpg', img)
    # show_image(fig, i, img, 'Final')

    # plt.show()
    # plt.get_current_fig_manager().frame.Maximize(True)

    lane_curve, car_pos = polyfitter.measure_curvature(img)

    if car_pos > 0:
        car_pos_text = '{}m right of center'.format(car_pos)
    else:
        car_pos_text = '{}m left of center'.format(abs(car_pos))

    cv2.putText(img, "Lane curve: {}m".format(lane_curve.round()), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                color=(255, 255, 255), thickness=2)
    cv2.putText(img, "Car is {}".format(car_pos_text), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 255, 255),
                thickness=2)

    # Add lane information to image
    img = add_lane_text(left_lane, right_lane, img)

    # show_image(fig, i, img, 'Final')
    # plt.imshow(img)
    # plt.show()

    return img
    #except:
    #    cv2.putText(undistorted, "EXCEPTION IN PROCESSING", (450, 340), cv2.FONT_HERSHEY_SIMPLEX, 1,
    #                color=(255, 0, 0), thickness=2)
    #    return undistorted


def show_image(fig, i, img, title, cmap=None):
    a = fig.add_subplot(2, 2, i)
    plt.imshow(img, cmap)
    a.set_title(title)
    return i + 1


def add_lane_text(left_lane, right_lane, img):
    cv2.putText(img, left_lane.solid, (10, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
    cv2.putText(img, right_lane.solid, (100, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
    cv2.putText(img, left_lane.color, (10, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
    cv2.putText(img, right_lane.color, (100, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
    return img

if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
