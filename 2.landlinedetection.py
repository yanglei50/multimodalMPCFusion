# coding=utf-8
import cv2
import numpy as np


def get_edge_img(color_img, gaussian_ksize=5, gaussian_sigmax=1,
                 canny_threshold1=50, canny_threshold2=100):
    """
    灰度化,模糊,canny变换,提取边缘
    :param color_img: 彩色图,channels=3
    """
    gaussian = cv2.GaussianBlur(color_img, (gaussian_ksize, gaussian_ksize),
                                gaussian_sigmax)
    gray_img = cv2.cvtColor(gaussian, cv2.COLOR_BGR2GRAY)
    edges_img = cv2.Canny(gray_img, canny_threshold1, canny_threshold2)
    return edges_img


def roi_mask(gray_img):
    """
    对gray_img进行掩膜
    :param gray_img: 灰度图,channels=1
    """
    # poly_pts = np.array([[[0, 368], [300, 210], [340, 210], [640, 368]]])
    poly_pts = np.array([[[425, 283], [548, 284], [954, 491], [174, 491]]])

    mask = np.zeros_like(gray_img)
    mask = cv2.fillPoly(mask, pts=poly_pts, color=255)
    img_mask = cv2.bitwise_and(gray_img, mask)
    return img_mask


def get_lines(edge_img):
    """
    获取edge_img中的所有线段
    :param edge_img: 标记边缘的灰度图
    """

    def calculate_slope(line):
        """
        计算线段line的斜率
        :param line: np.array([[x_1, y_1, x_2, y_2]])
        :return:
        """
        x_1, y_1, x_2, y_2 = line[0]
        return (y_2 - y_1) / (x_2 - x_1)

    def reject_abnormal_lines(lines, threshold=0.2):
        """
        剔除斜率不一致的线段
        :param lines: 线段集合, [np.array([[x_1, y_1, x_2, y_2]]),np.array([[x_1, y_1, x_2, y_2]]),...,np.array([[x_1, y_1, x_2, y_2]])]
        """
        slopes = [calculate_slope(line) for line in lines]
        while len(lines) > 0:
            mean = np.mean(slopes)
            diff = [abs(s - mean) for s in slopes]
            idx = np.argmax(diff)
            if diff[idx] > threshold:
                slopes.pop(idx)
                lines.pop(idx)
            else:
                break
        return lines

    def least_squares_fit(lines):
        """
        将lines中的线段拟合成一条线段
        :param lines: 线段集合, [np.array([[x_1, y_1, x_2, y_2]]),np.array([[x_1, y_1, x_2, y_2]]),...,np.array([[x_1, y_1, x_2, y_2]])]
        :return: 线段上的两点,np.array([[xmin, ymin], [xmax, ymax]])
        """
        x_coords = np.ravel([[line[0][0], line[0][2]] for line in lines])
        y_coords = np.ravel([[line[0][1], line[0][3]] for line in lines])
        poly = np.polyfit(x_coords, y_coords, deg=1)
        point_min = (np.min(x_coords), np.polyval(poly, np.min(x_coords)))
        point_max = (np.max(x_coords), np.polyval(poly, np.max(x_coords)))
        return np.array([point_min, point_max], dtype=np.int)

    # Methods to find center point
    # (originally avg of x and y was used but change to confirm I was not doing that wrong)
    def right_to_left(rightX, rightZ, leftX, leftZ):
        rtl_diff = np.array([leftX - rightX, leftZ - rightZ])
        rtl_mag = np.sqrt(sum(rtl_diff ** 2))
        rtl_dir = rtl_diff / rtl_mag
        return rtl_dir, rtl_mag

    def midPoint(rightX, rightZ, leftX, leftZ):
        rtl_dir, rtl_mag = right_to_left(rightX, rightZ, leftX, leftZ)
        mid_vec = rtl_dir * rtl_mag * .5
        return rightX + mid_vec[0], rightZ + mid_vec[1]
    # 获取所有线段
    lines = cv2.HoughLinesP(edge_img, 1, np.pi / 180, 20, minLineLength=20,
                            maxLineGap=20)
    # 按照斜率分成车道线
    left_lines = [line for line in lines if calculate_slope(line) > 0]
    right_lines = [line for line in lines if calculate_slope(line) < 0]

    # 剔除离群线段
    left_lines = reject_abnormal_lines(left_lines)
    right_lines = reject_abnormal_lines(right_lines)

    # 计算中心车道线
    midP = midPoint(right_lines[0][0], right_lines[1][0], left_lines[0][0], left_lines[1][0])
    mids = [[midP[0]], [midP[1]]]

    return least_squares_fit(left_lines), least_squares_fit(right_lines), least_squares_fit(mids)



def draw_lines(img, lines):
    left_line, right_line,middle_Line = lines
    cv2.line(img, tuple(left_line[0]), tuple(left_line[1]), color=(0, 255, 255),
             thickness=5)
    cv2.line(img, tuple(middle_Line[0]), tuple(middle_Line[1]), color=(0, 255, 0),
             thickness=3)
    cv2.line(img, tuple(right_line[0]), tuple(right_line[1]),
             color=(0, 255, 255), thickness=5)


def show_lane(color_img,index):
    edge_img = get_edge_img(color_img)
    mask_gray_img = roi_mask(edge_img)
    # cv2.imwrite('D:/DataContest/data/image2/1659666219.43_1659666263.64/mask_gray_img'+str(index)+'.png', mask_gray_img)
    lines = get_lines(mask_gray_img)
    draw_lines(color_img, lines)
    cv2.imwrite('D:/DataContest/data/image2/1659666219.43_1659666263.64/line_img'+str(index)+'.png', color_img)

    return color_img

if __name__ == '__main__':
    capture = cv2.VideoCapture('D:/DataContest/data/image2/1659666219.43_1659666263.64/output2.mp4')
    index=0
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        frame = show_lane(frame,index)
        cv2.imshow('frame', frame)
        cv2.waitKey(10)
        index=index+1
