import cv2
import numpy as np
import cv2 as cv
import os


def calibration():
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((9 * 6, 3), np.float32)
    objp[:, :2] = np.mgrid[0:6, 0:9].T.reshape(-1, 2)
    print(objp)


    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    for fname in os.listdir("./imgs"):
        img = cv.imread(os.path.join("./imgs", fname))
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (6, 9), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            cv.drawChessboardCorners(img, (6, 9), corners2, ret)
            #fname = fname.split(".")[0]
            #cv2.imwrite(f"./imgs/{fname}_check.jpg", img)

    # calibration
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    for fname in os.listdir("./imgs"):
        img = cv.imread(os.path.join("./imgs", fname))

        h, w = img.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        dst = cv.undistort(img, mtx, dist, None, newcameramtx)

        # crop the image
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]
        fname = fname.split(".")[0]
        cv2.imwrite(f"./check/{fname}_check.jpg", dst)


if __name__ == "__main__":
    calibration()