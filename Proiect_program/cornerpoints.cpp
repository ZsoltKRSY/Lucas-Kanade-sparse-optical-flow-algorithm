//
// Created by Dogmedve on 2025. 05. 03..
//

#include <opencv2/opencv.hpp>
#include "cornerpoints.h"

using namespace std;
using namespace cv;

void compute_structure_tensor(const Mat &img, Mat &Sxx, Mat &Syy, Mat &Sxy) {
    Mat Ix, Iy;

    calculateGradients(img, Ix, Iy);

    Mat Ixx = Ix.mul(Ix);
    Mat Iyy = Iy.mul(Iy);
    Mat Ixy = Ix.mul(Iy);

    Sxx = gaussianBlur<double>(Ixx, 3, 1.0);
    Syy = gaussianBlur<double>(Iyy, 3, 1.0);
    Sxy = gaussianBlur<double>(Ixy, 3, 1.0);
}

Mat compute_corner_response(const Mat &Sxx, const Mat &Syy, const Mat &Sxy) {
    Mat response(Sxx.rows, Sxx.cols, CV_64F);

    for (int y = 0; y < Sxx.rows; ++y) {
        for (int x = 0; x < Sxx.cols; ++x) {
            double a = Sxx.at<double>(y, x);
            double b = Sxy.at<double>(y, x);
            double d = Syy.at<double>(y, x);

            response.at<double>(y, x) = (a + d - sqrt((a - d) * (a - d) + 4 * b * b)) / 2.0;
        }
    }

    return response;
}

vector<Point2f> non_maximum_suppression(const Mat &response, int max_corners, double threshold, double min_distance) {
    vector<Point2f> corners;
    vector<pair<double, Point2f>> allCandidates;

    for (int y = 1; y < response.rows - 1; ++y) {
        for (int x = 1; x < response.cols - 1; ++x) {
            double val = response.at<double>(y, x);

            if (val >= threshold) {
                bool isLocalMax = true;
                for (int dy = -1; dy <= 1 && isLocalMax; ++dy) {
                    for (int dx = -1; dx <= 1 && isLocalMax; ++dx) {
                        if (dy != 0 && dx != 0) {
                            if (response.at<double>(y + dy, x + dx) >= val) {
                                isLocalMax = false;
                            }
                        }
                    }
                }

                if (isLocalMax) {
                    allCandidates.emplace_back(val, Point2f((float) x, (float) y));
                }
            }
        }
    }

    sort(allCandidates.begin(), allCandidates.end(),
         [](const pair<double, Point2f> &a, const pair<double, Point2f> &b) {
             return a.first > b.first;
         });

    for (const auto &candidate: allCandidates) {
        bool tooClose = false;
        for (const auto &pt: corners) {
            if (norm(pt - candidate.second) < min_distance) {
                tooClose = true;
                break;
            }
        }
        if (!tooClose) {
            corners.push_back(candidate.second);
            if ((int) corners.size() >= max_corners)
                break;
        }
    }

    return corners;
}

vector<Point2f> detect_cornerPoints(const Mat &img, int max_corners, double quality, int min_distance) {
    Mat Sxx, Syy, Sxy;
    compute_structure_tensor(img, Sxx, Syy, Sxy);

    Mat response = compute_corner_response(Sxx, Syy, Sxy);

    double maxVal = response.at<double>(0, 0);
    for (int i = 0; i < response.rows; ++i) {
        for (int j = 0; j < response.cols; ++j) {
            if (response.at<double>(i, j) > maxVal) {
                maxVal = response.at<double>(i, j);
            }
        }
    }
    double threshold = quality * maxVal;

    vector<Point2f> result;
    result = non_maximum_suppression(response, max_corners, threshold, min_distance);

    return result;
}

void cornerPoints_show(const Mat &img, const vector<Point2f> &cornerPoints, const char *title) {
    Mat img_show = grayscale_to_bgr(img);

    for (const auto &point: cornerPoints) {
        circle(img_show, point, 4, {0, 0, 255}, FILLED);
    }

    imshow(title, img_show);
}