//
// Created by Dogmedve on 2025. 04. 25..
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include "opticalflow.h"
#include "cornerpoints.h"

using namespace std;
using namespace cv;

vector<Mat> buildImgPyramid(const Mat &source, int nr_levels) {
    vector<Mat> pyramid;
    pyramid.push_back(source.clone());

    for (int i = 1; i <= nr_levels; ++i) {
        Mat down = gaussianBlur<double>(pyramid[i - 1], 5, 1.0);
        resize(down, down, Size(pyramid[i - 1].cols / 2, pyramid[i - 1].rows / 2), 0, 0, INTER_LINEAR);
        pyramid.push_back(down);
    }

    return pyramid;
}

double bilinearInterpolate(const Mat &source, double y, double x) {
    int x0 = (int) floor(x);
    int y0 = (int) floor(y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    if (x0 < 0 || y0 < 0 || x1 >= source.cols || y1 >= source.rows) {
        return 0.0;
    }

    double a = x - x0;
    double b = y - y0;

    double val = (1 - a) * (1 - b) * source.at<double>(y0, x0) +
                 a * (1 - b) * source.at<double>(y0, x1) +
                 (1 - a) * b * source.at<double>(y1, x0) +
                 a * b * source.at<double>(y1, x1);

    return val;
}

vector<Point2f>
calculate_optical_flow(const Mat &prevImg, const Mat &nextImg, const vector<Point2f> &prevPoints,
                       int window_size, int max_iters, double epsilon, int nr_levels) {
    Mat prevImgConv, nextImgConv;
    prevImg.convertTo(prevImgConv, CV_64F);
    nextImg.convertTo(nextImgConv, CV_64F);

    vector<Mat> prevPyramid, nextPyramid;
    prevPyramid = buildImgPyramid(prevImgConv, nr_levels);
    nextPyramid = buildImgPyramid(nextImgConv, nr_levels);

    vector<Point2f> currPoints = prevPoints;
    float scale = 1.0f / (float) (1 << nr_levels);

    for (auto &point: currPoints) {
        point *= scale;
    }

    for (int level = nr_levels; level >= 0; --level) {
        Mat prev = prevPyramid[level];
        Mat next = nextPyramid[level];

        Mat Ix, Iy;
        calculateGradients(prev, Ix, Iy);

        int level_window_size = window_size * (1 << (nr_levels - level));
        int half_window = level_window_size / 2;
        int rows = prev.rows, cols = prev.cols;

        for (int idx = 0; idx < currPoints.size(); ++idx) {
            Point2f point = currPoints[idx];

            if (point.x < 0 || point.y < 0) {
                continue;
            }

            Point2f flow = Point2f(0.0f, 0.0f);

            bool success = true;

            for (int iter = 0; iter < max_iters; ++iter) {
                double A11 = 0, A12 = 0, A22 = 0;
                double b1 = 0, b2 = 0;

                for (int dy = -half_window; dy <= half_window; ++dy) {
                    for (int dx = -half_window; dx <= half_window; ++dx) {
                        float x = point.x + flow.x + (float) dx;
                        float y = point.y + flow.y + (float) dy;

                        if (x < 0 || x >= (float) cols || y < 0 || y >= (float) rows) {
                            continue;
                        }

                        double ix = bilinearInterpolate(Ix, y, x);
                        double iy = bilinearInterpolate(Iy, y, x);

                        double i1 = bilinearInterpolate(prev, y, x);
                        double i2 = bilinearInterpolate(next, y, x);
                        double it = i2 - i1;

                        A11 += ix * ix;
                        A12 += ix * iy;
                        A22 += iy * iy;

                        b1 += -ix * it;
                        b2 += -iy * it;
                    }
                }

                double det = A11 * A22 - A12 * A12;
                if (fabs(det) < 1e-6) {
                    success = false;
                    break;
                }

                double u = (A22 * b1 - A12 * b2) / det;
                double v = (A11 * b2 - A12 * b1) / det;

                flow.x += (float) u;
                flow.y += (float) v;

                if (sqrt(u * u + v * v) < epsilon) {
                    break;
                }
            }

            if (success) {
                currPoints[idx] = point + flow;
            } else {
                currPoints[idx] = Point2f(-1.0f, -1.0f);
            }
        }

        if (level > 0) {
            for (auto &currPoint: currPoints) {
                if (currPoint.x != -1 && currPoint.y != -1) {
                    currPoint *= 2.0f;
                }
            }
        }
    }

    return currPoints;
}

vector<vector<Point2f>>
frames_optical_flow(const vector<Mat> &frames, int max_corners, double quality, int min_distance, int window_size,
                    int max_iters, double epsilon, int nr_levels, int cornerPoint_refresh_rate) {
    vector<vector<Point2f>> result;

    printf("Frame 0\n");
    vector<Point2f> cornerPoints_next, cornerPoints_prev = detect_cornerPoints(frames.at(0), max_corners,
                                                                               quality,
                                                                               min_distance);
    result.push_back(cornerPoints_prev);

    for (int i = 1; i < frames.size(); ++i) {
        if (cornerPoint_refresh_rate > 0 && i % cornerPoint_refresh_rate == 0) {
            cornerPoints_prev = detect_cornerPoints(frames.at(i), max_corners, quality,
                                                    min_distance);
        }

//        vector<uchar> status;
//        vector<float> err;
//        TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
//        calcOpticalFlowPyrLK(imgSequence.frames.at(i - 1), imgSequence.frames.at(i), cornerPoints_prev.points,
//                             cornerPoints_next.points, status,
//                             err, Size(15, 15), 0, criteria);
//        Mat img_show = grayscale_to_bgr(frames.at(i));
//        for (const auto &point: cornerPoints_next) {
//            circle(img_show, point, 4, {0, 0, 255}, FILLED);
//        }
//        imshow("Frame", img_show);
//        waitKey(30);

        printf("Frame %d\n", i);
        cornerPoints_next = calculate_optical_flow(frames.at(i - 1), frames.at(i), cornerPoints_prev, window_size,
                                                   max_iters, epsilon, nr_levels);
        result.push_back(cornerPoints_next);

        cornerPoints_prev = cornerPoints_next;
    }

    return result;
}

vector<Mat>
frames_show_optical_flow(const vector<Mat> &frames, const vector<vector<Point2f>> &all_points,
                         int cornerPoint_refresh_rate, int cornerPoint_visual_refresh_rate, bool points) {
    int max_vector_size = (int) all_points.front().size();
    for (const auto &point_vector: all_points) {
        if (point_vector.size() > max_vector_size) {
            max_vector_size = (int) point_vector.size();
        }
    }

    vector<Scalar> colors;
    RNG rng;
    for (int i = 0; i < max_vector_size; ++i) {
        int b, g, r;
        b = rng.uniform(0, 256);
        g = rng.uniform(0, 256);
        r = rng.uniform(0, 256);
        colors.emplace_back(b, g, r);
    }

    vector<Mat> frames_show;
    for (int i = 0; i < frames.size(); ++i) {
        Mat current_frame = grayscale_to_bgr(frames.at(i));

        int j;
        if (cornerPoint_visual_refresh_rate < 0) {
            j = 0;
        } else {
            j = (i / cornerPoint_visual_refresh_rate) * cornerPoint_visual_refresh_rate;
        }
        for (; j < i; ++j) {
            if (!points && j < 1) {
                continue;
            }
            if (!points && cornerPoint_refresh_rate > 0 && (j % cornerPoint_refresh_rate == 0)) {
                continue;
            }

            for (int k = 0; k < all_points.at(j).size(); ++k) {
                Point2f point = all_points.at(j).at(k);

                if (points) {
                    if (point.x != -1 && point.y != -1) {
                        circle(current_frame, point, 4, colors.at(k), FILLED);
                    }
                } else {
                    Point2f pointPrev = all_points.at(j - 1).at(k);
                    if (point.x != -1 && point.y != -1 && pointPrev.x != -1 && pointPrev.y != -1) {
                        line(current_frame, pointPrev, point, colors.at(k), 3);
                    }
                }
            }
        }

        frames_show.push_back(current_frame);
    }

    return frames_show;
}