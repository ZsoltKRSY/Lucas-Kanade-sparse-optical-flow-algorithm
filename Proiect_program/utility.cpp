//
// Created by Dogmedve on 2025. 05. 03..
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include "opticalflow.h"

using namespace std;
using namespace cv;

void frames_show(const vector<Mat> &frames) {
    for (int i = 0; i < frames.size(); ++i) {
        char windowName[10];
        sprintf(windowName, "Frame %d", i);

        imshow(windowName, frames[i]);
    }
}

bool isInside(int img_rows, int img_cols, int i, int j) {
    return (i < img_rows && i >= 0 && j < img_cols && j >= 0);
}

Mat bgr_to_grayscale(const Mat &source) {
    Mat result = Mat(source.rows, source.cols, CV_8UC1);

    for (int i = 0; i < source.rows; ++i) {
        for (int j = 0; j < source.cols; ++j) {
            Vec3b pixel = source.at<Vec3b>(i, j);
            result.at<uchar>(i, j) = (pixel[0] + pixel[1] + pixel[2]) / 3;
        }
    }

    return result;
}

Mat grayscale_to_bgr(const Mat &source) {
    Mat result = Mat(source.rows, source.cols, CV_8UC3);

    for (int i = 0; i < source.rows; ++i) {
        for (int j = 0; j < source.cols; ++j) {
            unsigned char pixel = source.at<uchar>(i, j);
            result.at<Vec3b>(i, j) = {pixel, pixel, pixel};
        }
    }

    return result;
}

template<typename T>
Mat gaussianBlur(const Mat &source, int kernel_size, double sigma) {
    Mat result = Mat::zeros(source.rows, source.cols, source.type());
    Mat kernel = Mat::zeros(kernel_size, kernel_size, CV_64F);

    int k = kernel_size / 2;
    double sum = 0.0;
    for (int i = -k; i < k; ++i) {
        for (int j = -k; j < k; ++j) {
            double val = exp(-(i * i + j * j) / (2.0 * sigma * sigma));
            kernel.at<double>(i + k, j + k) = val;
            sum += val;
        }
    }

    Mat sourcePadded;
    copyMakeBorder(source, sourcePadded, k, k, k, k, BORDER_REFLECT_101);

    kernel /= sum;
    for (int i = 0; i < source.rows; ++i) {
        for (int j = 0; j < source.cols; ++j) {
            double pixel = 0.0;
            for (int m = -k; m < k; ++m) {
                for (int n = -k; n < k; ++n) {
                    pixel += kernel.at<double>(m + k, n + k) * sourcePadded.at<T>(i + m + k, j + n + k);
                }
            }

            result.at<T>(i, j) = (T) pixel;
        }
    }

    return result;
}

template Mat gaussianBlur<uchar>(const Mat &source, int kernel_size, double sigma);

template Mat gaussianBlur<double>(const Mat &source, int kernel_size, double sigma);

Mat normalize_intensity(const Mat &source) {
    Mat result = source.clone();

    double minVal, maxVal;
    minVal = maxVal = source.at<uchar>(0, 0);
    for (int i = 0; i < source.rows; ++i) {
        for (int j = 0; j < source.cols; ++j) {
            unsigned char pixel = source.at<uchar>(i, j);
            if (pixel < minVal) {
                minVal = pixel;
            } else if (pixel > maxVal) {
                maxVal = pixel;
            }
        }
    }

    for (int i = 0; i < source.rows; ++i) {
        for (int j = 0; j < source.cols; ++j) {
            result.at<uchar>(i, j) = (uchar) (255.0 * (source.at<uchar>(i, j) - minVal) / (maxVal - minVal));
        }
    }

    return result;
}

Mat apply_kernel(const Mat &src, double G[3][3]) {
    Mat result = Mat::zeros(src.size(), CV_64F);

    for (int y = 1; y < src.rows - 1; ++y) {
        for (int x = 1; x < src.cols - 1; ++x) {
            double sum = 0.0;

            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    double pixel = src.at<double>(y + ky, x + kx);
                    sum += pixel * G[ky + 1][kx + 1];
                }
            }
            result.at<double>(y - 1, x - 1) = sum;
        }
    }

    return result;
}

void calculateGradients(const Mat &source, Mat &grad_x, Mat &grad_y) {
    Mat srcConv;
    source.convertTo(srcConv, CV_64F);
    Mat srcConvPadded;
    copyMakeBorder(srcConv, srcConvPadded, 1, 1, 1, 1, BORDER_REPLICATE);

    double Gx[3][3] = {
            {-1, 0, 1},
            {-2, 0, 2},
            {-1, 0, 1}
    };
    double Gy[3][3] = {
            {-1, -2, -1},
            {0,  0,  0},
            {1,  2,  1}
    };

    //printf("asd\n");
    grad_x = apply_kernel(srcConvPadded, Gx);
    grad_y = apply_kernel(srcConvPadded, Gy);
}