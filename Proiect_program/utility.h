//
// Created by Dogmedve on 2025. 05. 03..
//

#ifndef PROIECT_PROGRAM_UTILITY_H
#define PROIECT_PROGRAM_UTILITY_H

#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

void frames_show(const vector<Mat> &frames);

bool isInside(int img_rows, int img_cols, int i, int j);

Mat bgr_to_grayscale(const Mat &source);

Mat grayscale_to_bgr(const Mat &source);

template<typename T>
Mat gaussianBlur(const Mat &source, int kernel_size, double sigma);

Mat normalize_intensity(const Mat &source);

void calculateGradients(const Mat &source, Mat &grad_x, Mat &grad_y);

#endif //PROIECT_PROGRAM_UTILITY_H
