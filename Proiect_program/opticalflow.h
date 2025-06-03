//
// Created by Dogmedve on 2025. 04. 25..
//

#ifndef PROIECT_PROGRAM_OPTICALFLOW_H
#define PROIECT_PROGRAM_OPTICALFLOW_H

#include <opencv2/opencv.hpp>
#include "utility.h"

using namespace std;
using namespace cv;

vector<Point2f>
calculate_optical_flow(const Mat &prevImg, const Mat &nextImg, const vector<Point2f> &prevPoints,
                       int window_size, int max_iters, double epsilon, int nr_levels);

vector<vector<Point2f>>
frames_optical_flow(const vector<Mat> &frames, int max_corners, double quality, int min_distance, int window_size,
                    int max_iters, double epsilon, int nr_levels, int cornerPoint_refresh_rate);

vector<Mat>
frames_show_optical_flow(const vector<Mat> &frames, const vector<vector<Point2f>> &all_points,
                         int cornerPoint_refresh_rate, int cornerPoint_visual_refresh_rate, bool points);

#endif //PROIECT_PROGRAM_OPTICALFLOW_H
