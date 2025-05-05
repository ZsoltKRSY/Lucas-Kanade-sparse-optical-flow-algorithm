//
// Created by Dogmedve on 2025. 05. 03..
//

#ifndef PROIECT_PROGRAM_CORNERPOINTS_H
#define PROIECT_PROGRAM_CORNERPOINTS_H

#include <opencv2/opencv.hpp>
#include "utility.h"

using namespace std;
using namespace cv;

vector<Point2f> detect_cornerPoints(const Mat &img, int max_corners, double quality, int min_distance);

void cornerPoints_show(const Mat &img, const vector<Point2f> &cornerPoints, const char *title);

#endif //PROIECT_PROGRAM_CORNERPOINTS_H
