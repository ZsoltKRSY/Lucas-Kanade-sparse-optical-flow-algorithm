//
// Created by Dogmedve on 2025. 03. 26..
//

#ifndef PROIECT_PROGRAM_PREPROCESSING_H
#define PROIECT_PROGRAM_PREPROCESSING_H

#include <opencv2/opencv.hpp>
#include "utility.h"

using namespace std;
using namespace cv;

int convert_video_to_frames(const char *videoPath, const string &folderPath);

vector<Mat> frames_open(const string &folderPath, int &status);

vector<Mat> frames_to_grayscale(const vector<Mat> &source);

vector<Mat> frames_noise_filter_gaussianBlur(const vector<Mat> &source, int kernel_size, double sigma);

vector<Mat> frames_normalize_intensity(const vector<Mat> &source);

#endif //PROIECT_PROGRAM_PREPROCESSING_H
