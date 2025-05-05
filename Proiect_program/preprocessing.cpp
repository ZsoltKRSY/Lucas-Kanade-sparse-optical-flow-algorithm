//
// Created by Dogmedve on 2025. 03. 26..
//

#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "preprocessing.h"

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

int convert_video_to_frames(const char *videoPath, const char *folderPath) {
    VideoCapture capture(videoPath);
    if (!capture.isOpened()) {
        printf("Error: could not open video file.\n");
        return -1;
    }

    if (!fs::exists(folderPath)) {
        fs::create_directory(folderPath);
    }

    Mat frame;
    int nrFrames = 0;

    while (capture.read(frame)) {
        ostringstream fileName;
        fileName << folderPath << "/frame_" << std::setw(4) << std::setfill('0') << nrFrames << ".jpg";

        imwrite(fileName.str(), frame);
        ++nrFrames;
    }

    capture.release();

    return 0;
}

vector<Mat> frames_open(const char *folderPath) {
    vector<Mat> frames;

    if (!fs::exists(folderPath) || !fs::is_directory(folderPath)) {
        printf("Error: folder does not exist or is not a directory.\n");
        return frames;
    }

    for (const auto &entry: fs::directory_iterator(folderPath)) {
        if (entry.is_regular_file()) {
            string filePath = entry.path().string();

            Mat frame = imread(filePath, IMREAD_COLOR_BGR);
            if (frame.empty()) {
                printf("Error: could not read image %s\n", filePath.c_str());
                continue;
            }

            frames.push_back(frame);
        }
    }

    return frames;
}

vector<Mat> frames_to_grayscale(const vector<Mat> &source) {
    vector<Mat> result;

    for (const auto &frame: source) {
        result.push_back(bgr_to_grayscale(frame));
    }

    return result;
}

vector<Mat> frames_noise_filter_gaussianBlur(const vector<Mat> &source, int kernel_size, double sigma) {
    vector<Mat> result;

    for (const auto &frame: source) {
        result.push_back(gaussianBlur<uchar>(frame, kernel_size, sigma));
    }

    return result;
}

vector<Mat> frames_normalize_intensity(const vector<Mat> &source) {
    vector<Mat> result;

    for (const auto &frame: source) {
        result.push_back(normalize_intensity(frame));
    }

    return result;
}
