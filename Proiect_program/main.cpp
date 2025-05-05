
#include "preprocessing.h"
#include "opticalflow.h"
#include "cornerpoints.h"

using namespace std;
using namespace cv;

int main() {
    char mode;
    scanf_s("%c", &mode);

    if (mode == 'c') {
        printf("Converting mp4 file to .jpg frames...\n");

        printf("Conversion result: %d\n",
               convert_video_to_frames(R"(D:\Munkak\An3\Sem2\PI\Proiect\Proiect_program\people_walking.mp4)",
                                       R"(D:\Munkak\An3\Sem2\PI\Proiect\Proiect_program\people_walking_frames)"));
    } else if (mode == 'r') {
        printf("Running Sparse Optical Flow calculation...\n");

        vector<Mat> frames = frames_open(R"(D:\Munkak\An3\Sem2\PI\Proiect\Proiect_program\people_walking_frames_TEMP)");

        printf("Preprocessing the frames...\n");
        frames = frames_to_grayscale(frames);
        frames = frames_noise_filter_gaussianBlur(frames, 5, 1.5);
        frames = frames_normalize_intensity(frames);

//    frames_show(frames);
//
//        vector<Point2f> cornerPoints = detect_cornerPoints(frames.at(0), 100, 0.3, 7);
//        cornerPoints_show(frames.at(0), cornerPoints, "Corner Points");
//
//        waitKey(0);

        printf("Running Lucas Kanade Sparse Optical Flow algorithm...\n");
        const int CORNERPOINT_REFRESH_RATE = -1;
        vector<vector<Point2f>> all_points = frames_optical_flow(frames, 250, 0.3, 7,
                                                                 10, 3, 0.2, CORNERPOINT_REFRESH_RATE);
        //traffic best: 15, 3, 0.03/0.1; people walking best: 10, 3, 0.2

        printf("Running optical flow visualization algorithm...\n");
        vector<Mat> optical_flow_frames = frames_show_optical_flow(frames, all_points, CORNERPOINT_REFRESH_RATE,
                                                                   CORNERPOINT_REFRESH_RATE,
                                                                   false);

        while (true) {
            for (const auto &frame: optical_flow_frames) {
                imshow("Optical Flow Visualized", frame);

                int key = waitKey(75);
                if (key == 27) {
                    destroyAllWindows();
                    return 0;
                }
            }
        }
    } else {
        return 0;
    }

}
