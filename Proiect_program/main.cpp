
#include "preprocessing.h"
#include "opticalflow.h"
#include "cornerpoints.h"

using namespace std;
using namespace cv;

int main() {
    char mode;
    printf("Project run mode:\n(c - convert video to frames;\n r - run algorithms)\n");
    scanf_s(" %c", &mode);

    if (mode == 'c') {
        printf("Converting mp4 file to .jpg frames...\n");

        printf("Conversion result: %d\n",
               convert_video_to_frames(R"(D:\Munkak\An3\Sem2\PI\Proiect\Proiect_program\videos\accident.mp4)",
                                       R"(D:\Munkak\An3\Sem2\PI\Proiect\Proiect_program\accident_frames)"));
    } else if (mode == 'r') {
        printf("Frames to work with (1-6):\n");
        scanf_s(" %c", &mode);

        string folderPath;
        switch (mode) {
            case '1':
                folderPath = R"(D:\Munkak\An3\Sem2\PI\Proiect\Proiect_program\traffic_frames_TEMP)";
                break;
            case '2':
                folderPath = R"(D:\Munkak\An3\Sem2\PI\Proiect\Proiect_program\traffic_frames_fewer_TEMP)";
                break;
            case '3':
                folderPath = R"(D:\Munkak\An3\Sem2\PI\Proiect\Proiect_program\people_walking_frames_TEMP)";
                break;
            case '4':
                folderPath = R"(D:\Munkak\An3\Sem2\PI\Proiect\Proiect_program\seal_bounce_frames)";
                break;
            case '5':
                folderPath = R"(D:\Munkak\An3\Sem2\PI\Proiect\Proiect_program\accident_frames_TEMP1)";
                break;
            case '6':
                folderPath = R"(D:\Munkak\An3\Sem2\PI\Proiect\Proiect_program\accident_frames_TEMP2)";
                break;
            default:
                return 1;
        }

        int status;
        vector<Mat> frames = frames_open(folderPath, status);
        if (status != 0) {
            return 2;
        }

        printf("Running Sparse Optical Flow calculation...\n");
        printf("Preprocessing the frames...\n");
        frames = frames_to_grayscale(frames);
        frames = frames_noise_filter_gaussianBlur(frames, 5, 1.5);
        frames = frames_normalize_intensity(frames);

//        frames_show(frames);

//        vector<Point2f> cornerPoints = detect_cornerPoints(frames.at(0), 100, 0.3, 7);
//        cornerPoints_show(frames.at(0), cornerPoints, "Corner Points");
//
//        waitKey(0);

        printf("Running Lucas Kanade Sparse Optical Flow algorithm...\n");
        const int CORNERPOINT_REFRESH_RATE = -1;
        vector<vector<Point2f>> all_points = frames_optical_flow(frames, 150, 0.3, 7,
                                                                 10, 5, 0.03, 4, CORNERPOINT_REFRESH_RATE);
        //traffic best: 10, 5, 0.03/0.1, 2-3; people walking best: 10, 3, 0.03, 2-3

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
        return 3;
    }

    return 0;

}
