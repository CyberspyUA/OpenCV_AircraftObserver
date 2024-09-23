#include <opencv2/opencv.hpp>
#include <iostream>
#include <deque>

using namespace cv;
using namespace std;

// Function to normalize a 2D point (vector)
Point2f normalize(Point2f v)
{
    float magnitude = sqrt(v.x * v.x + v.y * v.y);
    return v / magnitude;
}

void detectAircraftCalculateTrajectory(VideoCapture& cap)
{
    if (!cap.isOpened()) {
        cout << "Error opening video stream or file" << endl;
        return;
    }

    Ptr<BackgroundSubtractor> pBackSub = createBackgroundSubtractorMOG2(500, 16, false);
    Mat frame, fgMask;
    deque<Point2f> previousCenters;  // Track movement history
    Point2f previousDirection(0, 0); // For smoothing the direction estimation
    const float timeInterval = 1.0 / 30.0; // Assuming 30 FPS video for time between frames

    while (true) {
        // Capture frame-by-frame
        cap >> frame;
        if (frame.empty()) break;

        // Apply background subtraction
        pBackSub->apply(frame, fgMask);
        Mat hsvFrame;
        cvtColor(frame, hsvFrame, COLOR_BGR2HSV);

        // Define color ranges for sky and trees
        Scalar lowerSky(80, 30, 150), upperSky(140, 255, 255);
        Scalar lowerTree(0, 10, 0), upperTree(100, 255, 130);
        
        // Create masks for sky and trees
        Mat skyMask, treeMask;
        inRange(hsvFrame, lowerSky, upperSky, skyMask);
        inRange(hsvFrame, lowerTree, upperTree, treeMask);
        
        // Combine masks and apply to foreground
        Mat combinedMask = skyMask | treeMask;
        bitwise_not(combinedMask, combinedMask); // Invert
        bitwise_and(fgMask, combinedMask, fgMask);

        // Remove noise
        erode(fgMask, fgMask, getStructuringElement(MORPH_RECT, Size(3, 3)));
        dilate(fgMask, fgMask, getStructuringElement(MORPH_RECT, Size(5, 5)));

        // Find contours of detected objects
        vector<vector<Point>> contours;
        findContours(fgMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        vector<RotatedRect> validBoundingBoxes;
        for (size_t i = 0; i < contours.size(); i++) 
        {
            if (contourArea(contours[i]) < 2000) continue;
            RotatedRect minRect = minAreaRect(contours[i]);

            double aspectRatio = (double)minRect.size.width / (double)minRect.size.height;
            if (aspectRatio > 0.5 && aspectRatio < 10.0) 
            {
                validBoundingBoxes.push_back(minRect);
            }
        }

        for (const auto& minRect : validBoundingBoxes) 
        {
            Point2f rectPoints[4];
            minRect.points(rectPoints);

            Point2f center = minRect.center;
            Point2f edge1 = rectPoints[0] - rectPoints[1];
            Point2f edge2 = rectPoints[1] - rectPoints[2];
            Point2f noseDir = (norm(edge1) >= norm(edge2)) ? normalize(edge1) : normalize(edge2);
            noseDir = (previousDirection * 0.9f) + (noseDir * 0.1f);
            previousDirection = normalize(noseDir);

            // Extrapolate trajectory using current position and velocity
            Point2f futurePosition = center + noseDir * 80;
            for (int j = 0; j < 4; j++) {
                line(frame, rectPoints[j], rectPoints[(j + 1) % 4], Scalar(0, 255, 0), 2);
            }
            line(frame, center, futurePosition, Scalar(255, 0, 0), 2);
            circle(frame, futurePosition, 4, Scalar(0, 0, 255), -1);

            // Add center to history for velocity calculation
            previousCenters.push_back(center);
            if (previousCenters.size() > 50) {
                previousCenters.pop_front();
            }
        }

        imshow("Aircraft Detection with Kinematic Trajectory", frame);
        imshow("Foreground Mask", fgMask);
        if (waitKey(30) == 'q') break;
    }

    cap.release();
    destroyAllWindows();
}

int main(int argc, char** argv) {
    VideoCapture cap("x64\\example.mp4");
    detectAircraftCalculateTrajectory(cap);
    return 0;
}
