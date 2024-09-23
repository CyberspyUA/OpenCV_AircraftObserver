#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

// Function to calculate the centroid of the detected object
Point2f calculateCentroid(const vector<Point>& contour) {
    Moments m = moments(contour);
    Point2f center;
    center.x = m.m10 / m.m00;
    center.y = m.m01 / m.m00;
    return center;
}

// Function to predict the next position of the aircraft
Point2f predictNextPosition(Point2f currentPos, Point2f prevPos, double timeInterval) {
    Point2f velocity = (currentPos - prevPos) / timeInterval;
    Point2f predictedPos = currentPos + velocity * timeInterval; // Assuming constant velocity
    return predictedPos;
}

int main() {
    // Open video capture (replace with your video file or use a camera)
    VideoCapture cap("x64\\example.mp4"); // Use 0 for webcam or a video file name
    if (!cap.isOpened()) {
        cout << "Error opening video stream" << endl;
        return -1;
    }

    // Variables to store previous and current position of the aircraft
    Point2f prevPos(0, 0), currentPos(0, 0);
    bool firstFrame = true;
    double timeInterval = 1.0 / 30.0; // Assuming 30 FPS

    while (1) {
        Mat frame;
        cap >> frame; // Capture each frame
        if (frame.empty())
            break;

        Mat gray, blur, thresh;
        // Convert to grayscale and blur for better contour detection
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        GaussianBlur(gray, blur, Size(5, 5), 0);

        // Thresholding to find the aircraft (simple threshold for demo)
        threshold(blur, thresh, 200, 255, THRESH_BINARY);

        // Find contours
        vector<vector<Point>> contours;
        findContours(thresh, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        if (!contours.empty()) {
            // Assuming the aircraft is the largest contour
            vector<Point> largestContour = contours[0];
            double maxArea = contourArea(largestContour);
            for (size_t i = 1; i < contours.size(); i++) {
                double area = contourArea(contours[i]);
                if (area > maxArea) {
                    maxArea = area;
                    largestContour = contours[i];
                }
            }

            // Calculate the current centroid of the aircraft
            currentPos = calculateCentroid(largestContour);

            // If it's not the first frame, calculate velocity and predict next position
            if (!firstFrame) {
                Point2f predictedPos = predictNextPosition(currentPos, prevPos, timeInterval);

                // Draw current and predicted position on the frame
                circle(frame, currentPos, 5, Scalar(0, 255, 0), -1); // Current position
                circle(frame, predictedPos, 5, Scalar(0, 0, 255), -1); // Predicted position

                // Draw trajectory (line from current to predicted position)
                line(frame, currentPos, predictedPos, Scalar(255, 0, 0), 2);

                // Update previous position
                prevPos = currentPos;
            } else {
                prevPos = currentPos;
                firstFrame = false;
            }
        }

        // Display the frame with tracking and trajectory
        imshow("Aircraft Tracking", frame);

        // Break the loop when 'q' is pressed
        char c = (char)waitKey(25);
        if (c == 'q')
            break;
    }

    // Release the video capture object
    cap.release();
    destroyAllWindows();

    return 0;
}
