#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// Function to normalize a 2D point
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
    vector<Point2f> previousCenters; // For tracking movement
    Point2f previousDirection(0, 0); // To smooth direction estimation
    while (true)
    {
        // Capture frame-by-frame
        cap >> frame;
        // If the frame is empty, break the loop
        if (frame.empty()) break;
        // Apply background subtraction
        pBackSub->apply(frame, fgMask);
        Mat hsvFrame;
        cvtColor(frame, hsvFrame, COLOR_BGR2HSV);
        // Define range for sky color (for clearer sky filtering)
        Scalar lowerSky(100, 30, 150);
        Scalar upperSky(140, 255, 255);
        Mat skyMask;
        inRange(hsvFrame, lowerSky, upperSky, skyMask);
        // Define range for tree color to attempt masking them out
        Scalar lowerTree(0, 10, 0);
        Scalar upperTree(100, 255, 130);
        Mat treeMask;
        inRange(hsvFrame, lowerTree, upperTree, treeMask);

        // Combine both masks and invert them to remove sky and trees from fgMask
        Mat combinedMask = skyMask | treeMask;
        bitwise_not(combinedMask, combinedMask); // Invert mask
        bitwise_and(fgMask, combinedMask, fgMask); // Apply combined mask to foreground mask
        // Remove noise
        erode(fgMask, fgMask, getStructuringElement(MORPH_RECT, Size(3, 3)));
        dilate(fgMask, fgMask, getStructuringElement(MORPH_RECT, Size(5, 5)));
        // Find contours of the detected foreground objects
        vector<vector<Point>> contours;
        findContours(fgMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        // List to store valid bounding boxes
        vector<RotatedRect> validBoundingBoxes;
        for (size_t i = 0; i < contours.size(); i++) 
        {
            // Ignore small contours
            if (contourArea(contours[i]) < 2000) continue;
            // Get the minimum area rectangle for the contour
            RotatedRect minRect = minAreaRect(contours[i]);
            // Flexible aspect ratio filtering to match the plane's shape at different angles
            double aspectRatio = (double)minRect.size.width / (double)minRect.size.height;
            if (aspectRatio > 0.5 && aspectRatio < 10.0) 
            {  // Loosen aspect ratio to accommodate different angles
                validBoundingBoxes.push_back(minRect); // Save the valid rotated rectangle
            }
        }
        // Track the movement of the aircraft and estimate nose direction
        for (const auto& minRect : validBoundingBoxes) {
            Point2f rectPoints[4];
            minRect.points(rectPoints);
            // Calculate the center of the bounding box
            Point2f center = minRect.center;
            // Identify the nose direction based on the longest edge of the bounding box
            Point2f edge1 = rectPoints[0] - rectPoints[1];  // Vector for one side of the rectangle
            Point2f edge2 = rectPoints[1] - rectPoints[2];  // Vector for the adjacent side
            // Determine which edge is longer
            Point2f noseDir;
            if (norm(edge1) >= norm(edge2)) 
            {
                noseDir = normalize(rectPoints[0] - rectPoints[1]);  // Normalize to get direction
            }
        	else 
            {
                noseDir = normalize(rectPoints[1] - rectPoints[2]);
            }
            // Smooth out the direction using the previous direction to prevent abrupt changes
            float smoothingFactor = 1.1f;
            noseDir = (previousDirection * smoothingFactor) + (noseDir * (1 - smoothingFactor));
            previousDirection = normalize(noseDir);
            // Extrapolate the trajectory using the smoothed nose direction
            Point2f futurePosition = center + noseDir * 50;
            // Draw the current bounding box
            for (int j = 0; j < 4; j++) 
            {
                line(frame, rectPoints[j], rectPoints[(j + 1) % 4], Scalar(0, 255, 0), 2);
            }
            // Draw the predicted trajectory (future positions based on nose direction)
            line(frame, center, futurePosition, Scalar(255, 0, 0), 2);  // Blue line indicating the trajectory
            circle(frame, futurePosition, 4, Scalar(0, 0, 255), -1);  // Red circle for predicted position
            // Track the centers for consistency filtering
            previousCenters.push_back(center);
            // Limit the size of the history of tracked centers
            if (previousCenters.size() > 50) 
            {
                previousCenters.erase(previousCenters.begin());
            }
        }
        // Show the current frame with bounding boxes and trajectory
        imshow("Plane Detection with nose-Based Trajectory", frame);
        // Show the foreground mask (debugging)
        imshow("Foreground Mask", fgMask);
        if (waitKey(30) == 'q')
            break;
    }
    cap.release();
    destroyAllWindows();
}

int main(int argc, char** argv)
{
    VideoCapture cap("x64\\example.mp4");
    detectAircraftCalculateTrajectory(cap);
    return 0;
}
