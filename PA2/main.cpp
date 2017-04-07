/*
 * CS440: Artificial Intelligence
 * Programming Assignment 2
 * Team: Miguel Valdez, Sparsh Kumar, Ekaterina Prokopeva
 */

#include <iostream>
#include <unistd.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


int minHue = 0;  //130   //1
int maxHue = 20; //160   //30
int minSaturation = 30;  //10   //10
int maxSaturation = 150; //40   //180
int minValue = 60;   //75     //90
int maxValue = 225; //130    //210

int inAngleMin = 160;
int inAngleMax = 320;
int angleMin = 180;
int angleMax = 345;
int lengthMin = 20;
int lengthMax = 80;

using namespace cv;
using namespace std;

int fontFace = FONT_HERSHEY_COMPLEX_SMALL;  //text arguments
double fontScale = 1.5;
int thickness = 2;

//Point textOrg(0, 30);
//string fingersText = "Fingers: 0";
Point textOrg2(0, 60);
string gameText = "Shoot in ";
Point textOrg3(0, 90);
string handText = "Your hand: ";
Point textOrg4(0, 120);
string vsText = "vs";
Point textOrg5(0, 150);
string computerHandText = "Computer hand: ";
Point textOrg6(0, 200);
string resultText = "Result";
Point textOrg7(0,260);
string scoreText;
Point textOrg8(0,290);
string computerScoreText;

static Mat frame;
size_t fingers;
int countt = 0;

int flag = 0;
int flag2 = 0;
int x = 4;
string hand;
string computerHand;
int handRandom;
int score = 0;
int computerScore = 0;

void myFrameDifferencing(Mat& prev, Mat& curr, Mat& dst);
void myMotionEnergy(vector<Mat> mh, Mat& dst);

float innerAngle(float pointx1, float pointy1,
                 float pointx2, float pointy2,
                 float cpointx1, float cpointy1) {
    
    
    float dist1 = sqrt(  (pointx1-cpointx1)*(pointx1-cpointx1) + (pointy1-cpointy1)*(pointy1-cpointy1) );
    float dist2 = sqrt(  (pointx2-cpointx1)*(pointx2-cpointx1) + (pointy2-cpointy1)*(pointy2-cpointy1) );
    
    float Ax, Ay;
    float Bx, By;
    float Cx, Cy;
    
    Cx = cpointx1;
    Cy = cpointy1;
    
    if(dist1 < dist2) {
        Bx = pointx1;
        By = pointy1;
        Ax = pointx2;
        Ay = pointy2;
        
    } else {
        Bx = pointx2;
        By = pointy2;
        Ax = pointx1;
        Ay = pointy1;
    }
    
    float Q1 = Cx - Ax;
    float Q2 = Cy - Ay;
    float P1 = Bx - Ax;
    float P2 = By - Ay;
    
    float A = acos( (P1*Q1 + P2*Q2) / (sqrt(P1*P1+P2*P2) * sqrt(Q1*Q1+Q2*Q2)));
    A = A * 180 / CV_PI;
    
    return A;
}


//function to let mouse clicks detect color you are looking for.
void CallbackFunc(int event, int x, int y, int flags, void* userdata) {
    
    Mat RGB = frame(Rect(x, y, 1, 1));
    Mat HSV;
    cvtColor(RGB, HSV, CV_BGR2HSV);
    Vec3b pixel = HSV.at<Vec3b>(0, 0);
    
    if (event == EVENT_LBUTTONDOWN) {
        
        int h = pixel.val[0];
        int s = pixel.val[1];
        int v = pixel.val[2];
        if (countt == 0) {
            minHue = h-20;
            maxHue = h+20;
            minSaturation = s-2;
            maxSaturation = s+2;
            minValue = v-2;
            maxValue = v+2;
            countt++;
            
        } else {
            
            if (h < minHue) {
                minHue = h;
                
            } else if (h > maxHue) {
                maxHue = h;
            }
            
            if (s < minSaturation) {
                minSaturation = s;
                
            } else if (s > maxSaturation) {
                maxSaturation = s;
            }
            
            if (v < minValue) {
                minValue = v;
                
            } else if (v > maxValue) {
                maxValue = v;
            }
        }
    }
    
}

/* for palm center detection */
double dist(Point x, Point y)
{
    return (x.x - y.x)*(x.x - y.x) + (x.y - y.y)*(x.y - y.y);
}

pair<Point, double> circleFromPoints(Point p1, Point p2, Point p3)
{
    double offset = pow(p2.x, 2) + pow(p2.y, 2);
    double bc = (pow(p1.x, 2) + pow(p1.y, 2) - offset) / 2.0;
    double cd = (offset - pow(p3.x, 2) - pow(p3.y, 2)) / 2.0;
    double det = (p1.x - p2.x) * (p2.y - p3.y) - (p2.x - p3.x)* (p1.y - p2.y);
    double TOL = 0.0000001;
    if (abs(det) < TOL) { cout << "POINTS TOO CLOSE" << endl; return make_pair(Point(0, 0), 0); }
    
    double idet = 1 / det;
    double centerx = (bc * (p2.y - p3.y) - cd * (p1.y - p2.y)) * idet;
    double centery = (cd * (p1.x - p2.x) - bc * (p2.x - p3.x)) * idet;
    double radius = sqrt(pow(p2.x - centerx, 2) + pow(p2.y - centery, 2));
    
    return make_pair(Point(centerx, centery), radius);
}

Point previousPoint; //stores the center of the palm of the hand of the previous frame
bool previousPointIsSet = false;
Mat drawingFrame; //frame that will hold all lines that have been drawn ???
vector<pair<Point, double>> palm_centers;

int main() {
    
    VideoCapture cap(0);
    // if not successful, exit program
    if (!cap.isOpened())
    {
        cout << "Cannot open the video cam" << endl;
        return -1;
    }
    
    namedWindow("My", WINDOW_AUTOSIZE);
    setMouseCallback("My", CallbackFunc, NULL);
    
    // we need these values to watch the hand movemen for the motion recognition
    int oldCoordinate = 0;
    int newCoordinate = 0;
    
    /* FOR WAVING DETECTION */
    Mat frame0;
    
    // read a new frame from video
    bool bSuccess0 = cap.read(frame0);
    
    //if not successful, break loop
    if (!bSuccess0)
    {
        cout << "Cannot read a frame from video stream" << endl;
    }
    vector<Mat> myMotionHistory;
    Mat fMH1, fMH2, fMH3;
    fMH1 = Mat::zeros(frame0.rows, frame0.cols, CV_8UC1);
    fMH2 = fMH1.clone();
    fMH3 = fMH1.clone();
    myMotionHistory.push_back(fMH1);
    myMotionHistory.push_back(fMH2);
    myMotionHistory.push_back(fMH3);
    
    while (1) {
        bool wavingHand = false;
        
        bool bSuccess = cap.read(frame);
        if (!bSuccess)
        {
            cout << "Cannot read a frame from video stream" << endl;
            break;
        }
        Mat frame1;
        bool bSuccess1 = cap.read(frame1);
        if (!bSuccess1)
        {
            cout << "Cannot read a frame from video stream" << endl;
            break;
        }
        
        
        Mat hsv;
        
        cvtColor(frame, hsv,CV_BGR2HSV);
        inRange(hsv,
                Scalar(minHue, minSaturation, minValue),
                Scalar(maxHue, maxSaturation, maxValue),
                hsv);
        
        int blurSize = 5;               // Pre processing
        int elementSize = 5;
        medianBlur(hsv, hsv, blurSize);
        
        /* START WAVING DETECTION */
        Mat frameDest;
        frameDest = Mat::zeros(frame1.rows, frame1.cols, CV_8UC1);
        myFrameDifferencing(frame0, frame1, frameDest);
        myMotionHistory.erase(myMotionHistory.begin());
        myMotionHistory.push_back(frameDest);
        Mat myMH = Mat::zeros(frame0.rows, frame0.cols, CV_8UC1);
        myMotionEnergy(myMotionHistory, myMH);
        //imshow("MyVideoMH", myMH);
        // count black and white pixels for the motion recognition
        int count_black = 0;
        int count_white = 0;
        for( int y = 0; y < myMH.rows; y++ ) {
            for( int x = 0; x < myMH.cols; x++ ) {
                if ( myMH.at<uchar>(y,x) == 255 ) {
                    // change this to to 'src.atuchar>(y,x) == 255'
                    // if your img has only 1 channel
                    if ( myMH.at<cv::Vec3b>(y,x) == cv::Vec3b(255,255,255) ) {
                        count_white++;
                    }
                    else if ( myMH.at<cv::Vec3b>(y,x) == cv::Vec3b(0,0,0) ) {
                        count_black++;
                    }
                }
            }
        }
        //cout << count_black << endl;
        //cout << count_white << endl;
        // you can play with this value: the ration of black and white pixels.
        float fraction = 0.8;
        if (count_white > count_black * fraction) {
            putText(frame, "Bye", Point(450, 400), FONT_HERSHEY_SIMPLEX, 4, Scalar(0, 255, 255), 10);
            wavingHand = true;
        }
        frame0 = frame1;
        /* END WAVING DETECTION */
        
        Mat element = getStructuringElement(MORPH_ELLIPSE,
                                            Size(2*elementSize+1, 2*elementSize+1),
                                            Point(elementSize, elementSize));
        dilate(hsv, hsv, element);
        
        vector<vector<Point>> contours;        // Contour detection
        vector<Vec4i> hierarchy;
        findContours(hsv, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
        
        int largestContour = 0;
        float largestArea = 0;
        
        // Find the greates contour
        for (int i = 1; i < contours.size(); i++) {
            if (contourArea(contours[i]) > contourArea(contours[largestContour])) {
                largestContour = i; //maxIndex
                largestArea = contourArea(contours[i]);
            }
        }
        
        drawContours(frame, contours, largestContour, Scalar(0, 0, 255), 1);
        
        if (!contours.empty()) {                //Convex Hull
            
            vector<vector<Point>> hull(1);
            convexHull(Mat(contours[largestContour]), hull[0], false);
            drawContours(frame, hull, 0, Scalar(0, 255, 0), 3);
            
            if (hull[0].size() > 2) {
                
                vector<int> hullIndexes;
                convexHull(Mat(contours[largestContour]), hullIndexes, true);
                
                vector<cv::Vec4i> defects;
                //convexity deffects
                cv::convexityDefects(Mat(contours[largestContour]), hullIndexes, defects);
                Rect boundingBox = boundingRect(hull[0]);
                // find the bounding box of the objects
                rectangle(frame, boundingBox, Scalar(255, 0, 0), 2);
                Point center = Point(boundingBox.x + boundingBox.width/2,
                                     boundingBox.y + boundingBox.height/2);
                
                vector<Point> validPoints;
                
                for (int i = 0; i < defects.size(); i++) {
                    
                    Point p1 = contours[largestContour][defects[i][0]];
                    Point p2 = contours[largestContour][defects[i][1]];
                    Point p3 = contours[largestContour][defects[i][2]];
                    double angle = atan2(center.y - p1.y, center.x - p1.x) * 180 / CV_PI;
                    double inAngle = innerAngle(p1.x, p1.y, p2.x, p2.y, p3.x, p3.y);
                    double length = sqrt(pow(p1.x - p3.x, 2) + pow(p1.y - p3.y, 2));
                    
                    if (angle > angleMin - 180 && angle < angleMax -180 &&
                        inAngle > inAngleMin - 180 && inAngle < inAngleMax - 180 &&
                        length > lengthMin / 100.00 * boundingBox.height && length < lengthMax / 100.0 * boundingBox.height) {
                        
                        validPoints.push_back(p1);
                    }
                }
                for (size_t i = 0; i < validPoints.size(); i++) {
                    circle(frame, validPoints[i], 9, Scalar(0, 255, 0), 2);
                    
                }
                fingers = validPoints.size(); // stores the number of fingertips
                
                /* PALM CENTER */
                
                Point rough_palm_center;
                if (defects.size() >= 3){
                    vector<Point> palm_points;
                    for (int j = 0; j<defects.size(); j++)
                    {
                        int startidx = defects[j][0]; Point ptStart(contours[largestContour][startidx]);
                        int endidx = defects[j][1]; Point ptEnd(contours[largestContour][endidx]);
                        int faridx = defects[j][2]; Point ptFar(contours[largestContour][faridx]);
                        //Sum up all the hull and defect points to compute average
                        rough_palm_center += ptFar + ptStart + ptEnd;
                        palm_points.push_back(ptFar);
                        palm_points.push_back(ptStart);
                        palm_points.push_back(ptEnd);
                    }
                    
                    rough_palm_center.x /= defects.size() * 3;
                    rough_palm_center.y /= defects.size() * 3;
                    Point closest_pt = palm_points[0];
                    vector<pair<double, int> > distvec;
                    for (int i = 0; i<palm_points.size(); i++)
                        distvec.push_back(make_pair(dist(rough_palm_center, palm_points[i]), i));
                    sort(distvec.begin(), distvec.end());
                    
                    //Keep choosing 3 points till you find a circle with a valid radius
                    pair<Point, double> soln_circle;
                    for (int i = 0; i + 2<distvec.size(); i++)
                    {
                        Point p1 = palm_points[distvec[i + 0].second];
                        Point p2 = palm_points[distvec[i + 1].second];
                        Point p3 = palm_points[distvec[i + 2].second];
                        soln_circle = circleFromPoints(p1, p2, p3);//Final palm center,radius
                        if (soln_circle.second != 0)
                            break;
                    }
                    //Find avg palm centers for the last few frames to stabilize its centers, also find the avg radius
                    palm_centers.push_back(soln_circle);
                    if (palm_centers.size()>10)
                        palm_centers.erase(palm_centers.begin());
                    Point palm_center;
                    double radius = 0;
                    for (int i = 0; i<palm_centers.size(); i++)
                    {
                        palm_center += palm_centers[i].first;
                        radius += palm_centers[i].second;
                    }
                    palm_center.x /= palm_centers.size();
                    palm_center.y /= palm_centers.size();
                    radius /= palm_centers.size();
                    
                    //Draw the palm center and the palm circle
                    circle(frame, palm_center, 5, Scalar(225, 85, 223), 3);
                    circle(frame, palm_center, radius, Scalar(250, 128, 114), 3);
                    
                    previousPoint = palm_center;
                    previousPointIsSet = true;
                    
                    /* gesture */
                    int temp = oldCoordinate;
                    oldCoordinate = newCoordinate;
                    newCoordinate = palm_center.x - temp;
                    // the distance that indicates that the motion is detected
                    int maxDistance = 1200; // you can play with this value
                    if (abs(newCoordinate - oldCoordinate) >= maxDistance) {
                        //putText(frame, "Motion Detected", Point(30, 70), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 255));
                        //break;
                    }
                    
                }
                /* END OF PALM CENTER */
            }
        }
        
        
        
        
        //fingersText = "Fingers: " + to_string(fingers);
        
        if(fingers <= 1) {
            hand = "Rock";
            handText = "Your hand: " + hand;
            
        } else if(fingers > 1 && fingers < 3) {
            hand = "Scissors";
            handText = "Your hand: " + hand;
            
        } else  {
            hand = "Paper";
            handText = "Your hand: " + hand;
        }
        
        //TextArea
        scoreText = "Your Score: " + to_string(score);
        computerScoreText = "Computer Score: " + to_string(computerScore);
        
        //putText(frame, fingersText, textOrg, fontFace, fontScale, Scalar::all(255), thickness,8);
        putText(frame, gameText, textOrg2, fontFace, fontScale, Scalar::all(255), thickness,8);
        putText(frame, handText, textOrg3, fontFace, fontScale, Scalar::all(255), thickness,8);
        putText(frame, vsText, textOrg4, fontFace, fontScale, Scalar::all(255), thickness,8);
        putText(frame, computerHandText, textOrg5, fontFace, fontScale, Scalar::all(255), thickness,8);
        putText(frame, resultText, textOrg6, fontFace, fontScale, Scalar::all(255), thickness,8);
        putText(frame, scoreText, textOrg7, fontFace, fontScale, Scalar::all(255), thickness,8);
        putText(frame, computerScoreText, textOrg8, fontFace, fontScale, Scalar::all(255), thickness,8);
        
        
        imshow("My", frame);                //show frame
        
        if (waitKey(30) == 27 || wavingHand == true) {            //press esc to exit application
            cout << "esc key is pressed by user" << endl;
            sleep(1.5);
            break;
        }
        
        /* GAME IMPLEMENTATION */
        if (flag2 && flag){             //game logic
            sleep(0.5);
        }
        
        if (waitKey(30) == 127 || flag == 1) {   //press delete to start a game
            if(x > 0) {
                
                gameText = "Shoot in " + to_string(x-1);
                x--;
                flag = 1;
                flag2 = 0;
                sleep(1);
                
            } else {
                
                if(flag2 == 0){
                    
                    handRandom = rand() % 3;
                    
                    if(handRandom == 0) {
                        computerHand = "Rock";
                        computerHandText = "Computer hand: " + computerHand;
                    } else if (handRandom == 1){
                        computerHand = "Scissors";
                        computerHandText = "Computer hand: " + computerHand;
                    } else {
                        computerHand = "Paper";
                        computerHandText = "Computer hand: " + computerHand;
                    }
                    gameText = "Shoot!";
                    
                }
                
                if (flag2){
                    
                    if(hand == computerHand) {
                        resultText = hand + " vs " + computerHand + ": Tie";
                    } else if(hand == "Paper" && computerHand == "Rock") {
                        resultText = hand + " vs " + computerHand + ": You won";
                        score++;
                    } else if(hand == "Paper" && computerHand == "Scissors") {
                        resultText = hand + " vs " + computerHand + ": You lost";
                        computerScore++;
                    } else if(hand == "Rock" && computerHand == "Scissors") {
                        resultText = hand + " vs " + computerHand + ": You won";
                        score++;
                    } else if(hand == "Rock" && computerHand == "Paper") {
                        resultText = hand + " vs " + computerHand + ": You lost";
                        computerScore++;
                    } else if(hand == "Scissors" && computerHand == "Paper") {
                        resultText = hand + " vs " + computerHand + ": You won";
                        score++;
                    } else if(hand == "Scissors" && computerHand == "Rock") {
                        resultText = hand + " vs " + computerHand + ": You lost";
                        computerScore++;
                    }
                    
                    x = 4;
                    flag = 0;
                }
                
                flag2 = 1;
            }
        }
    }
    cap.release();
    return 0;
}

void myFrameDifferencing(Mat& prev, Mat& curr, Mat& dst) {
    //For more information on operation with arrays: http://docs.opencv.org/modules/core/doc/operations_on_arrays.html
    //For more information on how to use background subtraction methods: http://docs.opencv.org/trunk/doc/tutorials/video/background_subtraction/background_subtraction.html
    absdiff(prev, curr, dst);
    Mat gs = dst.clone();
    cvtColor(dst, gs, CV_BGR2GRAY);
    dst = gs > 50;
    Vec3b intensity = dst.at<Vec3b>(200, 200);
}

void myMotionEnergy(vector<Mat> mh, Mat& dst) {
    Mat mh0 = mh[0];
    Mat mh1 = mh[1];
    Mat mh2 = mh[2];
    
    Mat locations;   // output, locations of non-zero pixels
    findNonZero(dst, locations);
    // access pixel coordinates
    //Point pnt = locations.at<Point>(13);
    //cout << pnt << endl;
    
    for (int i = 0; i < dst.rows; i++){
        for (int j = 0; j < dst.cols; j++){
            if (mh0.at<uchar>(i, j) == 255 || mh1.at<uchar>(i, j) == 255 || mh2.at<uchar>(i, j) == 255){
                dst.at<uchar>(i, j) = 255;
            }
            //            if (mh2.at<uchar>(i, j) - mh0.at<uchar>(i, j) == -255) {
            //                cout << "motion" << endl;
            //            }
            Vec3b drawingFP = dst.at<Vec3b>(i, j);
            //cout << dst.at<Vec3b>(i, j) << endl;
            //dst.at<Vec3b>(i, j) = 80;
            //dst.at<Vec3b>(i, j).val[1] = 135;
            //dst.at<Vec3b>(i, j).val[2] = 135;
            
        }
    }
}
