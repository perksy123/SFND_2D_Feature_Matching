/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"
#include "RingBuffer.h"

using namespace std;

#define RUN_AS_PERFORMANCE_EVALUATION

#ifdef RUN_AS_PERFORMANCE_EVALUATION
std::tuple<std::vector<int>, std::vector<double>, std::vector<double>> ProcessImages(string & detectorType, string &descType)
{
    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    RingBuffer<DataFrame> dataBuffer(dataBufferSize);
    bool bVis = false;            // visualize results
    /* MAIN LOOP OVER ALL IMAGES */
    std::vector<int> matchCounts;
    std::vector<double> detectorTimes;
    std::vector<double> descriptorTimes;
    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
    {
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = imgGray;
        dataBuffer.push_back(frame);

        /* DETECT IMAGE KEYPOINTS */

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image

        double t = (double)cv::getTickCount();
        if (detectorType.compare("SHITOMASI") == 0)
        {
            detKeypointsShiTomasi(keypoints, imgGray, false);
        }
        else
        {
            if (detectorType.compare("HARRIS") == 0)
            {
                detKeypointsHarris(keypoints, imgGray, false);
            }
            else
            {
                detKeypointsModern(keypoints,imgGray, detectorType, false);
            }            
        }
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        detectorTimes.push_back(1000 * t / 1.0);


        // only keep keypoints on the preceding vehicle
        bool bFocusOnVehicle = true;
        //                   x    y    wid  Hei
        cv::Rect vehicleRect(535, 180, 180, 150);
        if (bFocusOnVehicle)
        {
            std::vector<cv::KeyPoint> filtered;
            for (std::vector<cv::KeyPoint>::iterator it = keypoints.begin(); it != keypoints.end(); ++it)
            {
                cv::KeyPoint &kpt = *it;
                if ( (kpt.pt.x > vehicleRect.x && kpt.pt.x < (vehicleRect.x + vehicleRect.width)) &&
                     (kpt.pt.y > vehicleRect.y && kpt.pt.y < (vehicleRect.y + vehicleRect.height)) )
                {
                    filtered.push_back(kpt);
                }
            }

            keypoints = filtered;
        }

        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = false;
        if (bLimitKpts)
        {
            int maxKeypoints = 50;

            if (detectorType.compare("SHITOMASI") == 0)
            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << " NOTE: Keypoints have been limited!" << endl;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;

        cv::Mat descriptors;
        t = (double)cv::getTickCount();
        descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descType);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        descriptorTimes.push_back(1000 * t / 1.0);

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;

        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {
            /* MATCH KEYPOINT DESCRIPTORS */

            vector<cv::DMatch> matches;
            string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
            string descriptorType = "DES_BINARY"; // DES_BINARY, DES_HOG
            string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN

            matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                             matches, descriptorType, matcherType, selectorType);

            //// EOF STUDENT ASSIGNMENT

            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;

            matchCounts.push_back(matches.size());

            // visualize matches between current and previous image
            bVis = false;
            if (bVis)
            {
                cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                matches, matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                string windowName = "Matching keypoints between two camera images";
                cv::namedWindow(windowName, 7);
                cv::imshow(windowName, matchImg);
                cout << "Press key to continue to next image" << endl;
                cv::waitKey(0); // wait for key to be pressed
            }
            bVis = false;
        }

    } // eof loop over all images

    return std::tuple<std::vector<int>, std::vector<double>, std::vector<double>>(matchCounts, detectorTimes, descriptorTimes);
}
#endif

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{
#ifndef RUN_AS_PERFORMANCE_EVALUATION

    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    RingBuffer<DataFrame> dataBuffer(dataBufferSize);
    bool bVis = false;            // visualize results
#endif

#ifdef RUN_AS_PERFORMANCE_EVALUATION
    std::vector<string> detectorTypes = {"HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"};
    std::vector<string> descriptorTypes = {"BRISK", "BRIEF", "ORB", "AKAZE", "SIFT"};   // Missing FREAK as I'm getting 'Not Implemented' exception
    ofstream resultsFile;
    resultsFile.open("Results.dat");
    cout << "Detector Type, Descriptor Type" << endl;
    resultsFile << "Detector Type, Descriptor Type" << endl;
    for (std::vector<string>::iterator detectorIt = detectorTypes.begin(); detectorIt != detectorTypes.end(); ++detectorIt)
    {
        string detectorType = *detectorIt;
     
        for (std::vector<string>::iterator descriptorIt = descriptorTypes.begin(); descriptorIt != descriptorTypes.end(); ++descriptorIt)
        {
            string descriptorType = *descriptorIt;
            if ( !((detectorType.compare("HARRIS") == 0 &&      // I get a core dump when I try any of these combinations
                   (descriptorType.compare("AKAZE") == 0 || descriptorType.compare("SIFT") == 0)) ||
                   (detectorType.compare("FAST") == 0 &&
                   (descriptorType.compare("AKAZE") == 0 || descriptorType.compare("SIFT") == 0)) ||
                   (detectorType.compare("BRISK") == 0 &&
                   (descriptorType.compare("AKAZE") == 0 || descriptorType.compare("SIFT") == 0)) ||
                   (detectorType.compare("ORB") == 0 &&
                   (descriptorType.compare("AKAZE") == 0 || descriptorType.compare("SIFT") == 0)) ||
                   (detectorType.compare("AKAZE") == 0 &&
                   (descriptorType.compare("SIFT") == 0)) ||
                   (detectorType.compare("SIFT") == 0 &&
                   (descriptorType.compare("ORB") == 0 || descriptorType.compare("AKAZE") == 0 || descriptorType.compare("SIFT") == 0)))
                   )
            {
                std::tuple<std::vector<int>, std::vector<double>, std::vector<double>> results = ProcessImages(detectorType, descriptorType);

                cout << detectorType << ", " << descriptorType << ", ";
                resultsFile << detectorType << ", " << descriptorType << ", ";
                std::vector<int> matchCount = get<0>(results);
                for (std::vector<int>::iterator it = matchCount.begin(); it != matchCount.end(); ++it)
                {
                    cout << *it << ", ";
                    resultsFile << *it << ", ";
                }

                cout << "Detector Times, ";
                resultsFile << "Detector Times, ";
                std::vector<double> detectorTimes = get<1>(results);
                for (std::vector<double>::iterator it = detectorTimes.begin(); it != detectorTimes.end(); ++it)
                {
                    cout << *it << ", ";
                    resultsFile << *it << ", ";
                }

                cout << "Descriptor Times, ";
                resultsFile << "Descriptor Times, ";
                std::vector<double> descriptorTimes = get<2>(results);
                for (std::vector<double>::iterator it = descriptorTimes.begin(); it != descriptorTimes.end(); ++it)
                {
                    cout << *it << ", ";
                    resultsFile << *it << ", ";
                }

                cout << endl;
                resultsFile << endl;
            }
        }
    }
#endif

#ifndef RUN_AS_PERFORMANCE_EVALUATION
    /* MAIN LOOP OVER ALL IMAGES */

    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
    {
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        //// STUDENT ASSIGNMENT
        //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = imgGray;
        dataBuffer.push_back(frame);

        //// EOF STUDENT ASSIGNMENT
        cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

        /* DETECT IMAGE KEYPOINTS */

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image
//        string detectorType = "SHITOMASI";
//        string detectorType = "HARRIS";
        string detectorType = "FAST";
//        string detectorType = "BRISK";
//        string detectorType = "ORB";
//        string detectorType = "AKAZE";
//        string detectorType = "FREAK";    // Using this causes a 'Feature Not Implemented' exception.
//        string detectorType = "SIFT";

        //// STUDENT ASSIGNMENT
        //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
        //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT

        if (detectorType.compare("SHITOMASI") == 0)
        {
            detKeypointsShiTomasi(keypoints, imgGray, false);
        }
        else
        {
            if (detectorType.compare("HARRIS") == 0)
            {
                detKeypointsHarris(keypoints, imgGray, false);
            }
            else
            {
                detKeypointsModern(keypoints,imgGray, detectorType, false);
            }
            
        }
        //// EOF STUDENT ASSIGNMENT

        //// STUDENT ASSIGNMENT
        //// TASK MP.3 -> only keep keypoints on the preceding vehicle

        // only keep keypoints on the preceding vehicle
        bool bFocusOnVehicle = true;
        //                   x    y    wid  Hei
        cv::Rect vehicleRect(535, 180, 180, 150);
//        cv::Rect vehicleRect(550, 180, 150, 150);
        if (bFocusOnVehicle)
        {
            std::vector<cv::KeyPoint> filtered;
            for (std::vector<cv::KeyPoint>::iterator it = keypoints.begin(); it != keypoints.end(); ++it)
            {
                cv::KeyPoint &kpt = *it;
                if ( (kpt.pt.x > vehicleRect.x && kpt.pt.x < (vehicleRect.x + vehicleRect.width)) &&
                     (kpt.pt.y > vehicleRect.y && kpt.pt.y < (vehicleRect.y + vehicleRect.height)) )
                {
                    filtered.push_back(kpt);
                }
            }

            keypoints = filtered;
        }

        cout << "No Of KeyPoints Found in Image # " << imgIndex + 1 << " Pts = " << keypoints.size() << endl;

        //// EOF STUDENT ASSIGNMENT

        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = false;
        if (bLimitKpts)
        {
            int maxKeypoints = 50;

            if (detectorType.compare("SHITOMASI") == 0)
            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << " NOTE: Keypoints have been limited!" << endl;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;
        cout << "#2 : DETECT KEYPOINTS done" << endl;

        /* EXTRACT KEYPOINT DESCRIPTORS */

        //// STUDENT ASSIGNMENT
        //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
        //// -> BRIEF, ORB, FREAK, AKAZE, SIFT

        cv::Mat descriptors;
    //    string descType = "BRISK"; // BRIEF, ORB, FREAK, AKAZE, SIFT
        string descType = "BRIEF"; 
    //    string descType = "ORB"; 
    //    string descType = "AKAZE"; 
    //    string descType = "FREAK"; 
    //    string descType = "SIFT"; 
        descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descType);
        //// EOF STUDENT ASSIGNMENT

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;

        cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {

            /* MATCH KEYPOINT DESCRIPTORS */

            vector<cv::DMatch> matches;
            string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
            string descriptorType = "DES_BINARY"; // DES_BINARY, DES_HOG
            string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN

            //// STUDENT ASSIGNMENT
            //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
            //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp
            if (descType.compare("SIFT") == 0)
            {
                cv::Mat descSource = (dataBuffer.end() - 2)->descriptors.clone();
                cv::Mat descRef = (dataBuffer.end() - 1)->descriptors.clone();
                descSource.convertTo(descSource, CV_32F);
                descRef.convertTo(descRef, CV_32F);

                matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                                 descSource, descRef,
                                 matches, descriptorType, matcherType, selectorType);
            }
            else
            {
                matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                                (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                                matches, descriptorType, matcherType, selectorType);
            }

            //// EOF STUDENT ASSIGNMENT

            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;

            cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

            // visualize matches between current and previous image
            bVis = true;
            if (bVis)
            {
                cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                matches, matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                string windowName = "Matching keypoints between two camera images";
                cv::namedWindow(windowName, 7);
                cv::imshow(windowName, matchImg);
                cout << "Press key to continue to next image" << endl;
                cv::waitKey(0); // wait for key to be pressed
            }
            bVis = false;
        }

    } // eof loop over all images
#endif
    return 0;
}
