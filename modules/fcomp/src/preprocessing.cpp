/********************************************
 * FILE NAME: preprocessing.h               *
 * DESCRIPTION:                             *
 * VERSION:                                 *
 * AUTHORS: Victor García                   *
 ********************************************/

#include "../include/preprocessing.h"

void getHistogram(cv::Mat img, int histogram[3][256]){
    int i = 0, j = 0;
    // Initializing the histogram
    for(i=0; i<256; i++){
        histogram[0][i] = 0;
        histogram[1][i] = 0;
        histogram[2][i] = 0;
    }
    // Computing the histogram
    for(i=0; i<img.size().height; i++){
        for(j=0; j<img.size().width; j++){
            histogram[0][img.at<cv::Vec3b>(i,j)[0]] += 1;
            histogram[1][img.at<cv::Vec3b>(i,j)[1]] += 1;
            histogram[2][img.at<cv::Vec3b>(i,j)[2]] += 1;
        }
    }
}

void printHistogram(int histogram[256], std::string filename, cv::Scalar color){
    // Finding the maximum value of the histogram. It will be used to scale the
    // histogram to fit the image.
    int max = 0, i;
    for(i=0; i<256; i++){
        if( histogram[i] > max ) max= histogram[i];
    }
    // Creating an image from the histogram.
    cv::Mat imgHist(1480,1580, CV_8UC3, cv::Scalar(255,255,255));
    cv::Point pt1, pt2;
    pt1.y = 1380;
    for(i=0; i<256; i++){
        pt1.x = 150 + 5*i + 1;
        pt2.x = 150 + 5*i + 3;
        pt2.y = 1380 - 1280 * histogram[i] / max;
        cv::rectangle(imgHist,pt1,pt2,color,CV_FILLED);
    }
    // y-axis labels
    cv::rectangle(imgHist,cv::Point(130,1400),cv::Point(1450,80),cvScalar(0,0,0),1);
    cv::putText(imgHist, std::to_string(max), cv::Point(10,100), cv::FONT_HERSHEY_PLAIN, 1.5, cvScalar(0,0,0), 2.0);
    cv::putText(imgHist, std::to_string(max*3/4), cv::Point(10,420), cv::FONT_HERSHEY_PLAIN, 1.5, cvScalar(0,0,0), 2.0);
    cv::putText(imgHist, std::to_string(max/2), cv::Point(10,740), cv::FONT_HERSHEY_PLAIN, 1.5, cvScalar(0,0,0), 2.0);
    cv::putText(imgHist, std::to_string(max/4), cv::Point(10,1060), cv::FONT_HERSHEY_PLAIN, 1.5, cvScalar(0,0,0), 2.0);
    cv::putText(imgHist, std::to_string(0), cv::Point(10,1380), cv::FONT_HERSHEY_PLAIN, 1.5, cvScalar(0,0,0), 2.0);
    // x-axis labels
    cv::putText(imgHist, std::to_string(0), cv::Point(152-7*1,1430), cv::FONT_HERSHEY_PLAIN, 1.5, cvScalar(0,0,0), 2.0);
    cv::putText(imgHist, std::to_string(63), cv::Point(467-7*2,1430), cv::FONT_HERSHEY_PLAIN, 1.5, cvScalar(0,0,0), 2.0);
    cv::putText(imgHist, std::to_string(127), cv::Point(787-7*3,1430), cv::FONT_HERSHEY_PLAIN, 1.5, cvScalar(0,0,0), 2.0);
    cv::putText(imgHist, std::to_string(191), cv::Point(1107-7*3,1430), cv::FONT_HERSHEY_PLAIN, 1.5, cvScalar(0,0,0), 2.0);
    cv::putText(imgHist, std::to_string(255), cv::Point(1427-7*3,1430), cv::FONT_HERSHEY_PLAIN, 1.5, cvScalar(0,0,0), 2.0);

    // Saving the image
    cv::imwrite(filename, imgHist);
}

void colorChannelStretch(cv::Mat imgOriginal, cv::Mat imgStretched, int lowerPercentile, int higherPercentile){
    // Computing the histograms
    int histogram[3][256];
    getHistogram(imgOriginal, histogram);

    // Computing the percentiles
    int blueLowerPercentile = -1, blueHigherPercentile = -1;
    int greenLowerPercentile = -1, greenHigherPercentile = -1;
    int redLowerPercentile = -1, redHigherPercentile = -1;
    // Blue percentiles
    int i = 0, sum = 0;
    while ( sum < higherPercentile * imgOriginal.size().height * imgOriginal.size().width / 100 ){
        if(sum < imgOriginal.size().height * imgOriginal.size().width / 100) blueLowerPercentile++;
        blueHigherPercentile++;
        sum += histogram[0][i];
        i++;
    }
    // Green percentiles
    i = 0;
    sum = 0;
    while ( sum < higherPercentile * imgOriginal.size().height * imgOriginal.size().width / 100 ){
        if(sum < imgOriginal.size().height * imgOriginal.size().width / 100) greenLowerPercentile++;
        greenHigherPercentile++;
        sum += histogram[1][i];
        i++;
    }
    // Red percentiles
    i = 0;
    sum = 0;
    while ( sum < higherPercentile * imgOriginal.size().height * imgOriginal.size().width / 100 ){
        if(sum < imgOriginal.size().height * imgOriginal.size().width / 100) redLowerPercentile++;
        redHigherPercentile++;
        sum += histogram[2][i];
        i++;
    }

    // Creating the modified image, imgStretched, pixel by pixel
    int j;
    for(i=0; i<imgOriginal.size().height; i++){
        for(j=0; j<imgOriginal.size().width; j++){
            // Blue channel
            if ( imgOriginal.at<cv::Vec3b>(i,j)[0] < blueLowerPercentile) imgStretched.at<cv::Vec3b>(i,j)[0] = 0;
            else if ( imgOriginal.at<cv::Vec3b>(i,j)[0] > blueHigherPercentile ) imgStretched.at<cv::Vec3b>(i,j)[0] = 255;
            else imgStretched.at<cv::Vec3b>(i,j)[0] = ( 255 * ( imgOriginal.at<cv::Vec3b>(i,j)[0] - blueLowerPercentile ) ) / ( blueHigherPercentile - blueLowerPercentile );
            // Gren channel
            if ( imgOriginal.at<cv::Vec3b>(i,j)[1] < greenLowerPercentile) imgStretched.at<cv::Vec3b>(i,j)[1] = 0;
            else if ( imgOriginal.at<cv::Vec3b>(i,j)[1] > greenHigherPercentile ) imgStretched.at<cv::Vec3b>(i,j)[1] = 255;
            else imgStretched.at<cv::Vec3b>(i,j)[1] = ( 255 * ( imgOriginal.at<cv::Vec3b>(i,j)[1] - greenLowerPercentile ) ) / ( greenHigherPercentile - greenLowerPercentile );
            // Red channel
            if ( imgOriginal.at<cv::Vec3b>(i,j)[2] < redLowerPercentile) imgStretched.at<cv::Vec3b>(i,j)[2] = 0;
            else if ( imgOriginal.at<cv::Vec3b>(i,j)[2] > redHigherPercentile ) imgStretched.at<cv::Vec3b>(i,j)[2] = 255;
            else imgStretched.at<cv::Vec3b>(i,j)[2] = ( 255 * ( imgOriginal.at<cv::Vec3b>(i,j)[2] - redLowerPercentile ) ) / ( redHigherPercentile - redLowerPercentile );
        }
    }
}