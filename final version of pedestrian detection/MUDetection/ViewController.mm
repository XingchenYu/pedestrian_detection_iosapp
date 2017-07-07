//
//  ViewController.m
//  MUDetection
//
//  Created by kid on 11/20/15.
//  Copyright © 2015 kid. All rights reserved.
//

#import "ViewController.h"
#import <mach/mach_time.h>
#import <GPUImage/GPUImage.h>

// Include stdlib.h and std namespace so we can mix C++ code in here
#include <stdlib.h>
#include <numeric>

//#include "armadillo" // Includes the armadillo library
using namespace cv;
using namespace std;
//using namespace arma;

const Scalar YELLOW = Scalar(0,255,255);
const Scalar RED = Scalar(0,0,255);
const Scalar GREEN = Scalar(0,255,0);
const Scalar BLUE = Scalar(255,0,0);

@interface ViewController()
{
    GPUImageView *imageView_;
    UIImageView *liveView_; // Live output from the camera
    CvVideoCamera *videoCamera;
    cv::Mat RGface;
    Ptr<FaceRecognizer> model;
    vector<cv::Mat> images;
    vector<int> labels;
    uint64_t prevTime;
    UITextView *fpsView_; // Display the current FPS
    int64 curr_time_;

}

@end

@implementation ViewController


//===============================================================================================
// Setup view for excuting App
- (void)viewDidLoad {
    [super viewDidLoad];
    
    // Take into account size of camera input
    int view_width = self.view.frame.size.width;
    int view_height = (640*view_width)/480;
    int offset = (self.view.frame.size.height - view_height)/2;

    
    prevTime=0;
    
    
    // 1. Setup the your OpenCV view, so it takes up the entire App screen......
    int view_offset = (self.view.frame.size.height - view_height)/2;
    liveView_ = [[UIImageView alloc] initWithFrame:CGRectMake(0.0, view_offset, view_width, view_height)];
    
    
    
    
    [self.view addSubview:liveView_]; // Important: add liveView_ as a subview
    liveView_.hidden=false;
    
    
        
        
    videoCamera = [[CvVideoCamera alloc] initWithParentView:liveView_];
    videoCamera.defaultAVCaptureDevicePosition = AVCaptureDevicePositionBack;
    videoCamera.defaultAVCaptureSessionPreset = AVCaptureSessionPreset640x480;
    videoCamera.defaultAVCaptureVideoOrientation = AVCaptureVideoOrientationPortrait;
    videoCamera.defaultFPS = 30;
    videoCamera.grayscaleMode = NO;
    videoCamera.delegate = self;
    videoCamera.rotateVideo = YES;
    fpsView_ = [[UITextView alloc] initWithFrame:CGRectMake(0,15,view_width,std::max(offset,35))];
    [fpsView_ setOpaque:false]; // Set to be Opaque
    [fpsView_ setBackgroundColor:[UIColor clearColor]]; // Set background color to be clear
    [fpsView_ setTextColor:[UIColor redColor]]; // Set text to be RED
    [fpsView_ setFont:[UIFont systemFontOfSize:18]]; // Set the Font size
    [self.view addSubview:fpsView_];
    //videoCamera.recordVideo = YES;
    [videoCamera start];
    
    
//    // 1. Setup the your OpenCV view, so it takes up the entire App screen......
//    int view_offset = (self.view.frame.size.height - view_height)/2;
//    //    liveView_ = [[UIImageView alloc] initWithFrame:CGRectMake(0.0, view_offset, view_width, view_height)];
//    imageView_ = [[GPUImageView alloc] initWithFrame:CGRectMake(0.0, 0.0, self.view.frame.size.width, self.view.frame.size.height)];
//    [self.view addSubview:imageView_];
//    
//    //    [self.view addSubview:liveView_]; // Important: add liveView_ as a subview
//    imageView_.hidden=false;
//    
//    
//    
//    videoCamera = [[CvVideoCamera alloc] initWithParentView:imageView_];
//    
//    //    videoCamera = [[CvVideoCamera alloc] initWithParentView:liveView_];
//    videoCamera.defaultAVCaptureDevicePosition = AVCaptureDevicePositionBack;
//    videoCamera.defaultAVCaptureSessionPreset = AVCaptureSessionPreset640x480;
//    videoCamera.defaultAVCaptureVideoOrientation = AVCaptureVideoOrientationPortrait;
//    videoCamera.defaultFPS = 30;
//    videoCamera.grayscaleMode = NO;
//    videoCamera.delegate = self;
//    videoCamera.rotateVideo = YES;
//    fpsView_ = [[UITextView alloc] initWithFrame:CGRectMake(0,15,view_width,std::max(offset,35))];
//    [fpsView_ setOpaque:false]; // Set to be Opaque
//    [fpsView_ setBackgroundColor:[UIColor clearColor]]; // Set background color to be clear
//    [fpsView_ setTextColor:[UIColor redColor]]; // Set text to be RED
//    [fpsView_ setFont:[UIFont systemFontOfSize:18]]; // Set the Font size
//    [self.view addSubview:fpsView_];
//    //videoCamera.recordVideo = YES;
//    [videoCamera start];
   
}

//===============================================================================================

- (void)processImage:(cv::Mat &)image{

    // You can apply your OpenCV code HERE!!!!!
    // If you want, you can ignore the rest of the code base here, and simply place
    // your OpenCV code here to process images.
    cv::CascadeClassifier pede_cascade; // Cascade classifier for detecting the pedestrian
    NSString* cascadePath1 = [[NSBundle mainBundle]
                              pathForResource:@"lbp_pedestrian1" ofType:@"xml"];
    pede_cascade.load([cascadePath1 UTF8String]);
    
    



    cv::Mat cvImage=image;
    cv::Mat gray;
    cv::cvtColor(cvImage, gray, CV_RGBA2GRAY); // Convert to grayscale
    cv::Mat im = gray;
    cv::Mat display_im=cvImage;
    
    vector<cv::Rect> pedes;
    cv::Mat frame_gray=im;
    equalizeHist( frame_gray, frame_gray );
    
    
    
    
    
    cv::GaussianBlur(frame_gray, frame_gray, cv::Size(5, 5), 5, 5);
    
    
    
    pede_cascade.detectMultiScale( frame_gray, pedes, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE,cv::Size(50, 100) );
    vector<cv::Rect> n_pedes;
    
    
    //non-max supression
    
    if (pedes.size() == 0) {
        n_pedes = pedes;
    } else {
        int len = pedes.size();
        vector<int> pick;
        vector<int> x1;
        vector<int> y1;
        vector<int> x2;
        vector<int> y2;
        for (int i = 0; i < len; i++) {
            x1.push_back(pedes[i].x);
            x2.push_back(pedes[i].x + pedes[i].width);
            y1.push_back(pedes[i].y);
            y2.push_back(pedes[i].y + pedes[i].height);
        }
        vector<int> area;
        for (int i = 0; i < len; i++) {
            area.push_back((x2[i] - x1[i] + 1) * (y2[i] - y1[i] + 1));
        }
        
        
        vector<int> idx = ysort(y2);
        
        while(idx.size() > 0) {
            int last = idx.size() - 1;
            int i = idx[last];
            pick.push_back(i);
            vector<int> sup {last};
            
            for (int p = 0; p < last; p++) {
                int j = idx[p];
                int xx1 = max(x1[i], x1[j]);
                int yy1 = max(y1[i], y1[j]);
                int xx2 = min(x2[i], x2[j]);
                int yy2 = min(y2[i], y2[j]);
                
                int w = max(0, xx2 - xx1 + 1);
                int h = max(0, yy2 - yy1 + 1);
                
                float overlap = (float)(w * h) / (float)area[j];
                if (overlap > 0.3) {
                    sup.push_back(p);
                }
                
            }
            idx = delete_sup(idx, sup);
            
        }
        for (int n = 0; n < pick.size(); n++) {
            n_pedes.push_back(pedes[pick[n]]);
        }

    }
    
    
    if(n_pedes.size() > 0) {
        for(int i=0; i<n_pedes.size(); i++) {
            std::cout << n_pedes[i] << std::endl;
            cv::Point ellipse_middle(n_pedes[i].x + n_pedes[i].width / 2, n_pedes[i].y + n_pedes[i].height - n_pedes[i].width / 4);
            double ridus1 = n_pedes[i].width / 2;
            double ridus2 = n_pedes[i].width / 4;
            ellipse(display_im, ellipse_middle, cv::Size(ridus1, ridus2), 0, 0, 3606, YELLOW);
            
            rectangle(display_im, n_pedes[i], YELLOW,2,8,0);
        }
    }
    
    
    // Finally estimate the frames per second (FPS)
    int64 next_time = getTickCount(); // Get the next time stamp
    float fps = (float)getTickFrequency()/(next_time - curr_time_); // Estimate the fps
    curr_time_ = next_time; // Update the time
    NSString *fps_NSStr = [NSString stringWithFormat:@"FPS = %2.2f",fps];
    
    
    
    // Have to do this so as to communicate with the main thread
    // to update the text display
    dispatch_sync(dispatch_get_main_queue(), ^{
        fpsView_.text = fps_NSStr;
    });
    
    
    image =display_im;

}

vector<int> ysort(vector<int> y2) {
    int l = y2.size();
//    vector<int> y2_v;
//    for (int i = 0; i < l; i++) {
//        y2_v[i] = y2[i];
//    }
    vector<int> idx(l);
    iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(),
         [&y2](int i1, int i2) {return y2[i1] < y2[i2];});
    
    return idx;
}

vector<int> delete_sup(vector<int> idx, vector<int> sup) {
    //    vector<int>::iterator it;
    //
    //    for(it=idx.begin(); it!=idx.end();) {
    //
    //    }
    for (int i = 0; i < sup.size(); i++) {
        int p = sup[i];
        idx[p] = -1;
    }
    vector<int> new_idx;
    for (int j = 0; j < idx.size(); j++) {
        if (idx[j] >= 0) {
            new_idx.push_back(idx[j]);
        }
    }
    return new_idx;
}



//===============================================================================================
// Standard memory warning component added by Xcode
- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

@end
