//
//  ViewController.h
//  MUDetection
//
//  Created by kid on 11/20/15.
//  Copyright © 2015 kid. All rights reserved.
//

#import <UIKit/UIKit.h>

#ifdef __cplusplus
#import <opencv2/opencv.hpp>
#include "opencv2/contrib/contrib.hpp"
#import "opencv2/highgui/ios.h"
#endif

@interface ViewController : UIViewController<CvVideoCameraDelegate>


@end

