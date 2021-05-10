#include <iostream>
// #include "/usr/local/opt/libomp/include/omp.h"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include <vector>
#include <math.h>

#include <chrono>
using namespace std::chrono;
using namespace cv;
using namespace std;

float var(int hist[],int level,float val,int pix_num )
{
    long long total=pix_num*val;
    int n=0;
    long long m=0;
    for(int i=0;i<level;i++)
    {
        m+=i*hist[i];
        n+=hist[i];
    }
    long long rem=total-m;
    int rempix=pix_num-n;
    float w0=(1.0*n)/(1.0*pix_num);
    float w1=(1.0*rem)/(1.0*pix_num);
    float u0=(1.0*m)/(1.0*n);
    float u1=(1.0*rem)/(1.0*rempix);
    return w0*w1*(u0-u1)*(u0-u1);
}



int main()
{
    // timing purposes
    auto durationP0 = duration_cast<microseconds>(high_resolution_clock::now() - high_resolution_clock::now());
    auto durationP1 = duration_cast<microseconds>(high_resolution_clock::now() - high_resolution_clock::now());
    auto durationP2 = duration_cast<microseconds>(high_resolution_clock::now() - high_resolution_clock::now());
    auto durationP3 = duration_cast<microseconds>(high_resolution_clock::now() - high_resolution_clock::now());
    auto durationP4 = duration_cast<microseconds>(high_resolution_clock::now() - high_resolution_clock::now());

    auto A_t0 = high_resolution_clock::now();

    // loop over all pictures
    // # pragma omp parallel for
    for(int cur_img = 1; cur_img < 671; cur_img++) {

        if(cur_img % 50 == 0) {
	    cout << "Now processing " << cur_img << endl;
	}

        /**************** Step 0 ******************/
        /************ read input image ************/
        /******************************************/

        auto p0_t0 = high_resolution_clock::now();
        string name="./cell_imgs/" + std::to_string(cur_img) + ".png";
        Mat img = imread(name);
        cvtColor(img,img,COLOR_RGB2GRAY); // convert color to RGB
        auto p0_t1 = high_resolution_clock::now();
        durationP0 += duration_cast<microseconds>(p0_t1 - p0_t0);
        
        

        /**************** Step 1 *******************/
        /***** build pixel intensity histogram *****/
        /*******************************************/

        long long u = 0;
        int hist[256];   // histogram with all possible colors
        for(int i = 0; i < 256; i++)
            hist[i] = 0;

        //# pragma omp parallel for
        for(int i = 0;i < img.rows; i++)
        {
            for(int j = 0; j < img.cols; j++)
            {
                int intensity = img.at<uchar>(i,j);
                u += intensity;
                hist[intensity]++;
            }
        }
        auto p1_t1 = high_resolution_clock::now();
        durationP1 += duration_cast<microseconds>(p1_t1 - p0_t1);
        



        /**************** Step 2 *******************/
        /************* find threshold **************/
        /*******************************************/

        int pix_num = img.rows * img.cols;
        float val = (1.0 * u)/float(pix_num);
        float max = 0;
        int threshold = 0;

        //# pragma omp parallel for
        for(int i = 1; i < 255; i++)
        {
            int x = var(hist, i, val, pix_num);
            if(x > max)
            {
                max = x;
                threshold = i;
            }
        }
        auto p2_t1 = high_resolution_clock::now();
        durationP2 += duration_cast<microseconds>(p2_t1 - p1_t1);
        

        
        /**************** Step 3 *******************/
        /********** final segmentation *************/
        /*******************************************/
        
        auto p3_t0 = high_resolution_clock::now();

        //# pragma omp parallel for
        for(int i=0;i<img.rows;i++)
        {
            for(int j=0;j<img.cols;j++)
            {
                if(img.at<uchar>(i,j)>threshold)
                {
                    img.at<uchar>(i,j)=255;
                }
                else
                    img.at<uchar>(i,j)=0;
            }
        }
        auto p3_t1 = high_resolution_clock::now();
        durationP3 += duration_cast<microseconds>(p3_t1 - p2_t1);
        

        
        /**************** Step 4 *******************/
        /********** final segmentation *************/
        /*******************************************/

        auto p4_t0 = high_resolution_clock::now();
        imwrite("otsu_out/" + std::to_string(cur_img) + "out.png",img);
        auto p4_t1 = high_resolution_clock::now();
        durationP4 += duration_cast<microseconds>(p4_t1 - p3_t1);
    }

    auto A_t1 = high_resolution_clock::now();
    auto durationAll = duration_cast<microseconds>(A_t1 - A_t0);

    // print out timing statistics
    cout << "Overall time: " << durationAll.count() << endl;

    cout << "Part 0 - data load time: " << durationP0.count() << endl;
    cout << "         taking " << durationP0.count()*1.0/durationAll.count() << " of the total time." << endl;

    cout << "Part 1 - Compute histogram time: " << durationP1.count() << endl;
    cout << "         taking " << durationP1.count()*1.0/durationAll.count() << " of the total time." << endl;

    cout << "Part 2 - finding threshold time: " << durationP2.count() << endl;
    cout << "         taking " << durationP2.count()*1.0/durationAll.count() << " of the total time." << endl;

    cout << "Part 3 - final segmentation time: " << durationP3.count() << endl;
    cout << "         taking " << durationP3.count()*1.0/durationAll.count() << " of the total time." << endl;

    cout << "Part 4 - data save time: " << durationP4.count() << endl;
    cout << "         taking " << durationP4.count()*1.0/durationAll.count() << " of the total time." << endl;

    return 0;
}
