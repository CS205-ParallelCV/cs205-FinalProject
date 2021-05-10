#include <iostream>
// #include "/usr/local/opt/libomp/include/omp.h"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include <vector>
#include <math.h>
#include <omp.h>
#include <mpi.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>

#include <chrono>
using namespace std::chrono;
using namespace cv;
using namespace std;

float var(int hist[],int level,float val,int pix_num )
{
    long long total=pix_num*val;
    int n=0;
    long long m=0;
    # pragma omp parallel for
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



int main(int argc, char **argv)
{
    int my_rank, num_procs;

    /* Initialize the infrastructure necessary for communication */
    MPI_Init(&argc, &argv);

    /* Identify this process */
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    /* Find out how many total processes are active */
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int start_pic, end_pic;
    int partition = 670 / num_procs;

    if (my_rank != num_procs) {
        start_pic = my_rank * partition+1;
        end_pic = (my_rank+1) * partition;
    } else {
        start_pic = my_rank * partition+1;
        end_pic = 670;
    }

    // loop over all pictures
    # pragma omp parallel for
    for(int cur_img = start_pic; cur_img <= end_pic; cur_img++) {

        if(cur_img % 50 == 0) {
            cout << "Now processing " << cur_img << endl;
	    }

        /**************** Step 0 ******************/
        /************ read input image ************/
        /******************************************/

        string name="./cell_imgs/" + std::to_string(cur_img) + ".png";
        Mat img = imread(name);
        cvtColor(img,img,COLOR_RGB2GRAY); // convert color to RGB
        

        /**************** Step 1 *******************/
        /***** build pixel intensity histogram *****/
        /*******************************************/

        long long u = 0;
        int hist[256];   // histogram with all possible colors
        for(int i = 0; i < 256; i++)
            hist[i] = 0;

        # pragma omp parallel for
        for(int i = 0; i < img.rows; i++)
        {
            for(int j = 0; j < img.cols; j++)
            {
                int intensity = img.at<uchar>(i,j*2);
                u += intensity;
                hist[intensity]++;
            }

        }
        
        /**************** Step 2 *******************/
        /************* find threshold **************/
        /*******************************************/

        int pix_num = img.rows * img.cols;
        float val = (1.0 * u)/float(pix_num);
        float max = 0;
        int threshold = 0;

        for(int i = 1; i < 255; i++)
        {
            int x = var(hist, i, val, pix_num);
            if(x > max)
            {
                max = x;
                threshold = i;
            }
        }
        
        /**************** Step 3 *******************/
        /********** final segmentation *************/
        /*******************************************/

        for(int i=0;i<img.rows;i++) {
            for(int j=0;j<img.cols;j++) {
                if(img.at<uchar>(i,j)>threshold) {
                    img.at<uchar>(i,j)=255;
                }
                else {
                    img.at<uchar>(i,j)=0;
                }
            }
        }
        
        /**************** Step 4 *******************/
        /********** final segmentation *************/
        /*******************************************/

        imwrite("otsu_openmp_out/" + std::to_string(cur_img) + "out.png",img);
    }

    return 0;
}
