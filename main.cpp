#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/time.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

//#include <opencv2/highgui.hpp>
#include "RkNNImageFusion.h"
//#include "highgui.h"

extern "C" {
#include <jpeglib.h>
}

#define JPEG_QUALITY 100 //图片质量


using namespace std;
using namespace cv;

#define IMG_COUNT  2//133		//我们有133张图


//#define JPEG_LIB_VERSION 62





char szVisData[IMG_WIDTH * IMG_HEIGHT * 3];
char szInfData[IMG_WIDTH * IMG_HEIGHT * 3];
char buff[IMG_WIDTH * IMG_HEIGHT * 3];

void* a = NULL;
void** szFusionData=&a;


int ReadFile(const char *pszName, char *pszData, int iLen)
{
    int fd = -1;
    char szBuf[8] = {0};
    int ret = -1;

    fd = open(pszName, O_RDONLY);
    if (fd == -1)
    {
        printf("open %s failed, %s\n", pszName, strerror(errno));
        return -1;
    }
    ret = read(fd, pszData, iLen);
    if (ret > 0) {
        close(fd);
        return ret;
    }
    close(fd);

    return -1;
}

double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

int main(int argc, char *argv[])
{
//	if (argc < 2)
//	{
//		printf("%s model_name.rknn\n", argv[0]);
//		return -1;
//	}

	//初始化
	RKNN_ImgFusionInit(argv[1]);

	//VideoWriter video("out.avi", CV_FOURCC('M','P','4','2'), 10, Size(IMG_WIDTH, IMG_HEIGHT));
	Mat matBgr(Size(IMG_WIDTH, IMG_HEIGHT), CV_8UC3);
	char szRgbFileName[64];
	struct timeval start_time, stop_time;

#if 1
	gettimeofday(&start_time, NULL);
	for (int i = 1; i < IMG_COUNT; i++)
	{
		//分别读取红外和可见光图像

        sprintf(szRgbFileName, "./inf_512x384_%d.jpg", i);
        printf(" %s,%d, szRgbFileName is %s\n",__func__,__LINE__,szRgbFileName);
        ReadFile(szRgbFileName, szInfData, sizeof(szInfData));
        sprintf(szRgbFileName, "./vis_512x384_%d.jpg", i);
        printf(" %s,%d, szRgbFileName is %s\n",__func__,__LINE__,szRgbFileName);
        ReadFile(szRgbFileName, szVisData, sizeof(szVisData));


        const char *img_path_vis = "/oem/ImageFusion/vis_512x384_1.jpg";
        const char *img_path_ir = "/oem/ImageFusion/inf_512x384_1.jpg";
        // Load image    
        cv::Mat orig_img_vis = imread(img_path_vis);
        if (!orig_img_vis.data)
        {
        printf("cv::imread %s fail!\n", img_path_vis);
        return -1;
        }

        cv::Mat orig_img_ir = imread(img_path_ir);
        if (!orig_img_ir.data)
        {
        printf("cv::imread %s fail!\n", img_path_ir);
        return -1;
        }
        //szFusionData=buff;
        
		//执行融合
		//RKNN_ImgFusionProcess(szVisData, szInfData, szFusionData, IMG_WIDTH, IMG_HEIGHT);
		RKNN_ImgFusionProcess(orig_img_vis.data, orig_img_ir.data, szFusionData, IMG_WIDTH, IMG_HEIGHT);
        int count = 30;
        Mat Img;
        for(int k=0;k<count;k++)
        {
            Mat b(IMG_HEIGHT384,IMG_WIDTH512,CV_32FC1,(float*)(*szFusionData));
            Mat g(IMG_HEIGHT384,IMG_WIDTH512,CV_32FC1,(float*)((*szFusionData)+IMG_WIDTH512*IMG_HEIGHT384*1*sizeof(float)));
            Mat r(IMG_HEIGHT384,IMG_WIDTH512,CV_32FC1,(float*)((*szFusionData)+IMG_WIDTH512*IMG_HEIGHT384*2*sizeof(float)));
            b.convertTo(b,CV_8UC1);
            g.convertTo(g,CV_8UC1);
            r.convertTo(r,CV_8UC1);
    
            vector<Mat> Vecs;
            Vecs.push_back(b);
            Vecs.push_back(g);
            Vecs.push_back(r);
            merge(Vecs,Img);
        }
        sprintf(szRgbFileName, "fus_%d.jpg", i);
		cv::imwrite(szRgbFileName, Img);
		printf("save %s\n", szRgbFileName);
        szFusionData=NULL;
	}
	gettimeofday(&stop_time, NULL);

#else
	//这部分是测试单纯融合100次消耗的时间
	sprintf(szRgbFileName, "res/inf_%d.rgb", 0);
	ReadFile(szRgbFileName, szInfData, sizeof(szInfData));
	sprintf(szRgbFileName, "res/vis_%d.rgb", 0);
	ReadFile(szRgbFileName, szVisData, sizeof(szInfData));

	gettimeofday(&start_time, NULL);
	for (int i = 0; i < 100; i++)
	{
		//执行融合
		RKNN_ImgFusionProcess(szVisData, szInfData, szFusionData, IMG_WIDTH, IMG_HEIGHT);
	}
	gettimeofday(&stop_time, NULL);
	printf("run use %f ms , average time: %f ms\n", 
			(__get_us(stop_time) - __get_us(start_time)) / 1000.0,
			(__get_us(stop_time) - __get_us(start_time)) / 1000.0 / 100);
#endif


	//反初始化
	RKNN_ImgFusionExit();
}

