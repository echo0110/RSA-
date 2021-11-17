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

#define IMG_COUNT  1//133		//我们有133张图

#define IMG_WIDTH  (256*2)
#define IMG_HEIGHT (192*2)
//#define JPEG_LIB_VERSION 62


#define IMG_WIDTH1920  1920
#define IMG_HEIGHT1080 1080


char szVisData[IMG_WIDTH * IMG_HEIGHT * 3];
char szInfData[IMG_WIDTH * IMG_HEIGHT * 3];
char szFusionData[IMG_WIDTH * IMG_HEIGHT * 3];


#if 1
int savejpg(uchar *pdata, char *jpg_file, int width, int height)
{  //分别为RGB数据，要保存的jpg文件名，图片长宽
    int depth = 3;
    JSAMPROW row_pointer[1];//指向一行图像数据的指针
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    FILE *outfile;
 
    cinfo.err = jpeg_std_error(&jerr);//要首先初始化错误信息
    //* Now we can initialize the JPEG compression object.
    jpeg_create_compress(&cinfo);
 
    if ((outfile = fopen(jpg_file, "wb")) == NULL)
    {
        fprintf(stderr, "can't open %s\n", jpg_file);
        return -1;
    }
    jpeg_stdio_dest(&cinfo, outfile);
 
    cinfo.image_width = width;             //* image width and height, in pixels
    cinfo.image_height = height;
    cinfo.input_components = depth;    //* # of color components per pixel
    cinfo.in_color_space = JCS_RGB;     //* colorspace of input image
    jpeg_set_defaults(&cinfo);
 
    jpeg_set_quality(&cinfo, JPEG_QUALITY, TRUE ); //* limit to baseline-JPEG values
    jpeg_start_compress(&cinfo, TRUE);
 
    int row_stride = width * 3;
    while (cinfo.next_scanline < cinfo.image_height)
           {
            row_pointer[0] = (JSAMPROW)(pdata + cinfo.next_scanline * row_stride);//一行一行数据的传，jpeg为大端数据格式
            jpeg_write_scanlines(&cinfo, row_pointer, 1);
        }
 
    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);//这几个函数都是固定流程
    fclose(outfile);
 
    return 0;
}
#endif
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
	//RKNN_ImgFusionInit(argv[1]);

	//VideoWriter video("out.avi", CV_FOURCC('M','P','4','2'), 10, Size(IMG_WIDTH, IMG_HEIGHT));
	Mat matBgr(Size(IMG_WIDTH, IMG_HEIGHT), CV_8UC3);
    Mat imagevis(IMG_WIDTH,IMG_HEIGHT,1);
    Mat imageinf(IMG_WIDTH,IMG_HEIGHT,1);
	char szRgbFileName[64];
    char visFileName[64]={0};
    char infFileName[64]={0};
	struct timeval start_time, stop_time;

#if 1
	gettimeofday(&start_time, NULL);
	for (int i = 0; i < IMG_COUNT; i++)
	{
		//分别读取红外和可见光图像
		sprintf(szRgbFileName, "res/inf_%d.rgb", i);
		ReadFile(szRgbFileName, szInfData, sizeof(szInfData));
		sprintf(visFileName, "res/vis_%d.rgb", i);
		ReadFile(szRgbFileName, szVisData, sizeof(szVisData));

        

       
       
        
        Mat image2BGR(Size(IMG_WIDTH, IMG_HEIGHT), CV_8UC3);
        Mat matvis(Size(IMG_WIDTH, IMG_HEIGHT), CV_8UC3, szVisData);
		cvtColor(matvis, image2BGR, COLOR_RGB2BGR);
		//video << matBgr;
        
		//sprintf(szRgbFileName, "fus_%d.jpg", 0);
		cv::imwrite("./img0.jpg", image2BGR); 

        cv::Mat img = image2BGR.clone();
        cv::resize(image2BGR, img, cv::Size(IMG_WIDTH1920, IMG_HEIGHT1080), cv::INTER_LINEAR);
        imwrite("result2.jpg", img); 

        return 0;
        
        Mat matinf( Size(IMG_WIDTH, IMG_HEIGHT), CV_8UC3, szInfData);
        cvtColor(matinf, matBgr, COLOR_RGB2BGR);
        std::vector<Mat> mv_inf;
        split(matinf, (vector<Mat>&)mv_inf);
        //imageinf = mv_inf[0].clone();
        cv::imwrite("./img1.jpg", matinf);
		//执行融合
		RKNN_ImgFusionProcess(imagevis.data, imageinf.data, szFusionData, IMG_WIDTH, IMG_HEIGHT,matinf);

		Mat matRgb(Size(IMG_WIDTH, IMG_HEIGHT), CV_8UC3, szFusionData);
		cvtColor(matRgb, matBgr, COLOR_RGB2BGR);
		//video << matBgr;

		sprintf(szRgbFileName, "fus_%d.jpg", i);
		cv::imwrite(szRgbFileName, matBgr);
		printf("save %s\n", szRgbFileName);
	}
	gettimeofday(&stop_time, NULL);
	printf("run use %f ms , average time: %f ms\n", 
			(__get_us(stop_time) - __get_us(start_time)) / 1000.0,
			(__get_us(stop_time) - __get_us(start_time)) / 1000.0 / IMG_COUNT);

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

