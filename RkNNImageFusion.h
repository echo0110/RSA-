#ifndef RKNN_IMAGE_FUSION_H_
#define RKNN_IMAGE_FUSION_H_
/*
return: 				0: 表示成功, -1:表示失败
*/


#define IMG_WIDTH  (256*2)
#define IMG_HEIGHT (192*2)

#define IMG_WIDTH1920  1920
#define IMG_HEIGHT1080 1080

#define IMG_WIDTH512  512
#define IMG_HEIGHT384 384



#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

#include "rknn_api.h"
#include <iostream>

#include <string>
#include <iostream>

using namespace cv;

extern double __get_us(struct timeval t);


int RKNN_ImgFusionInit(const char *pszModelPath);

/*
pVisibleRgbData:		输入可见光图像数据,格式RGB
pInfrareRgbData:		输入红外图像数据,格式RGB
pFusionRgbData:			输出融合图像的数据,格式RGB
uiWidth:				图像宽度
uiHeight:				图像高度
return: 				0: 表示成功, -1:表示失败
*/
int RKNN_ImgFusionProcess(void *pVisibleRgbData,void *pInfrareRgbData, void **pFusionRgbData, unsigned int uiWidth, unsigned int uiHeight);
//char* RKNN_ImgFusionProcess(void *pVisibleRgbData, void *pInfrareRgbData, unsigned int uiWidth, unsigned int uiHeight);

void RKNN_ImgFusionExit(void);
void *thread2(void *arg);


 
std::string base64_encode(unsigned char const* , unsigned int len);
std::string base64_decode(std::string const& s);

//extern  std::string base64_chars;




#endif


