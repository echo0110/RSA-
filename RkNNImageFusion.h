#ifndef RKNN_IMAGE_FUSION_H_
#define RKNN_IMAGE_FUSION_H_
/*
return: 				0: 表示成功, -1:表示失败
*/



#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

#include "rknn_api.h"
#include <iostream>
using namespace cv;

int RKNN_ImgFusionInit(const char *pszModelPath);

/*
pVisibleRgbData:		输入可见光图像数据,格式RGB
pInfrareRgbData:		输入红外图像数据,格式RGB
pFusionRgbData:			输出融合图像的数据,格式RGB
uiWidth:				图像宽度
uiHeight:				图像高度
return: 				0: 表示成功, -1:表示失败
*/
int RKNN_ImgFusionProcess(void *pVisibleRgbData, void *pInfrareRgbData, void *pFusionRgbData, unsigned int uiWidth, unsigned int uiHeight,Mat &matinf);

void RKNN_ImgFusionExit(void);

#endif


