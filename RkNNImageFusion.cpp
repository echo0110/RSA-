#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "RkNNImageFusion.h"

int RKNN_ImgFusionInit(const char *pszModelPath)
{
	//todo 初始化加载rknn模型
	return 0;
}

/*
pVisibleRgbData:		输入可见光图像数据,格式RGB
pInfrareRgbData:		输入红外图像数据,格式RGB
pFusionRgbData:			输出融合图像的数据,格式RGB
uiWidth:				图像宽度
uiHeight:				图像高度
*/
int RKNN_ImgFusionProcess(void *pVisibleRgbData, void *pInfrareRgbData, void *pFusionRgbData, unsigned int uiWidth, unsigned int uiHeight)
{
	//todo 这里实现融合算法
	memcpy(pFusionRgbData, pVisibleRgbData, uiWidth * uiHeight * 3);

	return 0;
}

void RKNN_ImgFusionExit(void)
{
	//todo 
}

