#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "RkNNImageFusion.h"


#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

#include "rknn_api.h"
#include <iostream>

using namespace std;
using namespace cv;

#define DEMO_INPUT_NUM 2 //4
int MODEL_IN_WIDTH = 1920;
int MODEL_IN_HEIGHT = 1080;

static void printRKNNTensor(rknn_tensor_attr *attr)
{
    printf("index=%d name=%s n_dims=%d dims=[%d %d %d %d] n_elems=%d size=%d fmt=%d type=%d qnt_type=%d fl=%d zp=%d scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[3], attr->dims[2], attr->dims[1], attr->dims[0],
           attr->n_elems, attr->size, 0, attr->type, attr->qnt_type, attr->fl, attr->zp, attr->scale);
}

static unsigned char *load_model(const char *filename, int *model_size)
{
    FILE *fp = fopen(filename, "rb");
    if (fp == nullptr)
    {
        printf("fopen %s fail!\n", filename);
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    int model_len = ftell(fp);
    unsigned char *model = (unsigned char *)malloc(model_len);
    fseek(fp, 0, SEEK_SET);
    if (model_len != fread(model, 1, model_len, fp))
    {
        printf("fread %s fail!\n", filename);
        free(model);
        return NULL;
    }
    *model_size = model_len;
    if (fp)
    {
        fclose(fp);
    }
    return model;
}



int RKNN_ImgFusionInit(const char *pszModelPath)
{
   
	return 0;
}

/*
pVisibleRgbData:		输入可见光图像数据,格式RGB
pInfrareRgbData:		输入红外图像数据,格式RGB
pFusionRgbData:			输出融合图像的数据,格式RGB
uiWidth:				图像宽度
uiHeight:				图像高度
*/
int RKNN_ImgFusionProcess(void *pVisibleRgbData, void *pInfrareRgbData, void *pFusionRgbData, unsigned int uiWidth, unsigned int uiHeight,Mat &matinf)
{
	//todo 这里实现融合算法

    rknn_context ctx;
    int ret;
    int model_len = 0;
    unsigned char *model;
    const char *model_path = "/oem/image_fusion1080p_1126_sim.rknn";

  

    

    
    //todo 初始化加载rknn模型
    // Load RKNN Model
    model = load_model(model_path, &model_len);
    ret = rknn_init(&ctx, model, model_len, 0);
    if (ret < 0)
    {
       printf("rknn_init fail! ret=%d\n", ret);
       return -1;
    }

    // Get Model Input Output Info
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
       printf("rknn_query fail! ret=%d\n", ret);
       return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    printf("input tensors:\n");
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
       input_attrs[i].index = i;
       ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
       if (ret != RKNN_SUCC)
       {
           printf("rknn_query fail! ret=%d\n", ret);
           return -1;
       }
       printRKNNTensor(&(input_attrs[i]));
    }

    printf("output tensors:\n");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++)
    {
       output_attrs[i].index = i;
       ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
       if (ret != RKNN_SUCC)
       {
           printf("rknn_query fail! ret=%d\n", ret);
           return -1;
       }
       printRKNNTensor(&(output_attrs[i]));
    }

    assert(DEMO_INPUT_NUM == io_num.n_input);
    // Set Input struct
    rknn_input inputs[DEMO_INPUT_NUM];
    memset(inputs, 0, DEMO_INPUT_NUM * sizeof(rknn_input));
    for (int i = 0; i < io_num.n_input; ++i)
    {
       int channel = 0;
       if (input_attrs[i].fmt == RKNN_TENSOR_NCHW)
       {
           channel = input_attrs[i].dims[2];
       }
       else
       {
           channel = input_attrs[i].dims[0];
       }
       printf("channel = %d\n",channel);
       int img_size = MODEL_IN_WIDTH * MODEL_IN_HEIGHT * channel;
       inputs[i].index = i;
       inputs[i].type = RKNN_TENSOR_UINT8;
       inputs[i].size = img_size;
       inputs[i].fmt = RKNN_TENSOR_NHWC;
    }
    inputs[0].buf = pVisibleRgbData;
    inputs[1].buf = pInfrareRgbData;
    ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
    if (ret < 0)
    {
        printf("rknn_input_set fail! ret=%d\n", ret);
        return -1;
    }

    // Run
    printf("rknn_run\n");
    ret = rknn_run(ctx, nullptr);
    if (ret < 0)
    {
        printf("rknn_run fail! ret=%d\n", ret);
        return -1;
    }

    // Get Output
    rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 0;
    ret = rknn_outputs_get(ctx, 1, outputs, NULL);
    if (ret < 0)
    {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        return -1;
    }



    std::cout << " bgr H: "<<  matinf.rows << std::endl;
    std::cout << " bgr W: "<< matinf.cols << std::endl;
    std::cout << " bgr C: "<< matinf.channels() << std::endl;



    //inImage: input pic  of inf rgb 
	cv::Mat imageY(MODEL_IN_WIDTH,MODEL_IN_HEIGHT,1);
	cv::Mat imageU(MODEL_IN_WIDTH,MODEL_IN_HEIGHT,1);
	cv::Mat imageV(MODEL_IN_WIDTH,MODEL_IN_HEIGHT,1);	
	
	//cv::Mat image2YUV(orig_img3.rows,orig_img3.cols,3);
	cv::Mat image2YUV;
	cv::cvtColor(matinf,image2YUV,CV_BGR2YUV);
	std::vector<Mat> mv;
	split(matinf, (vector<Mat>&)mv);
	imageY = mv[0].clone();
	imageU = mv[1].clone();
	imageV = mv[2].clone();
    imageY.data=(uchar*)(outputs[0].buf);

    mv[0].data=(uchar*)(outputs[0].buf);

    cv::Mat m3(MODEL_IN_HEIGHT, MODEL_IN_WIDTH, CV_8UC3);
    cv::merge(mv, m3);
    
    cv::imwrite("./imgyuv.jpg", m3);
    

    // Release rknn_outputs
    rknn_outputs_release(ctx, 1, outputs);

    // Release
    if (ctx >= 0)
    {
       rknn_destroy(ctx);
    }
    if (model)
    {
       free(model);
    }
	//memcpy(pFusionRgbData, pVisibleRgbData, uiWidth * uiHeight * 3);

	return 0;
}

void RKNN_ImgFusionExit(void)
{
	//todo
	
    
}

