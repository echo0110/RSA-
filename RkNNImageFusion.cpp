#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <sys/time.h>

#include "RkNNImageFusion.h"


#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

#include "rknn_api.h"
#include <iostream>
#include <unistd.h>

using namespace std;
using namespace cv;





#define DEMO_INPUT_NUM 2 //4





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
int RKNN_ImgFusionProcess(void *pVisibleRgbData, void *pInfrareRgbData, void *pFusionRgbData, unsigned int uiWidth, unsigned int uiHeight)
{
	//todo 这里实现融合算法

    rknn_context ctx;
    int ret;
    int model_len = 0;
    unsigned char *model;
    const char *model_path = "/oem/image_fusion1080p_1126_sim.rknn";

    Mat image2BGR;
    Mat matvis(Size(IMG_WIDTH, IMG_HEIGHT), CV_8UC3, pVisibleRgbData);
    cvtColor(matvis, image2BGR, COLOR_RGB2BGR);
    cv::Mat img_vis = matvis.clone();
    cv::resize(matvis, img_vis, cv::Size(IMG_WIDTH1920, IMG_HEIGHT1080), cv::INTER_LINEAR);
    imwrite("vis_0.jpg", img_vis); 
    sleep(5);
    cv::Mat orig_img = imread("/oem/vis_0.jpg",0);//vis

    imwrite("vis_01.jpg", orig_img);
    

    if (img_vis.cols != IMG_WIDTH1920 || img_vis.rows != IMG_HEIGHT1080)
    {
        printf("resize %d %d to %d %d\n", img_vis.cols, img_vis.rows, IMG_WIDTH1920, IMG_HEIGHT1080);
        //cv::resize(img_vis, img, cv::Size(IMG_WIDTH1920, IMG_HEIGHT1080), cv::INTER_LINEAR);
    }


    Mat image3BGR( Size(IMG_WIDTH, IMG_HEIGHT), CV_8UC3);
    Mat matinf(Size(IMG_WIDTH, IMG_HEIGHT), CV_8UC3, pInfrareRgbData);
    cvtColor(matinf, image3BGR, COLOR_RGB2BGR);
    cv::Mat img_inf = image3BGR.clone();
    cv::resize(image3BGR, img_inf, cv::Size(IMG_WIDTH1920, IMG_HEIGHT1080), cv::INTER_LINEAR);
    imwrite("inf_0.jpg", img_inf);
    cv::Mat img_inf2 = imread("./inf_0.jpg",IMREAD_GRAYSCALE);//inf
    imwrite("inf_01.jpg", img_inf2);
    return 0;
    
    
	
//    std::vector<Mat> img_inf_mv;
//    split(img_inf, (vector<Mat>&)img_inf_mv);
//
//    cv::Mat img_inf_yuv;
//	cv::cvtColor(img_inf,img_inf_yuv,CV_BGR2YUV);
//    std::vector<Mat> img_inf_mv;
//    split(img_inf, (vector<Mat>&)img_inf_mv);
    
    
    
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
       int img_size = IMG_WIDTH1920 * IMG_HEIGHT1080 * channel;
       inputs[i].index = i;
       inputs[i].type = RKNN_TENSOR_UINT8;
       inputs[i].size = img_size;
       inputs[i].fmt = RKNN_TENSOR_NCHW;
       inputs[0].pass_through = 0;
    }
    inputs[0].buf = orig_img.data; /*vis*/
    inputs[1].buf = img_inf2.data; /*inf*/

    cv::Mat vis= Mat(IMG_WIDTH1920, IMG_HEIGHT1080, CV_8UC1, inputs[0].buf);
    cv::imwrite("./inputs_vis.jpg", vis);

    cv::Mat inf= Mat(IMG_WIDTH1920, IMG_HEIGHT1080, CV_8UC1, inputs[1].buf);
    cv::imwrite("./inputs_inf.jpg", inf);
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



    std::cout << " bgr H: "<<  img_inf.rows << std::endl;
    std::cout << " bgr W: "<< img_inf.cols << std::endl;
    std::cout << " bgr C: "<< img_inf.channels() << std::endl;


//    img_inf_mv[0].data=(uchar*)(outputs[0].buf);
//
//    cv::Mat m3(IMG_WIDTH1920, IMG_HEIGHT1080, CV_8UC3);
//    cv::merge(img_inf_mv, m3);
//    
//    cv::imwrite("./imgyuv.jpg", m3);


    Mat temp= Mat(IMG_WIDTH1920, IMG_HEIGHT1080, CV_8UC1,  outputs[0].buf);
    cv::imwrite("./output.jpg", temp);

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

