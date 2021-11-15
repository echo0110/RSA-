#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "RkNNImageFusion.h"


#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

#include "rknn_api.h"

using namespace std;
using namespace cv;

#define DEMO_INPUT_NUM 2 //4
const int MODEL_IN_WIDTH = 1920;
const int MODEL_IN_HEIGHT = 1080;


int RKNN_ImgFusionInit(const char *pszModelPath)
{
    const int MODEL_IN_WIDTH = 1920;
    const int MODEL_IN_HEIGHT = 1080;

    rknn_context ctx;
    int ret;
    int model_len = 0;
    unsigned char *model;
    const char *model_path = argv[1];
    const char *img_path = argv[2];
    const char *img2_path = argv[3];
    const char *img3_path = argv[4];
    
    
    // Load image
    cv::Mat orig_img = imread(img2_path,0);//vis
    cv::Mat orig_img2 = imread(img_path,0);//inf
    cv::Mat orig_img3 = imread(img3_path);//inf 3 channel
    if (!orig_img.data)
    {
        printf("cv::imread %s fail!\n", img_path);
        return -1;
    }
     if (!orig_img2.data)
    {
        printf("cv::imread %s fail!\n", img2_path);
        return -1;
    }

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
    std::vector<cv::Mat> imgs;
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

    inputs[0].buf = orig_img.data;
    inputs[1].buf = orig_img2.data;

	struct timeval start_time, stop_time;
    gettimeofday(&start_time, NULL);
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


    Mat vis= Mat(MODEL_IN_HEIGHT, MODEL_IN_WIDTH, CV_8UC1, inputs[0].buf);
    cv::imwrite("./inputs0.jpg", vis);

    Mat inf= Mat(MODEL_IN_HEIGHT, MODEL_IN_WIDTH, CV_8UC1, inputs[1].buf);
    cv::imwrite("./inputs1.jpg", inf);
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



    std::cout << " bgr H: "<<  orig_img3.rows << std::endl;
    std::cout << " bgr W: "<< orig_img3.cols << std::endl;
    std::cout << " bgr C: "<< orig_img3.channels() << std::endl;



    //inImage: input pic  of inf rgb 
	cv::Mat imageY(orig_img3.rows,orig_img2.cols,1);
	cv::Mat imageU(orig_img3.rows,orig_img2.cols,1);
	cv::Mat imageV(orig_img3.rows,orig_img2.cols,1);	
	
	//cv::Mat image2YUV(orig_img3.rows,orig_img3.cols,3);
	cv::Mat image2YUV;
	cv::cvtColor(orig_img3,image2YUV,CV_BGR2YUV);
	std::vector<Mat> mv;
	split(orig_img3, (vector<Mat>&)mv);
	imageY = mv[0].clone();
	imageU = mv[1].clone();
	imageV = mv[2].clone();
    imageY.data=(uchar*)(outputs[0].buf);

    mv[0].data=(uchar*)(outputs[0].buf);

    cv::Mat m3(MODEL_IN_HEIGHT, MODEL_IN_WIDTH, CV_8UC3);
    cv::merge(mv, m3);
    






     
	

    Mat temp= Mat(MODEL_IN_HEIGHT, MODEL_IN_WIDTH, CV_8UC1,  outputs[0].buf);
    cv::imwrite("./imgoutput.jpg", temp);
    Mat imgy= Mat(MODEL_IN_HEIGHT, MODEL_IN_WIDTH, CV_8UC1,  imageY.data);
    cv::imwrite("./imgy.jpg", imgy);

    Mat imgu= Mat(MODEL_IN_HEIGHT, MODEL_IN_WIDTH, CV_8UC1,  imageU.data);
    cv::imwrite("./imgu.jpg", imgu);

    Mat imgv= Mat(MODEL_IN_HEIGHT, MODEL_IN_WIDTH, CV_8UC1,  imageV.data);
    cv::imwrite("./imgv.jpg", imgv);

    Mat imgyuv= Mat(MODEL_IN_HEIGHT, MODEL_IN_WIDTH, CV_8UC3,  imageV.data);
    cv::imwrite("./imgyuv.jpg", m3);
    
    // Release rknn_outputs
    rknn_outputs_release(ctx, 1, outputs);
	memcpy(pFusionRgbData, pVisibleRgbData, uiWidth * uiHeight * 3);

	return 0;
}

void RKNN_ImgFusionExit(void)
{
	//todo
	
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
}

