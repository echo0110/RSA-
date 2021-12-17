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


const int MODEL_IN_WIDTH = 512;
const int MODEL_IN_HEIGHT = 384;




#define DEMO_INPUT_NUM 2 //4

rknn_input inputs[DEMO_INPUT_NUM];

rknn_context ctx;

rknn_input_output_num io_num;

int ret=0;

unsigned char *model;


rknn_output outputs[1];

char hardCde[200];


char* RSADecryptString(char *ciphertext)
{
    FILE *stream = NULL;
    char buf[100];
    char *pstring;
    memset(buf, 0, sizeof(buf));
    char cmd[1024];
    memset(cmd, 0, sizeof(cmd));
    snprintf(cmd, sizeof(cmd), "./cryptest.exe rd priv_key %s", ciphertext);
    if ((stream = popen(cmd, "r")) == NULL) {
        fprintf(stderr, "%s", strerror(errno));
    }
    /* output the message */
    while (fgets(buf, sizeof(buf), stream) != NULL) {
    }
    pstring=strdup((char *)buf);
    return pstring;
}

void getSerialNum(char buf[]){
    FILE *stream = NULL;
    memset(buf, 0, sizeof(buf));
    char cmd[100];
    memset(cmd, 0, sizeof(cmd));
    sprintf(cmd, "cat /proc/cpuinfo | grep Serial");
    if ((stream = popen(cmd, "r")) == NULL) {
        fprintf(stderr,"func is %s,%d, %s",__func__,__LINE__, strerror(errno));
    }
    /* output the message */
    while (fgets(buf, 100, stream) != NULL) {
    }
}





static void printRKNNTensor(rknn_tensor_attr *attr)
{
    printf("index=%d name=%s n_dims=%d dims=[%d %d %d %d] n_elems=%d size=%d fmt=%d type=%d qnt_type=%d fl=%d zp=%d scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[3], attr->dims[2], attr->dims[1], attr->dims[0],
           attr->n_elems, attr->size, 0, attr->type, attr->qnt_type, attr->fl, attr->zp, attr->scale);
}

//static unsigned char *load_model(const char *filename, int *model_size)
static unsigned char *load_model(FILE *fp, int *model_size)
{
    char buf[1024];
//    FILE *fp = fopen(filename, "rb");
    
    if (fp == nullptr)
    {
        printf("fopen %s fail!\n", __func__);
        return NULL;
    }

    while (fgets(buf, sizeof(buf), fp) != NULL) {
        printf("func is %s,%d,%s\n", __func__,__LINE__,buf);
    }
    printf("func is %s,%d, %s\n",__func__,__LINE__,"*******************");
    fseek(fp, 0, SEEK_END);
    long int model_len = ftell(fp);
    printf("func is %s,model_len is %ld, %s\n",__func__,__LINE__,model_len);
    unsigned char *model = (unsigned char *)malloc(model_len);
    fseek(fp, 0, SEEK_SET);
    printf("func is %s,%d, %s\n",__func__,__LINE__,"*******************");
    if (model_len != fread(model, 1, 20, fp))
    {
        printf("fread %s fail!\n", __func__);
        free(model);
        return NULL;
    }
    printf("func is %s,%d, %s\n",__func__,__LINE__,"*******************");
    *model_size = model_len;
    if (fp)
    {
        fclose(fp);
    }
    return model;
}



int RKNN_ImgFusionInit(char *ciphertext)
{
  
    char buf[100]={0};   
    int model_len = 0;
    std::string cmd;
    std::string rknn_decode_str;
    char *decryptStr=NULL;
    FILE *rknn_file = NULL;
    decryptStr=RSADecryptString(ciphertext);
    
    getSerialNum(buf);
    if(strcmp(decryptStr,(&buf[10]))==0) {
        printf("func is %s,%d,%s\n",__func__,__LINE__,"Verification is successful (Sequence number matches successfully)");
        free(decryptStr);
    }
    else
    return -2;

    // base64 decode
    rknn_decode_str=base64_decode(base64_chars);
    model_len = rknn_decode_str.size();

    // Load RKNN Model
    unsigned char *model = (unsigned char *)malloc(model_len);
    model=(uchar *)(rknn_decode_str.c_str());
   // model = load_model(rknn_file, &model_len);
    ret = rknn_init(&ctx, model, model_len, 0);
    if (ret < 0)
    {
       printf("rknn_init fail! ret=%d\n", ret);
       return -1;
    }

    // Get Model Input Output Info
    //rknn_input_output_num io_num;
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
    printf("222func is %s,%d,io_num.n_input is %d\n",__func__,__LINE__,io_num.n_input);
    assert(DEMO_INPUT_NUM == io_num.n_input);
    // Set Input struct
    //rknn_input inputs[DEMO_INPUT_NUM];
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

    
	return 0;
}

/*
pVisibleRgbData:		输入可见光图像数据,格式RGB
pInfrareRgbData:		输入红外图像数据,格式RGB
pFusionRgbData:			输出融合图像的数据,格式RGB
uiWidth:				图像宽度
uiHeight:				图像高度
*/
int RKNN_ImgFusionProcess(void *pVisibleRgbData, void *pInfrareRgbData,void **pFusionRgbData, unsigned int uiWidth, unsigned int uiHeight)
{
	//todo 这里实现融合算法
    inputs[0].buf = pVisibleRgbData;//pVisibleRgbData;
    inputs[1].buf = pInfrareRgbData;//pInfrareRgbData;

    struct timeval start_time, stop_time;
    gettimeofday(&start_time, NULL);
    rknn_inputs_set(ctx, io_num.n_input, inputs);
    ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
    if (ret < 0)
    {
        printf("rknn_input_set fail! ret=%d\n", ret);
        return -1;
    }

    // Run
    ret = rknn_run(ctx, nullptr);
    if (ret < 0)
    {
        printf("rknn_run fail! ret=%d\n", ret);
        return -1;
    }

    // Get Output
   // rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1;
    ret = rknn_outputs_get(ctx, 1, outputs, NULL);
    if (ret < 0)
    {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        return -1;
    }

    struct timeval start_time1;

    *pFusionRgbData = outputs[0].buf;
    gettimeofday(&start_time1, NULL);
    printf(" run total = %f ms\n", 
           (__get_us(start_time1) - __get_us(start_time)) / 1000.0);


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


