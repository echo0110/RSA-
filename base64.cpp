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

#if 1
static const std::string base64_chars3 = 
             "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
             "abcdefghijklmnopqrstuvwxyz"
             "0123456789+/";




static inline bool is_base64(unsigned char c) {
  return (isalnum(c) || (c == '+') || (c == '/'));
}

std::string base64_decode(std::string const& encoded_string) {
  int in_len = encoded_string.size();
  int i = 0;
  int j = 0;
  int in_ = 0;
  unsigned char char_array_4[4], char_array_3[3];
  std::string ret;
  printf("func is %s,%d,in_len is  %d\n",__func__,__LINE__,in_len);
  printf("func is %s,%d,encoded_string[in_] is  %c\n",__func__,__LINE__,encoded_string[in_]);
  while (in_len-- && ( encoded_string[in_] != '=') && is_base64(encoded_string[in_])) {
    char_array_4[i++] = encoded_string[in_]; in_++;
    printf("func is %s,%d,in_ is  %d\n",__func__,__LINE__,in_);
    printf("func is %s,%d,i is  %d\n",__func__,__LINE__,i);
    if (i ==4) {
      for (i = 0; i <4; i++)
        char_array_4[i] = base64_chars3.find(char_array_4[i]);
 
      char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
      char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
      char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];
 
      for (i = 0; (i < 3); i++)
        ret += char_array_3[i];
      i = 0;
    }
  }
 
  if (i) {
    for (j = i; j <4; j++)
      char_array_4[j] = 0;
 
    for (j = 0; j <4; j++)
      char_array_4[j] = base64_chars3.find(char_array_4[j]);
 
    char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
    char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
    char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];
 
    for (j = 0; (j < i - 1); j++) ret += char_array_3[j];
  }
 
  return ret;
}
#endif