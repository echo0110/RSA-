#SDK_TOP_DIR=/Work/RV1126_RV1109_LINUX_SDK_V2.1.0_20210512/
#CROSS= $(SDK_TOP_DIR)/prebuilts/gcc/linux-x86/arm/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf/bin/arm-linux-gnueabihf-


SDK_TOP_DIR=/home/haitu/newcode/Cx_rv1126_v1.8.0/

CROSS=/home/haitu/newcode/Cx_rv1126_v1.8.0/prebuilts/gcc/linux-x86/arm/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf/bin/arm-linux-gnueabihf-

RKNN_API_DIR=$(SDK_TOP_DIR)external/rknpu/rknn/rknn_api
RKNN_DEMO_DIR=$(RKNN_API_DIR)/examples
JPEG_DIR=/home/haitu/newcode/Cx_rv1126_v1.8.0/external/minigui/src/image/
#JPEG_DIR=/home/haitu/RV1126_RV1109_LINUX_SDK_V2.1.0_20210512/buildroot/output/rockchip_rv11xx_rtcore_38ipc/host/arm-buildroot-linux-gnueabihf/sysroot/usr/include
HIG_DIR=/home/haitu/newcode/Cx_rv1126_v1.8.0/external/rkmedia/examples/3rdparty/opencv/opencv-linux-armhf/include/opencv/
CFLAGS=-I include \
       -I $(RKNN_DEMO_DIR)/3rdparty/stb \
       -I $(RKNN_DEMO_DIR)/3rdparty/ \
       -I $(RKNN_DEMO_DIR)/3rdparty/rga/include \
       -I $(RKNN_DEMO_DIR)/3rdparty/drm/include \
       -I $(RKNN_DEMO_DIR)/3rdparty/drm/include/libdrm \
       -I $(RKNN_DEMO_DIR)/3rdparty/opencv/opencv-linux-armhf/include \
       -I $(RKNN_API_DIR)/librknn_api/include \
       -I $(JPEG_DIR) \
       -I $(HIG_DIR) \

LDFLAGS=-ldl $(RKNN_API_DIR)/librknn_api/lib/librknn_api.so

LDFLAGS += -L $(RKNN_DEMO_DIR)/3rdparty/opencv/opencv-linux-armhf/lib/ \
             -lopencv_imgcodecs \
             -lopencv_imgproc \
             -lopencv_core \
             -pthread \
             $(RKNN_DEMO_DIR)/3rdparty/opencv/opencv-linux-armhf/share/OpenCV/3rdparty/lib/libtegra_hal.a \
             $(RKNN_DEMO_DIR)/3rdparty/opencv/opencv-linux-armhf/share/OpenCV/3rdparty/lib/libquirc.a \
             $(RKNN_DEMO_DIR)/3rdparty/opencv/opencv-linux-armhf/share/OpenCV/3rdparty/lib/liblibwebp.a \
             $(RKNN_DEMO_DIR)/3rdparty/opencv/opencv-linux-armhf/share/OpenCV/3rdparty/lib/liblibpng.a \
             $(RKNN_DEMO_DIR)/3rdparty/opencv/opencv-linux-armhf/share/OpenCV/3rdparty/lib/liblibjasper.a \
             $(RKNN_DEMO_DIR)/3rdparty/opencv/opencv-linux-armhf/share/OpenCV/3rdparty/lib/liblibtiff.a \
             $(RKNN_DEMO_DIR)/3rdparty/opencv/opencv-linux-armhf/share/OpenCV/3rdparty/lib/liblibjpeg-turbo.a \
             $(RKNN_DEMO_DIR)/3rdparty/opencv/opencv-linux-armhf/share/OpenCV/3rdparty/lib/liblibprotobuf.a \
             $(RKNN_DEMO_DIR)/3rdparty/opencv/opencv-linux-armhf/share/OpenCV/3rdparty/lib/libIlmImf.a \
             $(RKNN_DEMO_DIR)/3rdparty/opencv/opencv-linux-armhf/share/OpenCV/3rdparty/lib/libzlib.a


SRCS= main.cpp RkNNImageFusion.cpp new.cpp RkNNImageFusion.h

all:
	$(CROSS)g++ $(SRCS) $(CFLAGS) $(LDFLAGS) -o test
clean:
	-rm -f *.out *.o

install:
	mv test.out  /Work/sambaboot/NPU/ -f && sync && sync
