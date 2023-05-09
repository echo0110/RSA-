1、crytop生产 公钥和私钥
2、获取瑞芯微硬件板卡 serial id
#cat /proc/cpuinfo | grep serial
3、拿到serial id后 交给cryptopp进行 RSA加密

# RSA-encryption
瑞芯微平台，Linux系统，基于cryptopp对 算法模型 进行 RSA加密 
下图是简单的对算法模型的加密流程
![image](https://user-images.githubusercontent.com/24950840/236977848-4a1a678c-dfc9-49dd-a0a6-2aed3a300e8b.png)

