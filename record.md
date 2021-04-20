### 硬件信息

![image-20210420160656291](C:\Users\10125\AppData\Roaming\Typora\typora-user-images\image-20210420160656291.png)

| 名称                           | 具体信息                                                     |
| ------------------------------ | ------------------------------------------------------------ |
| 英特尔® 至强® 金牌 6240 处理器 | 18个核心（每个核心两个线程），但是这里只有9个核心。TDP官网上写的是150W， 如果只有9个核心的话，姑且算个75W吧 |
| RTX 2080Ti                     | TDP 250W                                                     |
| ascend 910                     | TDP 310W                                                     |

### 实验设定

#### 1.微bench

测评跑矩阵乘法和卷积算子（三种不同大小）时不同硬件设备的GOP/S和GOP/J

##### 产出

a.矩阵算子对比图，卷积算子对比图

b.NPU分析图



#### 2.宏bench

测评模型（三种batch-size）

产出

a.CPU VS GPU(所有模型)  CPU VS GPU VS NPU(部分模型) （GOP/S 和 GOP/J）

b.CPU VS GPU VS NPU(部分模型) （Sample/s 和 Sample/j)

c.NPU分析

