### 1 安装Grobid

> 详细内容也可以参考[这里]([Install GROBID - GROBID Documentation](https://grobid.readthedocs.io/en/latest/Install-Grobid/))。

Grobid的配置是解析工作的基础，它是使用Java编写的，在解析时需要在终端开启Grobid服务。

### 2 安装Python GROBID client

> 详细内容也可以参考[这里]([kermitt2/grobid_client_python: Python client for GROBID Web services (github.com)](https://github.com/kermitt2/grobid_client_python))。

下载该文件夹到工作目录下，该库的作用是可以使用python命令调用Grobid服务。

### 3 安装Paddleocr

> 详细内容也可以参考[这里](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/ppstructure/docs/quickstart.md)。

该库可以实现表格的ocr识别

### 4 安装Htmltabletomd

> 详细内容也可以参考[这里]([htmltabletomd · PyPI](https://pypi.org/project/htmltabletomd/))。

`pip install htmltabletomd`即可。

该包用于将解析出来html格式的表格转为markdown格式

### 5 安装Pix2text

详细内容也可以参考[这里](https://github.com/breezedeus/pix2text)。

`pip install pix2text`即可。

该库用于解析公式图片

### 6 安装Pdf_parser

详细内容也可以参考[这里]([Acemap-Tools / pdf_parser · GitLab](https://git.acemap.cn/acemap-tools/pdf_parser))。

`pip install -U git+https://git.acemap.cn/acemap-tools/pdf_parser.git`即可。

使用该工具得挂实验室的VPN。主要用其pdffigures2的功能。

### 7 使用示例

```
cd ./grobid-0.7.2
./gradlew run
# 再开个终端
cd ./pdf_parser_v2 
python parser.py Elsvier1.pdf
```

 <center class="half">
    <img src=".\图片\Snipaste_2023-09-03_22-56-20.jpg" width="400"/>
    <img src=".\图片\Snipaste_2023-09-03_22-57-00.jpg" width="200"/>
</center>

<center>图1. 效果示意（文本）</center>

 <center class="half">
    <img src=".\图片\Snipaste_2023-09-03_23-11-56.jpg" width="300"/>
    <img src=".\图片\Snipaste_2023-09-03_23-14-46.jpg" width="400"/>
</center>

<center>图2. 效果示意（图片）</center>

 <center class="half">
    <img src=".\图片\Snipaste_2023-09-03_23-17-04.jpg" width="300"/>
    <img src=".\图片\Snipaste_2023-09-03_23-17-18.jpg" width="300"/>
</center>

<center>图3. 效果示意（表格）</center>

 <center class="half">
    <img src=".\图片\Snipaste_2023-09-03_23-20-51.jpg" width="300"/>
    <img src=".\图片\微信图片_20230902232519.png" width="400"/>
</center>

<center>图4. 效果示意（公式）</center>

说明：

​	目前的解析pipeline是针对于Elsevier的一种版式进行的，该版式如给出的示例所示，示例pdf在./inputs/下。首先将pdf送入Grobid进行第一步的解析，得到pdf文档的各种信息，文本，大纲，图表，公式，引用，链接等等。由于Grobid的图标解析在此时表现不佳，使用Pdffigures2解析图表。清洗Grobid解析得到的xml文件，将上述信息添加到markdown文件中，表格使用Paddleocr解析，公式使用Pix2Text解析。

注意：

（1）python3.6+

（2）关闭科学上网工具

（3）目前关注解析效果，解析速度方面后续还会优化

（4）仅支持单pdf解析

（5）段内公式解析效果差







