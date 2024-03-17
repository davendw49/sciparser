import fitz
import re
import json
import cv2
import xml.etree.ElementTree as ET # 解析Grobid解析出来xml文件的库
from paddleocr import PPStructure # 解析表格的库
import htmltabletomd # 将解析出来html格式的表格转为markdown格式的库
from pix2text import Pix2Text, merge_line_texts # 解析公式的库
from grobid_client_python.grobid_client.grobid_client import GrobidClient #安装grobid_client_python到工作目录下，还要同时配置Grobid
from pdf_parser import Parser # 实验室接口

#获取文档名
parser = argparse.ArgumentParser()
parser.add_argument("file_name")
args = parser.parse_args()
file_name = args.file_name.strip('.pdf')
input_pdf = f"./inputs/{file_name}.pdf"

# 调用Grobid，解析pdf
client = GrobidClient(config_path='./grobid_client_python/config.json')# 这个需要根据下载的grobid_client_python的路径改一下
client.process("processFulltextDocument", input_path="./inputs/", output='./outputs/', n=20, include_raw_citations=True, include_raw_affiliations=True, tei_coordinates=True, force=True, verbose=True)

# 调用Pdffigures2，解析图表
parser = Parser('pdffigures2')
parser.parse('figure', './inputs/', './outputs/')

tree = ET.parse(f'./outputs/{file_name}.tei.xml')
root = tree.getroot()
doc = fitz.open(f'./inputs/{file_name}.pdf')

p = re.compile(r'(\D*)(\d+)(\D*)')# 解析label的正则化表达式
p2t = Pix2Text(
    analyzer_config=dict(  # 声明 LayoutAnalyzer 的初始化参数
        model_name='mfd',
        model_type='yolov7_tiny',  # 表示使用的是 YoloV7 模型，而不是 YoloV7_Tiny 模型
        model_fp='./config/yolov7_tiny_mfd-pytorch/analysis/mfd-yolov7_tiny.pt',  # 注：修改成你的模型文件所存储的路径
    ),
		formula_config = {'model_fp': './config/p2t-mfr-model-20230702/p2t-mfr-20230702.pth'},  # 注：修改成你的模型文件所存储的路径
)# 解析公式的模型config

table_engine = PPStructure(layout=False, show_log=True)# 表格解析模型初始化

with open(f'./outputs/{file_name}.pdffigures2.figure/figure-data.json','r',encoding='utf8')as f:
    figure_data = json.load(f)
fp = open(f'./outputs/{file_name}.md','w')
title = root[0][0][0][0].text
fp.write('# ' + '<center>' + title + '</center>' + '\r\n')
abstract = root[0][2].findall('{http://www.tei-c.org/ns/1.0}abstract')[0][0][0].text
fp.write('## ' + 'Abstract' + '\r\n')
fp.write(abstract + '\r\n')
body = root[2][0]
for element in body:
    #段落结构
    if element.tag.endswith('div'):
        for ind in range(len(element)):
            content = element[ind]
            if content.tag.endswith('head'):
                head = content
                #head中没有段落级别
                if 'n' not in head.attrib:
                    fp.write('<b>' + head.text + '</b>' + '\r\n')
                elif head.attrib['n'].count('.') == 1:
                    fp.write('## ' + head.attrib['n'] + head.text + '\r\n')
                elif head.attrib['n'].count('.') == 2:
                    fp.write('### ' + head.attrib['n'] + head.text + '\r\n')
                elif head.attrib['n'].count('.') == 3:
                    fp.write('#### ' + head.attrib['n'] + head.text + '\r\n')
                else:
                    fp.write('##### ' + head.attrib['n'] + head.text + '\r\n')   
            elif content.tag.endswith('p'):
                para = content
                # 无引用
                if len(para) == 0:
                    fp.write(para.text + '\r\n')
                else:
                    fp.write(para.text)
                    for ref in para:
                        # 如果是reference引用
                        if ref.attrib['type'] == "bibr":
                            att = ref.attrib
                            # 有链接索引的引用
                            if 'target' in att:
                                # 单引用
                                if ref.text.startswith('[') and ref.text.endswith(']'):
                                    fp.write('[' + ref.text  + '(' + att['target'] + ')' + ']') 
                                    if ref.tail is not None: 
                                        fp.write(ref.tail)
                                # 多个引用    
                                else:
                                    # 为了避免少信息
                                    ref_before_num = p.findall(ref.text)[0][0]
                                    ref_num = p.findall(ref.text)[0][1]
                                    ref_after_num = p.findall(ref.text)[0][2]
                                    if ref_before_num != '':
                                        fp.write(ref_before_num)
                                    fp.write('[' + '[' + ref_num + ']' + '(' + att['target'] + ')' + ']') 
                                    if ref_after_num != '':
                                        fp.write(ref_after_num)
                                    if ref.tail is not None:
                                        fp.write(ref.tail)
                            else:
                                fp.write(ref.text) 
                                if ref.tail is not None:
                                    fp.write(ref.tail)                              
                        # 如果是图片或表格引用
                        elif ref.attrib['type'] == "figure" or ref.attrib['type'] == "table":
                            att = ref.attrib
                            # 有链接索引的引用
                            if 'target' in att:
                                ref_before_num = p.findall(ref.text)[0][0]
                                ref_num = p.findall(ref.text)[0][1]
                                ref_after_num = p.findall(ref.text)[0][2]
                                if ref_before_num != '':
                                    fp.write(ref_before_num)
                                fp.write( '[' + ref_num + ']' + '(' + att['target'] + ')')
                                if ref_after_num != '':
                                    fp.write(ref_after_num)  
                            else:
                                fp.write(ref.text)
                            if ref.tail is not None:
                                fp.write(ref.tail)
                        #其他类型的引用
                        else:
                            fp.write(ref.text)
                            if ref.tail is not None:
                                fp.write(ref.tail)                  
                    fp.write('\r\n')
            elif content.tag.endswith('formula'):
                formula = content
                if len(formula.text) > 5:
                    id = formula.attrib['{http://www.w3.org/XML/1998/namespace}id']
                    if len(formula)==0:#没检测出来label
                        label = ''
                    else:
                        label = '' if len(p.findall(formula[0].text)) == 0 else p.findall(formula[0].text)[0][1]
                    coords = formula.attrib['coords']
                    page_num = int(coords.split(',')[0])
                    page = doc[page_num-1]
                    tl_x = float(coords.split(',')[1])
                    tl_y = float(coords.split(',')[2])
                    rb_x = tl_x + float(coords.split(',')[3])
                    rb_y = tl_y + float(coords.split(',')[4])
                    clip = fitz.Rect(tl_x-1, tl_y-1, rb_x - 15, rb_y+1)
                    # 增加分辨率
                    zoom_x = 2  # horizontal zoom
                    zoom_y = 2  # vertical zoom
                    mat = fitz.Matrix(zoom_x, zoom_y)  # zoom factor 2 in each dimension
                    pix = page.get_pixmap(matrix=mat,clip=clip)
                    pix.save(f"./outputs/{file_name}.pdffigures2.figure/Formula-{id.split('_')[-1]}.png")      
                    img_path = f"./outputs/{file_name}.pdffigures2.figure/Formula-{id.split('_')[-1]}.png"
                    outs = p2t(img_path, resized_shape=608)  # 也可以使用 `p2t.recognize(img_fp)` 获得相同的结果
                    # 如果只需要识别出的文字和Latex表示，可以使用下面行的代码合并所有结果
                    only_text = merge_line_texts(outs, auto_line_break=True)
                    only_text = only_text.strip()
                    if label != '':
                        temp_list = list(only_text)
                        if only_text.endswith('$$'):#如果最后是$$，从倒数第二个插入
                            temp_list.insert(-2,f'\n\\tag{{{label}}}\n')
                            temp_list.insert(0,'\n')
                        elif only_text.endswith('$'):#如果最后是$，从倒数第一个插入
                            temp_list.insert(-1,f'\n\\tag{{{label}}}\n$')#变成$$
                            if only_text.startswith('$'):
                                temp_list.insert(0,'\n$')#变成$$
                            else:
                                temp_list.insert(0,'\n')
                        only_text = ''.join(temp_list)
                        fp.write(only_text)
                    else:
                        fp.write(only_text)
                    fp.write('\r\n')
        fp.write('\r\n')
    if element.tag.endswith('figure'):
        id = element.attrib['{http://www.w3.org/XML/1998/namespace}id']
        if id.startswith('fig'):
            figType = 'Figure'
            name = element[1].text
        elif id.startswith('tab'):
            figType = 'Table'
            name = element[1].text  
        if name is None:
            continue
        else:
            for figure in figure_data:
                if  figType == figure['figType'] and name == figure['name']:
                    renderURL = figure['renderURL']
                    caption = figure['caption']
                    # 是图
                    if figType == 'Figure': 
                        fp.write(f'![](./outputs/{file_name}.pdffigures2.figure/{renderURL})'+'\r\n')
                        fp.write(f'<a name="{id}">' + '</a>' + '\r\n')
                        fp.write('<center>' + caption + '</center>' + '\r\n')
                    elif figType == 'Table':
                        img_path = f'./outputs/{file_name}.pdffigures2.figure/{renderURL}'
                        img = cv2.imread(img_path)
                        result = table_engine(img)
                        html = result[0]['res']['html']
                        md_table = htmltabletomd.convert_table(html)
                        fp.write('\r\n')
                        fp.write(md_table)
                        fp.write('\r\n')
                        fp.write(f'<a name="{id}">' + '</a>' + '\r\n')
                        fp.write('<center>' + caption + '</center>' + '\r\n')   
        fp.write('\r\n')                

back = root[2][1]
fp.write('## ' + 'Reference' + '\r\n')
for div in back:
    if div.attrib['type'] == "references":
        listBibl = div[0]
        for ind in range(len(listBibl)):
            bib = listBibl[ind]
            id = bib.attrib['{http://www.w3.org/XML/1998/namespace}id']
            for element in bib:
                if element.tag.endswith('note') and 'type' in element.attrib and element.attrib['type']=="raw_reference":
                    reference = element.text
            fp.write(f'<a name="{id}">' + '</a>' + '\r\n')
            fp.write('[' + str(ind+1) + '] ' + reference + '\r\n')
            fp.write('\r\n')
            
fp.close()


