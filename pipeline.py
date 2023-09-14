"""
Grobid XML to Markdown for cleaner LLM corpus
@Author: Cheng Deng
"""
import os
import time

from grobid_parser import parse
from pdf_parser import Parser

def pipeline(file_path):
    parser = Parser('grobid', host='10.10.10.10', port='8074')
    parser.parse('text', file_path, '.tmp', 50)
    print("FINISH PARSING")
    result = []
    xml_result = []
    with open(f'''./tmp/{file_path.split('/')[-1].replace('.pdf', '')}.grobid.xml''') as f:
        xml_text = f.read()
        xml_result.append(xml_text)
        res = parse.parse_document_xml(xml_text)
        result = [res.header, res.abstract, res.body]
    
    print(res.header)
    title = result[0].title
    abstract = result[1]
    
    print(title, abstract)
    if len(title.strip()) != '':
        title_text = f"\n\n# {title}\n\n"
    else:
        title_text = ''
    
    if len(abstract.strip()) != '':
        abstract_text = f"## Abstract\n\n{abstract}\n\n"
    else:
        abstract_text = ''
    
    final_text = f"{title_text}{abstract_text}{result[2]}"
    xml_res = xml_result[0]
    print("FINISH REPARSING")
    return xml_res, final_text
        

if __name__ == "__main__":
    data = pipeline('1611.01144.pdf')
    print(data)