import gradio as gr
import base64
from PyPDF2 import PdfFileReader
from pipeline import pipeline
import markdown

def view_pdf(pdf_file):
    with open(pdf_file.name,'rb') as f:
        pdf_data = f.read()
    # pdf_data = pdf_file
    b64_data = base64.b64encode(pdf_data).decode()
    # print(b64_data)
    return f"<embed src='data:application/pdf;base64,{b64_data}' type='application/pdf' width='100%' height='700px' />"

def extract_text(pdf_file):
    xml, md = pipeline(pdf_file.name)
    # print(text_data)
    res = markdown.markdown(md, extensions=['tables']).replace("<s>", "")
    res_rich_md = f'<div style="max-height: 775px; overflow-y: auto;">{res}</div>'
    res_xml = f'{xml}'
    res_md = f'{md}'
    return res_xml, res_md, res_rich_md
    

with gr.Blocks() as demo:
    gr.Markdown(
        '''<p align="center" width="100%">
        <img src="https://big-cheng.com/img/pdf.png" alt="pdf-logo" width="50"/>
        <p>
        
        <h1 align="center">🧚🏻‍♀️ Preparing Text Corpus For Training Academic Language Model</h1>
        '''
    )
    with gr.Row():
        with gr.Column():
            gr.Markdown('## Upload PDF')
            file_input = gr.File(type="file")
            with gr.Row():
                with gr.Column():
                    viewer_button = gr.Button("View PDF")
                with gr.Column():
                    parser_button = gr.Button("Parse PDF")
            file_out = gr.HTML()
        with gr.Column():
            gr.Markdown('## Parsing file')
            with gr.Tab("XML Result"):
                xml_out = gr.Textbox(
                    lines=36,
                )
            with gr.Tab("Markdown Result"):
                md_out = gr.Textbox(
                    lines=36,
                )
            with gr.Tab("Rich Markdown Result"):
                rich_md_out = gr.HTML()
            
    viewer_button.click(view_pdf, inputs=file_input, outputs=file_out)
    parser_button.click(extract_text, inputs=file_input, outputs=[xml_out, md_out, rich_md_out])

demo.launch(server_name="0.0.0.0", debug=True)
