<div style="text-align:center">
<img src="https://big-cheng.com/img/sciparser.png" alt="sciparser-logo" width="200"/>
<h2>PDF parsing toolkit for preparing text corpus</h2>
</div>
<a href='https://sciparser.acemap.info/'><img src='https://img.shields.io/badge/Sciparser-Demo-C71585'></a> <a href='https://github.com/davendw49/sciparser'><img src='https://img.shields.io/badge/Github-Code-4169E1'></img></a>

# Introduction

This repo contain a PDF parsing toolkit for preparing text corpus using to transfer PDF to Markdown. Based on [PDF Parser ToolKits](https://github.com/Acemap/pdf_parser), gathering most-use PDF OCR tools for academic papers, and inspired by `grobid_tei_xml`, an open-sourced PyPI package, we develop sciparser 1.0 for text corpus pre-processing, in recent works like [K2](https://github.com/davendw49/k2) and [GeoGalactica](https://github.com/davendw49/geogalactica), we use this tool and upgrade grobid backend solution to pre-process the text corpus. And the online demo is publicly available.

- Try [DEMO](https://sciparser.acemap.info/)

In this repo and demo, we only share the secondary processing solution on Grobid. In the near future, we will share the multiple-backend combinition solution on PDF parsing.

# Requirements

```bash
git clone https://github.com/Acemap/pdf_parser.git
cd pdf_parser
pip install -r requirements.txt
python setup install

git clone https://github.com/davendw49/sciparser.git
cd sciparser
pip install -r requirements.txt
```

# Usage

- **python**

First we should clone the hold repo.
```bash
git clone https://github.com/davendw49/sciparser.git
```

Then `import` the `pipeline` file to do the parsing.
```python
from pipeline import pipeline
data = pipeline('/path/to/your/pdf/')
```

- **gradio**

```bash
python main.py
```

# Citation

```
@misc{sciparser,
  author = {Cheng Deng},
  title = {Sciparser: PDF parsing toolkit for preparing text corpus},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/davendw49/sciparser}},
}
```

# Reference

- PDF Parser ToolKits: https://github.com/Acemap/pdf_parser
- TEI-XML Parser (grobid_tei_xml): https://gitlab.com/internetarchive/grobid_tei_xml