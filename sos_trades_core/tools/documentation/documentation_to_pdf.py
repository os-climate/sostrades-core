import re
import os
import subprocess
from os import listdir
from os.path import abspath, basename, dirname, isdir, isfile, join, splitext
import base64
import pathlib as path
import markdown2
import markdown
import mdtex2html
import sys
import pdflatex

from xhtml2pdf import pisa

"""
conversion in client side:
- makepdf: works but need a lib to transform the markdown in html
- jspdf: no css conversion
- html2canvas convert in an image (no copy possible)
- html-to-pdfmake (https://www.npmjs.com/package/html-to-pdfmake): ok but for simple rendering, no formula


conversion markdown to pdf
- pandoc : ok, needs some adjustment on markdown (aligned on formula for multilines, break lines before * or images)


conversion markdown to html:
- mdtex2html: show very simple formula but no fraction or others
- markdown2: ok but no formula conversion
- markdown: bad rendering of lists, sources... no formulas


conversion html to PDF:
- xhtml2pdf: ok accepted for now
- https://github.com/ljpengelen/markdown-to-pdf :lib error, neeed an env var
- weasyprint: lib error, did not manage to locate a library called 'gobject-2.0-0'
- pdfkit: have to install exe for wkhtmltopdf, pb with our protected env
- Pyppeteer: not tried
- https://pypi.org/project/pdflatex/ 


https://www.smashingmagazine.com/2019/06/create-pdf-web-application/

"""
DEFAULT_EXTRAS = [
    'tables'
]

def convert_to_html(md_file_path,fileName):
    output = """<!DOCTYPE html>
        <html lang="en">
        <head>
            <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
            <style type="text/css">
            table, th, td { border: 1px solid;}
            </style>
            <script type="text/javascript" async
              src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML">
            </script>
        </head>
        <body>
        """
    os.chdir(md_file_path)
    mkin = open(fileName, encoding="utf8")
    output += markdown2.markdown(mkin.read(), extras=DEFAULT_EXTRAS)
    #output += markdown.markdown(mkin.read(), extensions=DEFAULT_EXTRAS)
    #output += mdtex2html.convert(mkin.read())
    output += """</body>
        </html>
        """
    htmlfile = os.path.splitext(fileName)[0] + ".html"

    outfile = open(htmlfile, "w")
    outfile.write(output)
    outfile.close()

    # create pdf
    #df = weasyprint.HTML(htmlfile).write_pdf()
    #pdfkit.from_file(htmlfile, os.path.splitext(htmlfile)[0] + ".pdf", options=PDF_OPTIONS)
    pdf_file_name = os.path.splitext(fileName)[0] + ".pdf"
    result_file = open(pdf_file_name, "w+b")

    # convert HTML to PDF
    pisa_status = pisa.CreatePDF(
            output.encode('utf-8'),                # the HTML to convert
            dest=result_file,
            encoding='utf-8')           # file handle to recieve result

    # close output file
    result_file.close()                 # close output file

    #subprocess.run(['pandoc', f'{fileName} --pdf-engine=C:\\Users\\NG91784\\AppData\\Local\\Programs\\MiKTeX\\miktex\\bin\\x64\\pdflatex -o {pdf_file_name}'])

def convert_md_to_pdf(md_file_path,fileName):
    os.chdir(md_file_path)
    tex_filename = os.path.splitext(fileName)[0] + ".pdf"
    os.system(f'pandoc {fileName} --pdf-engine=xelatex  -o {tex_filename} -V geometry:"top=2cm, bottom=1.5cm, left=2cm, right=2cm"')



if '__main__' == __name__:
    print("conversion md in pdf")
    documentation_apds = "C:/Users/NG91784/Documents/Sostrades/business_case/business_case/v3/sos_processes/business_case/apds_manual_simple/documentation"
    name_apds = "process.markdown"

    documentation_agri = "C:/Users/NG91784/Documents/Sostrades/witness-core/climateeconomics/sos_wrapping/sos_wrapping_agriculture/agriculture/documentation"
    name_agri = "agriculture_disc.md"
    convert_md_to_pdf(documentation_agri,name_agri )

