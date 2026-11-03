# -*- coding: utf-8 -*-
"""Convert the verified Round-7 Markdown files to clean, editable DOCX."""
from pathlib import Path
import re
import shutil
import docx
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH

ROOT=Path(__file__).resolve().parents[1]
PKG=ROOT/"投稿包_BMC"

def convert(md_path,docx_path):
    backup=docx_path.with_name(docx_path.stem+"_pre_round7"+docx_path.suffix)
    if not backup.exists(): shutil.copyfile(docx_path,backup)
    lines=md_path.read_text(encoding="utf-8").splitlines()
    doc=docx.Document()
    normal=doc.styles["Normal"]
    normal.font.name="Times New Roman";normal.font.size=Pt(10.5)
    def paragraph(text,level=0):
        text=text.replace("**","").replace("`","")
        p=doc.add_paragraph()
        r=p.add_run(text)
        r.font.name="Times New Roman"
        r.font.size=Pt({0:10.5,1:15,2:13,3:11.5,4:10.5}[level])
        r.bold=level>0
        p.paragraph_format.space_after=Pt(6 if level==0 else 4)
        return p
    def table(rows):
        parsed=[[c.strip() for c in row.strip().strip("|").split("|")] for row in rows]
        parsed=[r for r in parsed if not all(set(c)<=set("-: ") for c in r)]
        if not parsed: return
        t=doc.add_table(rows=len(parsed),cols=len(parsed[0]))
        t.style="Table Grid"
        for i,row in enumerate(parsed):
            for j,value in enumerate(row):
                cell=t.cell(i,j);cell.text=""
                r=cell.paragraphs[0].add_run(value.replace("**","").replace("`",""))
                r.font.name="Times New Roman";r.font.size=Pt(7 if len(parsed[0])>=7 else 8.5);r.bold=i==0
        doc.add_paragraph()
    i=0
    while i<len(lines):
        line=lines[i].strip()
        if not line:
            i+=1;continue
        if line.startswith("|"):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i].strip());i+=1
            table(rows);continue
        m=re.match(r"^!\[(.*?)\]\((.*?)\)$",line)
        if m:
            image=PKG/m.group(2)
            p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
            p.add_run().add_picture(str(image),width=Cm(16))
            i+=1;continue
        if line.startswith("#### "): paragraph(line[5:],4)
        elif line.startswith("### "): paragraph(line[4:],3)
        elif line.startswith("## "): paragraph(line[3:],2)
        elif line.startswith("# "): paragraph(line[2:],1)
        else: paragraph(line)
        i+=1
    doc.save(docx_path)
    print(docx_path.name,"paragraphs",len(doc.paragraphs),"tables",len(doc.tables))

convert(PKG/"Manuscript_SICM_BMC.md",PKG/"Manuscript_SICM_BMC.docx")
convert(PKG/"Tables_BMC.md",PKG/"Tables_BMC.docx")
convert(PKG/"Supplement_SICM_BMC.md",PKG/"Supplement_SICM_BMC.docx")
convert(PKG/"CoverLetter_BMC.md",PKG/"CoverLetter_BMC.docx")
convert(PKG/"TRIPOD_AI_checklist_27.md",PKG/"TRIPOD_AI_checklist_27.docx")
