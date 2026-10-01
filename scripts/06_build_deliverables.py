"""Build six-page LaTeX report and explanatory, executable notebook."""
from pathlib import Path
import hashlib
import json
import re
import sys
import nbformat as nbf
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from study_content import sections, ROOT, REPO
from src.figures import build as build_figures

def esc(x):
    mapping={'&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_',
             '{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
    return ''.join(mapping.get(c,c) for c in str(x))

def prose(s):
    saved=[]
    def keep(value):saved.append(value);return f'ZZTOKEN{len(saved)-1}ZZ'
    s=re.sub(r'\$\$(.*?)\$\$',lambda m:keep(r'\['+m[1]+r'\]'),s,flags=re.S)
    s=re.sub(r'\$([^$]+)\$',lambda m:keep('$'+m[1]+'$'),s)
    s=re.sub(r'\[([^]]+)\]\((https?://[^)]+)\)',lambda m:keep(r'\href{'+esc(m[2])+r'}{'+esc(m[1])+'}'),s)
    s=re.sub(r'\*\*(.*?)\*\*',lambda m:keep(r'\textbf{'+esc(m[1])+'}'),s)
    s=re.sub(r'`([^`]+)`',lambda m:keep(r'\texttt{'+esc(m[1])+'}'),s)
    s=esc(s)
    for i,value in enumerate(saved):s=s.replace(f'ZZTOKEN{i}ZZ',value)
    return s

def latex_table(item):
    n=len(item['headers'])
    # Widths refer to the usable text area after intercolumn padding.
    widths=item.get('widths') or [str(1/n)]*n
    layout=''.join(r'>{\raggedright\arraybackslash}p{'+str(float(w)) + r'\dimexpr\linewidth-'+str(2*n)+r'\tabcolsep\relax}' for w in widths)
    lines=[r'\par\smallskip\noindent{\fontsize{9.3}{11.5}\selectfont',r'\begin{tabular}{'+layout+'}',r'\toprule',
           ' & '.join(r'\textbf{'+esc(x)+'}' for x in item['headers'])+r' \\',r'\midrule']
    lines+=[' & '.join(esc(x) for x in row)+r' \\[3pt]' for row in item['rows']]
    lines += [r'\bottomrule\end{tabular}}\par\smallskip']
    return '\n'.join(lines)

PREAMBLE=r'''\documentclass[11pt,letterpaper]{article}
\usepackage[margin=0.70in,headheight=15pt,headsep=15pt,footskip=20pt]{geometry}
\usepackage{fontspec}
\IfFontExistsTF{Cambria}{\setmainfont{Cambria}}{\setmainfont{Latin Modern Roman}}
\IfFontExistsTF{Calibri}{\setsansfont{Calibri}}{\setsansfont{Latin Modern Sans}}
\usepackage{amsmath,booktabs,array,graphicx,xcolor,fancyhdr,titlesec,hyperref}
\definecolor{navy}{HTML}{17334D}
\definecolor{teal}{HTML}{168C8C}
\hypersetup{colorlinks=true,urlcolor=teal,linkcolor=navy,pdfauthor={Panagiotis Housos},pdftitle={AI Risk Sentiment and Equity Returns}}
\pagestyle{fancy}\fancyhf{}
\fancyhead[L]{\sffamily\small FRE-GY 7871A | Assignment 4}
\fancyhead[R]{\sffamily\small Panagiotis Housos | ph2606}
\fancyfoot[L]{\sffamily\footnotesize September 8--30, 2026 | Text ends at final market close}
\fancyfoot[R]{\sffamily\small\thepage\ / 6}
\renewcommand{\headrulewidth}{0.3pt}
\titleformat{\section}{\sffamily\large\bfseries\color{navy}}{}{0pt}{}
\titlespacing*{\section}{0pt}{0pt}{7pt}
\setlength{\parindent}{0pt}\setlength{\parskip}{5.5pt}
\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.05}
\emergencystretch=2em
\begin{document}
\fontsize{10.5}{12.5}\selectfont
'''

def build_report(content):
    tex=[PREAMBLE]
    md=['# AI Risk Sentiment and Equity Returns','Panagiotis Housos | ph2606 | FRE-GY 7871A | Fall 2026','']
    for index,(title,items) in enumerate(content):
        if index:tex.append(r'\newpage')
        if index==0:
            tex.append(r'{\sffamily\fontsize{19}{22}\selectfont\bfseries\color{navy} AI Risk Sentiment and Equity Returns}\par')
            tex.append(r'{\sffamily\small NLP and the Investment Process | Fall 2026}\par\smallskip')
        tex.append(r'\section*{'+esc(title)+'}')
        # References need a compact final page; main analytical pages remain 10pt.
        if index==5:tex.append(r'\fontsize{9.7}{11.8}\selectfont')
        md+=['## '+title,'']
        for item in items:
            if item['type']=='p':tex.append(prose(item['text'])+'\n'+r'\par');md +=[item['text'],'']
            elif item['type']=='table':
                tex.append(latex_table(item))
                md += ['| '+' | '.join(item['headers'])+' |','| '+' | '.join(['---']*len(item['headers']))+' |']
                md += ['| '+' | '.join(map(str,r))+' |' for r in item['rows']];md.append('')
            else:
                tex.append(r'\begin{center}\includegraphics[width='+item['width']+r'\linewidth]{outputs/'+item['name']+r'.pdf}\end{center}')
                tex.append(r'{\footnotesize\color{navy} '+prose(item['caption'])+r'\par}')
                md += ['!['+item['caption']+'](outputs/'+item['name']+'.png)','']
    tex.append(r'\end{document}')
    (ROOT/'report.tex').write_text('\n'.join(tex),encoding='utf-8')
    (ROOT/'REPORT.md').write_text('\n'.join(md),encoding='utf-8')

def build_notebook(content):
    previous=None
    if '--preserve-outputs' in sys.argv:
        previous=nbf.read(ROOT/'assignment4.ipynb',as_version=4)
    nb=nbf.v4.new_notebook()
    cells=[nbf.v4.new_markdown_cell('# Assignment 4 — AI risk sentiment and equity returns\n\n**Panagiotis Housos | ph2606 | FRE-GY 7871A | Fall 2026**\n\nSeptember 8–30, 2026, through the final 16:00 New York market close. Later text lies outside the return window. The six-page PDF and this notebook share the same computed exhibits. See README for acquisition and refresh; raw/derived datasets are intentionally excluded from Git.'),
    nbf.v4.new_code_cell("from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import display, Image\nROOT = Path.cwd()\nassert (ROOT / 'src/analysis.py').exists(), 'Run from the repository root.'\npd.set_option('display.max_columns', 14)\npd.set_option('display.max_colwidth', 65)\npd.set_option('display.precision', 4)\nfrom src.config import CUTOFF\nprint('Observed cutoff:', CUTOFF, '| Requested end: 2026-09-30')"),
    nbf.v4.new_markdown_cell('## Recalculate from the cached observations\n\nThe next cell recalculates the wording measures, tests and charts from locally cached source records. No observations are downloaded or simulated here. Synthetic tests are separate from study data. Language audit is a detector sensitivity and must be acquired before this cell to reproduce all specifications.'),
    nbf.v4.new_code_cell("from src.text_features import build as score_text\nfrom src.analysis import analyze\nfrom src.figures import build as draw_figures\nassert (ROOT / 'data/processed/language_audit.csv').exists(), 'Run python -m src.language_audit first.'\nscore_text()\npanel = analyze()\ndraw_figures()\nsummary = json.loads((ROOT / 'outputs/summary.json').read_text())\nassert panel.index.is_unique\nassert panel[['basket','spy','qqq']].notna().all().all()\nprint('Computed all exhibits from observed source data.')")]
    exhibits=[['constituents'],['analysis_filter_audit'],['trends','topic_summary'],['granger_core','granger_sensitivity','stationarity','leave_one_out'],['bias_descriptive','news_group_bias','bias_regressions','engagement_bias'],['basket_exclusions']]
    for index,(title,items) in enumerate(content):
        cells.append(nbf.v4.new_markdown_cell('## '+title))
        for item in items:
            if item['type']=='p':cells.append(nbf.v4.new_markdown_cell(item['text']))
            elif item['type']=='table':
                text='| '+' | '.join(item['headers'])+' |\n| '+' | '.join(['---']*len(item['headers']))+' |\n'
                text+='\n'.join('| '+' | '.join(map(str,r))+' |' for r in item['rows'])
                cells.append(nbf.v4.new_markdown_cell(text))
            else:
                cells.append(nbf.v4.new_code_cell(f"display(Image(filename=str(ROOT / 'outputs/{item['name']}.png')))"))
                cells.append(nbf.v4.new_markdown_cell(item['caption']))
        for name in exhibits[index]:
            prefix='data/processed' if name in ['constituents','analysis_filter_audit'] else 'outputs'
            cells.append(nbf.v4.new_code_cell(f"print('{name}')\ndisplay(pd.read_csv(ROOT / '{prefix}/{name}.csv'))"))
    cells += [nbf.v4.new_markdown_cell('## Reconciliation and source audit\n\nThe following checks validate counts and dates, not sentiment accuracy or structural causality. Detailed document-level flags, source requests and review texts remain local. The saved notebook contains aggregate exhibits only.'),
      nbf.v4.new_code_cell("news = json.loads((ROOT / 'outputs/news_manifest.json').read_text())\nsocial = json.loads((ROOT / 'outputs/social_audit.json').read_text())\nassert news['raw_discovered_rows'] == news['included_headlines'] + sum(news['exclusions'].values())\nassert social['unique_candidate_ids'] == social['clean_records'] + sum(x['removed_at_stage'] for x in social['filter_flow'])\nassert social['unresolved_truncated_leaf_requests'] == 0\ndocs = pd.read_csv(ROOT / 'data/processed/scored_documents.csv', low_memory=False)\nkept = docs.loc[docs.included_analysis]\nassert not kept.id.duplicated().any()\nassert pd.to_datetime(kept.date).max() <= pd.Timestamp(CUTOFF)\nassert len(kept) == sum(summary['documents'].values())\nprint('Counts, uniqueness, cutoff and HN query completeness checks passed.')\ndisplay(pd.read_csv(ROOT / 'outputs/language_summary_by_type.csv'))")]
    nb.cells=cells
    nb.metadata={'kernelspec':{'display_name':'Python 3 (Assignment 4)','language':'python','name':'assignment4'},
                 'language_info':{'name':'python','version':'3.12.0'}}
    if previous is not None:
        old_code=[c for c in previous.cells if c.cell_type=='code']
        new_code=[c for c in nb.cells if c.cell_type=='code']
        assert [c.source for c in old_code]==[c.source for c in new_code], 'Code changed; execute the notebook again.'
        snapshot=(ROOT/'SNAPSHOT.md').read_text(encoding='utf-8')
        for name in ['news.csv','social.csv','market_levels.csv','scored_documents.csv','language_audit.csv']:
            assert hashlib.sha256((ROOT/'data/processed'/name).read_bytes()).hexdigest() in snapshot, 'Inputs changed; execute again.'
        for old,new in zip(old_code,new_code):
            assert old.execution_count is not None
            new.execution_count=old.execution_count
            new.outputs=old.outputs
        nb.metadata=previous.metadata
    nbf.write(nb,ROOT/'assignment4.ipynb')
    print(f'Built report.tex and notebook with {len(cells)} cells.')

if __name__=='__main__':
    build_figures();content=sections();build_report(content);build_notebook(content)
