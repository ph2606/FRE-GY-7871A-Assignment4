"""Execute every notebook cell; save outputs only if every cell succeeds."""
from pathlib import Path
import os
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[1]
runtime=ROOT/'outputs/jupyter_runtime';runtime.mkdir(parents=True,exist_ok=True)
os.environ.setdefault('JUPYTER_RUNTIME_DIR',str(runtime))
os.environ.setdefault('IPYTHONDIR',str(ROOT/'outputs/ipython'))
path=ROOT/'assignment4.ipynb'
nb=nbformat.read(path,as_version=4)
NotebookClient(nb,timeout=600,kernel_name='assignment4',resources={'metadata':{'path':str(ROOT)}}).execute()
assert not [o for c in nb.cells if c.cell_type=='code' for o in c.get('outputs',[]) if o.output_type=='error']
nbformat.validate(nb);nbformat.write(nb,path)
print(f'Executed {sum(c.cell_type=="code" for c in nb.cells)} code cells with zero errors.')
