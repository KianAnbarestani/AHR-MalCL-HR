#!/usr/bin/env python3
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]
def md(text):return {'cell_type':'markdown','metadata':{},'source':[x+'\n' for x in text.splitlines()]}
def code(text):return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':[x+'\n' for x in text.splitlines()]}

cells=[
 md('# V15.5 Standard-A GPU execution\nThis notebook is only an interface to the packaged launcher. Select a Tesla T4 runtime first.'),
 code("from google.colab import drive\ndrive.mount('/content/drive')"),
 code("from pathlib import Path\nimport shutil\nZIP = Path('/content/drive/MyDrive/AHR_MalCL_V15_5_STANDARD_A_GPU_HANDOFF.zip')\nPACKAGE_DIR = Path('/content/AHR_MalCL_V15_5_STANDARD_A_GPU_HANDOFF')\nassert ZIP.is_file(), ZIP\nif PACKAGE_DIR.exists(): shutil.rmtree(PACKAGE_DIR)\nshutil.unpack_archive(str(ZIP), '/content')\nassert (PACKAGE_DIR / 'launch_v15_5_standard_a.py').is_file()\nprint(PACKAGE_DIR)"),
 code("!python -m pip install -q -r {PACKAGE_DIR / 'requirements.txt'}"),
 code("DATA_ROOT = '/content/data'\nOUTPUT_ROOT = '/content/drive/MyDrive/AHR_MalCL_V15_5_STANDARD_A'\nLAUNCHER = str(PACKAGE_DIR / 'launch_v15_5_standard_a.py')\nprint('DATA_ROOT =', DATA_ROOT)\nprint('OUTPUT_ROOT =', OUTPUT_ROOT)"),
 code("!python {LAUNCHER} --data-root {DATA_ROOT} --output-root {OUTPUT_ROOT} --preflight"),
 code("!python {LAUNCHER} --data-root {DATA_ROOT} --output-root {OUTPUT_ROOT} --smoke-tests"),
 code("!python {LAUNCHER} --data-root {DATA_ROOT} --output-root {OUTPUT_ROOT} --run-all --max-hours 10"),
 code("!python {LAUNCHER} --output-root {OUTPUT_ROOT} --status"),
 code("!python {LAUNCHER} --data-root {DATA_ROOT} --output-root {OUTPUT_ROOT} --validate"),
 code("from datetime import datetime, timezone\narchive_base = f\"/content/drive/MyDrive/AHR_MalCL_V15_5_STANDARD_A_OUTPUTS_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}\"\narchive = shutil.make_archive(archive_base, 'zip', root_dir=OUTPUT_ROOT)\nprint('Archived:', archive)"),
]
notebook={'cells':cells,'metadata':{'accelerator':'GPU','colab':{'gpuType':'T4','name':'V15_5_STANDARD_A_GPU_EXECUTION_COLAB.ipynb'},'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
(ROOT/'V15_5_STANDARD_A_GPU_EXECUTION_COLAB.ipynb').write_text(json.dumps(notebook,indent=2)+'\n')
print('notebook written')
