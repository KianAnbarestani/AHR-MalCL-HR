# Final MalCL Baseline Colab Packaging Commands

Run this locally from the terminal:

```bash
echo "v52-zip - create CLEAN tiny final MalCL baseline Colab package with 3-epoch protocol and in-process tqdm progress bars"

cd /home/kian/projects/shz_uni/paper-works

rm -rf /tmp/final_malcl_colab_pkg
rm -f final_malcl_baseline_colab_package.zip

mkdir -p /tmp/final_malcl_colab_pkg/AHR-MalCL/paper_ready

rsync -a AHR-MalCL/paper_ready/final_malcl_baseline_colab_run/ \
  /tmp/final_malcl_colab_pkg/AHR-MalCL/paper_ready/final_malcl_baseline_colab_run/ \
  --exclude "__pycache__/" \
  --exclude ".ipynb_checkpoints/" \
  --exclude "*.zip" \
  --exclude "*.pdf" \
  --exclude "raw_outputs/" \
  --exclude "logs/" \
  --exclude "tables/" \
  --exclude "memory/" \
  --exclude "validation/"

OFFICIAL_SRC="AHR-MalCL/paper_ready/phase9a_malcl_reproduction_design/external_code/MalCL_official"
OFFICIAL_DST="/tmp/final_malcl_colab_pkg/AHR-MalCL/paper_ready/phase9a_malcl_reproduction_design/external_code/MalCL_official"

mkdir -p "$OFFICIAL_DST"

rsync -a "$OFFICIAL_SRC/MalCL_torch/" "$OFFICIAL_DST/MalCL_torch/" \
  --exclude "__pycache__/" \
  --exclude ".ipynb_checkpoints/"

cp "$OFFICIAL_SRC/LICENSE" "$OFFICIAL_DST/LICENSE"
cp "$OFFICIAL_SRC/Readme.md" "$OFFICIAL_DST/Readme.md"
git -C "$OFFICIAL_SRC" rev-parse HEAD > "$OFFICIAL_DST/OFFICIAL_COMMIT.txt"

find /tmp/final_malcl_colab_pkg -type d \( \
  -name ".venv" -o -name ".venv_stats" -o -name "venv" -o -name "env" -o \
  -name "__pycache__" -o -name ".ipynb_checkpoints" -o -name "site-packages" \
\) -prune -exec rm -rf {} +

find /tmp/final_malcl_colab_pkg -type f \( \
  -name "*.npz" -o -name "*.pt" -o -name "*.pth" -o -name "*.ckpt" -o \
  -name "*.tar" -o -name "*.tar.gz" -o -name "*.zip" -o -name "*.pdf" -o \
  -name "*.pyc" -o -name "*.pyo" \
\) -delete

cd /tmp/final_malcl_colab_pkg

zip -r /home/kian/projects/shz_uni/paper-works/final_malcl_baseline_colab_package.zip AHR-MalCL

cd /home/kian/projects/shz_uni/paper-works

ls -lh final_malcl_baseline_colab_package.zip

echo "Check largest files inside zip:"
unzip -l final_malcl_baseline_colab_package.zip | sort -nr | head -30

echo "Check forbidden payload is absent:"
if unzip -l final_malcl_baseline_colab_package.zip | grep -E '(^|/)(\.git|\.venv|\.venv_stats|venv|env|site-packages|__pycache__|data|result|result_external_baselines)/|\.npz|\.pt|\.pth|\.ckpt|\.tar|\.tar\.gz|\.zip|\.pdf|\.pyc|\.pyo'; then
  echo "ERROR: forbidden files are still in the package"
  exit 1
fi
```

Upload:

```text
/home/kian/projects/shz_uni/paper-works/final_malcl_baseline_colab_package.zip
```

to:

```text
/content/drive/MyDrive/final_malcl_baseline_colab_package.zip
```

Then paste `final_malcl_baseline_colab_one_cell.py` into one Colab GPU cell.

The official MalCL `.git` directory is intentionally not shipped. The package writes `OFFICIAL_COMMIT.txt`, and the Colab validator accepts that marker when git metadata is absent.
