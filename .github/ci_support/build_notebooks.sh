#!/bin/bash
# pyiron config 
python .github/ci_support/pyironconfig.py

# import dataset
bash .github/ci_support/import_dataset.sh

# install nglview
bash .github/ci_support/install_nglview.sh

# conda install papermill
conda install -c conda-forge papermill

# execute notebooks
current_dir=$(pwd)
i=0;
for f in $(find . -name "example.ipynb" | sort -n); do
    cd $(dirname $f);
    notebook=$(basename $f);
    papermill ${notebook} ${notebook%.*}-out.${notebook##*.} -k "python3" || i=$((i+1));
    cd $current_dir;
done;

# push error to next level
if [ $i -gt 0 ]; then
    exit 1;
fi;
