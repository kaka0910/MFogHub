# Time: 2020/02/22
# Author: Rosun
#!/usr/bin/env bash


PYTHON=${PYTHON:-"python"}


echo "Building dcn op..."
cd ../dcn
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace



