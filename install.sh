#!/bin/bash

packages=(
    "flask"
    "flask_wtf"
    "flask_babel"
    "bs4"
    "lxml"
    "scikit-learn"
    "pandas"
    "sem"
)

for package in "${packages[@]}"; do
    pip install "$package"
done

