# Code obfuscation detection using Machine Learning
This repository contains the code for the code obfuscation detection using n-grams, entropy and string length.

### Dataset Sourcing
For the implementation, we have used the javascript obfuscation code dataset present on Kaggle. 
Obfuscated-javascript-dataset: 
https://www.kaggle.com/fanbyprinciple/obfuscated-javascript-dataset

The dataset comprises of:
1. Obfuscated javascript directory: This folder consists of javascript files (.js), where the syntax is utf-8 encoded so that the contextual understanding of the code is hidden.
2. Non-obfuscated javascript directory: This folder consists of javascript files (.js) where the code is not encoded and it's straightforward to understand it so that the contextual understanding of the code is visible. 

## Features
Our implementation, which aims at detecting malicious samples, is divided into several packages with distinct functionalities:
  - *js* for the detection of valid JavaScript code;
  - *features* for the extraction of specific features from inputs;

### JavaScript Detection Tool
Detection of JavaScript samples respecting the grammar, detection of broken JavaScript, and files not written in JavaScript.   
To use this tool: *python3 \<path-of-js/is_js.py\> --help*.

### Classification and Clustering of JavaScript Inputs
An AST-based analysis of JavaScript samples can be performed. This study is based on a frequency analysis of the n-grams present in the considered files.

- Detection of malicious JavaScript documents.   
To use this tool:  
1) *python \<path-of-clustering/learner.py\> --help*;  
2) *python \<path-of-clustering/updater.py\> --help*;  
3) *python \<path-of-clustering/classifier.py\> --help*.





