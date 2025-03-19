Nils Olivier
2025-03-21

# Algae Bloom Prediction

## Introduction

This repository contains the code for the Algae Bloom Prediction project from a course in GeoAI (AG2418) at KTH Royal Institute of Technology. The project is based on a second place solution to the [DrivenData TickTickBloom competition](https://github.com/drivendataorg/tick-tick-bloom), which was submitted by Andy Wheeler. This project is an extension of that work, with the goal of reproducing the results and analyzing the results.

The code is based on the work by [Andy Wheeler](https://github.com/drivendataorg/tick-tick-bloom/tree/main/2nd%20Place), and the code is extended by Nils Olivier. 

Nils Olivier
nolivier@kth.se

Andy Wheeler
apwheele@gmail.com

## Python Environment

Set up a python environment using Anaconda:

    conda create --name bloom python=3.9 pip geopandas
    conda activate bloom
    pip install -r requirements.txt
    ipython kernel install --name "bloom_jpy" --user

Where `requirements.txt` has the necessary libraries.

## Run the Code
everything should be run from root directory

First, prepare the local sqlite database with the data needed to run the models: 

    python main_prepdata.py

Then you can run the main file:

    python extension.py

This will save a model in the `./models` folder with the current date. 

You will also get a folder with figures which were used in the final report.

## Notes from competition approach (Andy Wheeler)

In addition to this, I have in the root folder `main_hypertune.py`, hypertuning experiments. And these results are saved in `hypertune_results.txt` (e.g. by running `python main_hypertune.py > hypertune_results.txt`. These helped guided the final models that I experiemented with, but the final ones are due to more idiosyncratic experimentation uploading every day.

To go over the modeling strategy, see the notebook `model_strategy.ipynb`. 