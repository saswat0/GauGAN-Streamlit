# GauGAN-Streamlit
Doodle to Image generator using GauGAN

This project is a checkpoint for my learning in streamlit. generates images from doodles using GauGan. It uses GauGAN to generate the images given the semantic maps (doodles). The model is based on Conditional GAN where given a particular image and a condition an image gets generated.


### Usage
* Create conda environment
    ```bash
    conda create -n ggan python=3.10
    ```
* Activate the environment
    ```bash
    conda activate ggan
    ```
* Install necessary packages
    ```bash
    pip install -r requirements.txt
    ```
* Run app
    ```bash
    streamlit run app.py
    ```