<h1 align="left">
    Mask Detection
</br>
</h1>

[Google Meet Link](https://meet.google.com/zat-ugag-hvi)

## Description
The goal of this project is to implement an end-to-end machine learning pipeline to detect 
if a person wears a mask in an image. Concretely it can be used in pratice to check if 
someone is respecting sanitary measures about COVID-19.


## Requirements & Specifications
- Clean code & Reproductibility
- A vision model trained from scratch to predict presence of wearable mask in an image
- Compliance with best pratices in ML process
- Deployement using a common-used framework

## Getting started


## Project Development


### Dataset
The dataset used, `Covid-19-PIS Dataset`, is a collection of images stratified in two classes 
which are the groups of images of people wearing a mask to comply to COVID-19 measures and 
a group of images of people with their faces uncovered. See below in the reference 
`covid-19-pis_dataset` for more details.

### Model
The model aims to be a supervised one since images labels are known. Built with 
pytorch framework, the vision model is neural-nets based with the following architecture 

### Deployment


## References & Citations
[Hands-on machine Learning with ScikitLearn](https://drive.google.com/file/d/1Ic13Zuk2FAUZem-3h75YSfSVbJi04Jp4/view?usp=sharing)

```bibtex
@misc{ covid-19-pis_dataset,
    title = { Covid-19-PIS Dataset },
    type = { Open Source Dataset },
    author = { PyImageSearch },
    howpublished = { \url{ https://universe.roboflow.com/pyimagesearch/covid-19-pis } },
    url = { https://universe.roboflow.com/pyimagesearch/covid-19-pis },
    journal = { Roboflow Universe },
    publisher = { Roboflow },
    year = { 2023 },
    month = { mar },
    note = { visited on 2023-12-02 },

}
```
