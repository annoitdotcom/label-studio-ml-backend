## MMOCR

```
conda create -n env37 python=3.7 -y
conda activate env37

# install latest pytorch prebuilt with the default prebuilt CUDA version (usually the latest)
conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.1 -c pytorch

# install the latest mmcv-full
pip install mmcv-full==1.3.4

# install mmdetection
pip install mmdet==2.11.0

# install mmocr
git clone https://github.com/open-mmlab/mmocr.git
cd mmocr

pip install -r requirements.txt
pip install -v -e .  # or "python setup.py develop"
export PYTHONPATH=$(pwd):$PYTHONPATH
```




```
label-studio-ml start my_ml_backend --with det_config=./mmocr/configs/textdet/dbnet/dbnet_r50dcnv2_fpnc_1200e_icdar2015.py det_ckpt=./data/dbnet_r50dcnv2_fpnc_sbn_1200e_icdar2015_20210325-91cef9af.pth recog_config=./mmocr/configs/textrecog/tps/crnn_tps_academic_dataset.py recog_ckpt=./data/crnn_tps_academic_dataset_20210510-d221a905.pth image_dir='/home/lionel/data_labeling_website/data/media/upload' device='cuda'
```

## MMDETECTION
```
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

# install latest pytorch prebuilt with the default prebuilt CUDA version (usually the latest)
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -r requirements/build.txt
pip install "git+https://github.com/open-mmlab/cocoapi.git#subdirectory=pycocotools"
pip install -v -e .

pip install mmcv-full==1.3.8 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html

label-studio-ml init coco-detector --from label_studio_ml/examples/mmdetection/mmdetection.py

label-studio-ml start coco-detector --with config_file=mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py checkpoint_file=./data/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth image_dir='/home/lionel/data_labeling_website/data/media/upload' device=cuda -p 9091 
```

## What is the Label Studio ML backend?

The Label Studio ML backend is an SDK that lets you wrap your machine learning code and turn it into a web server.
You can then connect that server to a Label Studio instance to perform 2 tasks:

- Dynamically pre-annotate data based on model inference results
- Retrain or fine-tune a model based on recently annotated data

If you just need to load static pre-annotated data into Label Studio, running an ML backend might be overkill for you. Instead, you can [import preannotated data](https://labelstud.io/guide/predictions.html).

## How it works

1. Get your model code
2. Wrap it with the Label Studio SDK
3. Create a running server script
4. Launch the script
5. Connect Label Studio to ML backend on the UI


## Quickstart

Follow this example tutorial to run an ML backend with a simple text classifier:

0. Clone the repo
   ```bash
   git clone https://github.com/heartexlabs/label-studio-ml-backend  
   ```
   
1. Setup environment
    
    It is highly recommended to use `venv`, `virtualenv` or `conda` python environments. You can use the same environment as Label Studio does. [Read more](https://docs.python.org/3/tutorial/venv.html#creating-virtual-environments) about creating virtual environments via `venv`.
   ```bash
   cd label-studio-ml-backend
   
   # Install label-studio-ml and its dependencies
   pip install -U -e .
   
   # Install example dependencies
   pip install -r label_studio_ml/examples/requirements.txt
   ```
   
2. Initialize an ML backend based on an example script:
   ```bash
   label-studio-ml init my_ml_backend --script label_studio_ml/examples/simple_text_classifier.py
   ```
   This ML backend is an example provided by Label Studio. See [how to create your own ML backend](#Create_your_own_ML_backend).

3. Start ML backend server
   ```bash
   label-studio-ml start my_ml_backend
   ```
   
4. Start Label Studio and connect it to the running ML backend on the project settings page.

## Create your own ML backend

Follow this tutorial to wrap existing machine learning model code with the Label Studio ML SDK to use it as an ML backend with Label Studio. 

Before you start, determine the following:
1. The expected inputs and outputs for your model. In other words, the type of labeling that your model supports in Label Studio, which informs the [Label Studio labeling config](https://labelstud.io/guide/setup.html#Set-up-the-labeling-interface-for-your-project). For example, text classification labels of "Dog", "Cat", or "Opossum" could be possible inputs and outputs. 
2. The [prediction format](https://labelstud.io/guide/predictions.html) returned by your ML backend server.

This example tutorial outlines how to wrap a simple text classifier based on the [scikit-learn](https://scikit-learn.org/) framework with the Label Studio ML SDK.

Start by creating a class declaration. You can create a Label Studio-compatible ML backend server in one command by inheriting it from `LabelStudioMLBase`. 
```python
from label_studio_ml.model import LabelStudioMLBase

class MyModel(LabelStudioMLBase):
```

Then, define loaders & initializers in the `__init__` method. 

```python
def __init__(self, **kwargs):
    # don't forget to initialize base class...
    super(MyModel, self).__init__(**kwargs)
    self.model = self.load_my_model()
```

There are special variables provided by the inherited class:
- `self.parsed_label_config` is a Python dict that provides a Label Studio project config structure. See [ref for details](). Use might want to use this to align your model input/output with Label Studio labeling configuration;
- `self.label_config` is a raw labeling config string;
- `self.train_output` is a Python dict with the results of the previous model training runs (the output of the `fit()` method described bellow) Use this if you want to load the model for the next updates for active learning and model fine-tuning.

After you define the loaders, you can define two methods for your model: an inference call and a training call. 

### Inference call

Use an inference call to get pre-annotations from your model on-the-fly. You must update the existing predict method in the example ML backend scripts to make them work for your specific use case. Write your own code to override the `predict(tasks, **kwargs)` method, which takes [JSON-formatted Label Studio tasks](https://labelstud.io/guide/tasks.html#Basic-Label-Studio-JSON-format) and returns predictions in the [format accepted by Label Studio](https://labelstud.io/guide/predictions.html).

**Example**

```python
def predict(self, tasks, **kwargs):
    predictions = []
    # Get annotation tag first, and extract from_name/to_name keys from the labeling config to make predictions
    from_name, schema = list(self.parsed_label_config.items())[0]
    to_name = schema['to_name'][0]
    for task in tasks:
        # for each task, return classification results in the form of "choices" pre-annotations
        predictions.append({
            'result': [{
                'from_name': from_name,
                'to_name': to_name,
                'type': 'choices',
                'value': {'choices': ['My Label']}
            }],
            # optionally you can include prediction scores that you can use to sort the tasks and do active learning
            'score': 0.987
        })
    return predictions
```


### Training call
Use the training call to update your model with new annotations. You don't need to use this call in your code, for example if you just want to pre-annotate tasks without retraining the model. If you do want to retrain the model based on annotations from Label Studio, use this method. 

Write your own code to override the `fit(annotations, **kwargs)` method, which takes [JSON-formatted Label Studio annotations](https://labelstud.io/guide/export.html#Raw-JSON-format-of-completed-labeled-tasks) and returns an arbitrary dict where some information about the created model can be stored.

**Example**
```python
def fit(self, completions, workdir=None, **kwargs):
    # ... do some heavy computations, get your model and store checkpoints and resources
    return {'checkpoints': 'my/model/checkpoints'}  # <-- you can retrieve this dict as self.train_output in the subsequent calls
```


After you wrap your model code with the class, define the loaders, and define the methods, you're ready to run your model as an ML backend with Label Studio. 

For other examples of ML backends, refer to the [examples in this repository](label_studio_ml/examples). These examples aren't production-ready, but can help you set up your own code as a Label Studio ML backend.
