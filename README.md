# EUMothClassifier
Example Project demonstrating a ViT-B/16 Vision Transformer Model for classifying European moth species.

### Purpose

The code in this repository trains a Vision Transformer Model on image data from the EU Moths Dataset (*inf-cv.uni-jena.de/home/research/datasets/eu_moths_dataset/*) with PyTorch and serves the model in Gradio for easy access/demonstration. 
This is mostly a stripped-down toy example for demonstration purposes, so accuracy may be subpar for certain cases.

### How to use

#### Clone the repository
```
git clone https://github.com/EeveelutionaryBiologist/EUMothClassifier.git
cd EUMothClassifier/
```

We will need to setup an environment for the dependencies. The probably easiest way is to use venv:


#### Virtual Environment
```
python -m venv .venv
pip install -r requirements.txt
```

We then need to download the training data externally:
```
git clone https://github.com/cvjena/eu-moths-dataset.git
```

If you are in the EUMothClassifier directory, you can just leave it there. Otherwise copy it to the root of this repository, as that
is where the training pipeline will look for it. The folder structure should look like this:
```
EUMothClassifier/
  eu-moths-dataset/
  main.py
  ...
```

Now we can run the training pipeline. Depending on available hardware that may take a moment:
```
python main.py
```

Afterwards, the trained model should exist in the App/Model directory. To serve the model and predict some examples, 
go to the App/ subdirectory and run the Gradio client like this:
```
cd App/
python app.py
```
The client should be served on the localhost http://127.0.0.1:7860. If you manaeuver there in your browser, you should see an interface.
For an idea on what the client should look like, compare the image in Screenshots/.

:)
