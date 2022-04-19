## Super Resolution

### Prepare data
Place your images in a directory. We will refer to this folder as INPUT_FOLDER.

Then, find a name for your output directory. We refer to this as OUTPUT_FOLDER.

Run and wait till completion:

``python prepare_data.py INPUT_FOLDER OUTPUT_FOLDER``

### Training
Run
``python main.py --help`` for full list of parameters

For training with default params, run:
``python main.py OUTPUT_FOLDER``, where OUTPUT_FOLDER corresponds to the folder generated in the previous step.