# Datasets
* Original Dataset Link: https://huggingface.co/datasets/SyedNazmusSakib/PlantExpertVQA
* Stratified Dataset Drive Link: https://drive.google.com/drive/folders/16WxkEQ5NczknlBrdOAAFktwaBcGoUtWE?usp=sharing
    - Train Images reduced from : 105k to 12.6k
    - Val Images only remained original of 15k (but in the /configs/train.yaml we will be using only 1500)
    - Train Images: Original of 30k
    - 1 image = 5 QnA

    + Inside drive
        - test_images.zip, train_images.zip and val_images.zip are the zip files containing Images
        - data/train.csv, test.csv, val.csv are the QnA's and has metadata

# Guide on training
- required files: 
    - configs/train.yaml
    - src/dataset.py
    - train.py
    - requirements.py

    + Make venv
    + pip install -r requirements.txt
    + python train.py --config configs/train.yaml
