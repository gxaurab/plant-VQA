
# Files and Folder structure 

plant-expert-vlm/
├── data/
│   ├── dataset.csv          # Metadata (image IDs, questions, answers)
│   └── images/              # Raw image files (.jpg, .png)
├── src/
│   ├── __init__.py
│   ├── dataset.py           # Loads CSV + formats Qwen Chat Template
│   └── utils.py             # Memory / VRAM helper utilities
├── configs/
│   └── lora_config.json     # Hyperparameters, batch sizes, learning rate
├── train.py                 # Main entrypoint script
├── requirements.txt         # Package dependencies
└── README.md
