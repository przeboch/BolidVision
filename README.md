"BolidVision:Classification and Visual Analysis of F1 cars"
Project description: The project uses advanced visual analysis techniques based on the VGG16 model and tools such as Grad-CAM and LIME. 
Its main goal is to automatically recognize cars in photos and classify them based on key features.
The model was trained on photos of Red Bull and Mercedes cars.

BolidVision/
│
├── data/
│   ├── raw/                     # Raw data
|   ├── test_dataset/            # Data for testing model
│   ├── processed/               # Processed data
|   ├── model_explanations/      # LIME results
│
├── notebooks/                   # Jupyter Notebooks or other notebooks
│
├── src/                         # Source code
│   ├── data/                    # Scripts for data acquisition, processing, and cleaning
│   ├── features/                # Scripts for feature extraction
│   ├── models/                  # Scripts for model training and evaluation
│   ├── visualization/           # Scripts for data visualization
│   ├── functions_in_progress/   # Scripts for data visualization
│
├── models/                      # Project configuration files
│
├── reports/                     # Project reports and documentation
│
├── requirements.txt             # Dependency list
|
├── setup.py                     # Configuring and installing packages
│
├── README.md                    # Project documentation
