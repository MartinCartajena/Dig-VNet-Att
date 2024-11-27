# Lung Nodule Segmentation with an Improved V-Net

This repository contains a replication of the **Lung Nodule Segmentation** model described in the paper:  
**"An improved V-Net lung nodule segmentation model based on pixel threshold separation and attention mechanism."**  

The objective of this work is to replicate and study the performance of the proposed model, leveraging its enhancements for precise segmentation of pulmonary nodules. The paper's innovations include:  

- **Pixel Threshold Separation:** A preprocessing step that improves segmentation accuracy by refining boundary delineation.  
- **Attention Mechanism:** Enhances the model's focus on critical regions, ensuring better performance in complex medical images.

By implementing this model, we aim to validate its efficacy and explore potential applications in clinical and research settings.

---

## Repository Structure  

- **`models/`**: Contains the implementation of the models arquitecture.  
- **`evaluate/`**: evaluation functions...
- **`utils/`**: util functions... 

---

## Requirements  

To run this project, ensure you have the following installed:  

- Python 3.9+  
- PyTorch (tested with version 2.0.0)  
- Other dependencies listed in `requirements.txt`.  

Install the dependencies with:  
```bash
pip install -r requirements.txt
