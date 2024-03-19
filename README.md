Faster Cheque Clearing System
 
This project aims to automate the cheque clearing process using machine learning techniques such as Convolutional Neural Networks (CNNs) and pre-trained models like VGG16. The system is designed to process cheque images, perform technical verifications, and detect potential frauds, thereby reducing human efforts and processing time.

Problem Statement

Bank cheque clearing processes involve manual verification steps, which are time-consuming and labor-intensive. The goal is to automate these processes using AI/ML/ICR/OCR techniques.

Project Goals

Automation of the cheque clearing process
Automatic Data Entry & Technical Verification
Signature Verification
Support for Multilingual cheques
Reduction of Human Efforts and Processing Time
Detection of Potential Frauds

Components

Image Preprocessing: Resize cheque images to a standard size.
Model Training: Train a CNN model for classification.
Data Augmentation: Augment training data using ImageDataGenerator.
Transfer Learning: Utilize pre-trained VGG16 model for feature extraction.
Evaluation: Evaluate model performance using classification report and confusion matrix.

Requirements

Python 3.x
OpenCV (cv2)
NumPy
scikit-learn
TensorFlow
matplotlib

Usage

Clone the repository.
Place cheque images in the Images folder.
Run the main.py script to train and evaluate the models.
Check the output for training/validation accuracy and loss.
Review the classification report and confusion matrix for model performance.

Future Enhancements

Incorporate more advanced machine learning techniques.
Fine-tune pre-trained models for better performance.
Implement real-time cheque processing capabilities.
Enhance user interface for easier interaction.

Contributors

[M.Harini]




License

This project is licensed under the MIT License.

