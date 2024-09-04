# Autonomous F1 Racing

### Project By Aditya Subramanian

---

## Table of Contents

- [Introduction](#introduction)
- [Objectives](#objectives)
- [Framework and Architecture](#framework-and-architecture)
  - [Data Collection and Preprocessing](#data-collection-and-preprocessing)
  - [Model Architecture: AlexNet](#model-architecture-alexnet)
  - [Training Process](#training-process)
  - [Deployment and Real-Time Operation](#deployment-and-real-time-operation)
  - [Performance Evaluation and Iteration](#performance-evaluation-and-iteration)
- [Methodology](#methodology)
  - [Creating the Training Data](#creating-the-training-data)
  - [Balancing Training Data](#balancing-training-data)
  - [Training the Model](#training-the-model)
  - [Testing the Autonomous F1 Racing Model](#testing-the-autonomous-f1-racing-model)
- [Future Implementation](#future-implementation)

---

## Introduction

The rise of autonomous technologies, from health diagnostics to motorsports, has been remarkable. Autonomous Formula 1 racing, particularly within simulated environments like the F1 2020 game, represents a fascinating convergence of advanced artificial intelligence and high-speed racing. This project aims to design and implement an autonomous racing system for the F1 2020 game, incorporating machine learning algorithms, sensor data processing, and real-time decision-making.

The goal is to develop an AI-driven racing agent capable of participating in F1 2020 races autonomously, navigating complex tracks, optimizing racing strategies, and dynamically adapting during races to achieve competitive lap times. This report details the technical approach taken, including the architecture of the AI system, the algorithms used, and the challenges encountered. It also evaluates the performance of the autonomous agent against human players and existing AI systems, highlighting successes and areas for future improvement.

---

## Objectives

1. **AlexNet-Based Autonomous Racing Agent Development**: Design and implement an AI-driven racing agent using the AlexNet architecture to process real-time visual data and make decisions within the F1 2020 game.

2. **Data Capture and Preprocessing**: Capture and preprocess screen frames during human gameplay, embedding each frame with corresponding keystroke inputs to create a dataset for training the AlexNet model.

3. **Training and Optimization**: Train the AlexNet model using the dataset to predict keystrokes and implement racing strategies based on visual inputs from the game environment.

4. **Real-Time Decision Making and Adaptation**: Fine-tune the model for real-time decision-making during races, allowing it to adapt to dynamic in-race conditions and optimize performance across different tracks in F1 2020.

5. **Performance Evaluation and Benchmarking**: Assess the effectiveness of the trained AlexNet model by comparing its performance against human players and existing AI systems, identifying areas for improvement.

6. **Advancing Autonomous Racing Technologies**: Contribute to the research in autonomous racing by documenting development, challenges, and successes, providing insights into the potential of convolutional neural networks like AlexNet in AI-driven motorsports.

---

## Framework and Architecture

### 1. Data Collection and Preprocessing

- **Capture the Gameplay Data**: Frames are captured from the screen during gameplay in the F1 2020 game. The screen frames are preprocessed to suit the model, and keystrokes corresponding to acceleration, braking, or steering are embedded in each frame.

- **Frame Preprocessing**: Captured frames are resized, converted to grayscale, and normalized to ensure consistency. These preprocessed frames are then used as inputs for the AlexNet model.

- **Keystroke Embedding**: Keystroke data for each frame is encoded into a binary array, representing the different keystroke combinations, which serve as input for the neural network.

### 2. Model Architecture: AlexNet

- **Input Layer**: The model takes grayscale images with dimensions `[160x120x1]`.

- **Convolutional Layers**:
  - **First Convolutional Layer**: 96 filters of size 11x11 with a stride of 4 and ReLU activation.
  - **First Pooling Layer**: Max-pooling with a 3x3 kernel and stride of 2, followed by Local Response Normalization (LRN).
  - **Second Convolutional Layer**: 256 filters of size 5x5 with ReLU activation, followed by max-pooling and LRN.
  - **Third, Fourth, and Fifth Convolutional Layers**: 384, 384, and 256 filters of size 3x3 with ReLU activation, with the fifth layer followed by max-pooling and LRN.

- **Fully Connected Layers**:
  - **First Fully Connected Layer**: 4096 units with tanh activation and dropout.
  - **Second Fully Connected Layer**: 4096 units with tanh activation and dropout.

- **Output Layer**: Softmax-activated output vector into 8 classes corresponding to possible keystrokes or their combinations.

- **Regression Layer**: Categorical cross-entropy loss is used, with momentum optimization applied to update model weights during training.

### 3. Training Process

- **Supervised Learning**: The model is trained by mapping input frames to their corresponding keystroke labels, minimizing categorical cross-entropy loss.

- **Optimization**: Momentum optimizer is used to accelerate convergence.

- **Data Splitting**: The dataset is split into training, validation, and testing sets, with the validation set used for hyperparameter tuning and the test set for final performance evaluation.

- **Epochs**: Training is performed for a set number of epochs, with the model saved after training.

### 4. Deployment and Real-Time Operation

- **Real-Time Decision-Making**: The trained model is deployed in the F1 2020 game, processing live video feeds and predicting keystrokes to control the car autonomously.

- **Adaptation to Dynamic Environments**: The model adapts to race dynamics, making split-second decisions based on visual input.

### 5. Performance Evaluation and Iteration

- **Benchmarking**: Performance is monitored using metrics like accuracy, with snapshots taken periodically.

- **Iterative Improvement**: The model is retrained based on performance outcomes to fine-tune its capabilities.

---

## Methodology

### Creating the Training Data

1. **Screen Capture and Preprocessing**: Frames are captured from the F1 2020 game and converted to grayscale. Images are resized to 160x120 pixels, balancing detail with computational efficiency.

2. **Keystroke Capture and Encoding**: Keystrokes are detected during gameplay and converted into a multi-hot encoded array representing the player's actions.

3. **Data Collection and Storage**: Training data pairs (preprocessed frame and keystroke array) are saved at intervals to avoid data loss, with the option to pause and resume data collection.

4. **Final Data Preparation**: Once sufficient data is collected, the dataset is saved for training the AI model.

### Balancing Training Data

1. **Loading and Preprocessing**: The dataset is loaded and converted into a DataFrame to analyze class distribution.

2. **Categorizing Data**: Data is sorted into lists based on action type, and each list is truncated to match the smallest class size.

3. **Combining and Shuffling**: The balanced lists are combined into a final dataset, shuffled, and saved for training.

### Training the Model

1. **Model Architecture and Configuration**: The AlexNet model is initialized with the appropriate parameters, and the training data is loaded and split.

2. **Data Preprocessing**: Input images and labels are reshaped to fit the model's requirements.

3. **Model Training**: The model is trained on the dataset, with validation performed during training. Snapshots are taken periodically.

4. **Model Saving and Logging**: The trained model is saved, and the training process is logged for analysis in TensorBoard.

5. **Evaluation and Iteration**: The model's performance is evaluated, with iterative improvements made based on the results.

### Testing the Autonomous F1 Racing Model

1. **Model and Environment Setup**: The trained model is loaded, and screen capture is set up in the game.

2. **Preprocessing Screen Data**: Captured images are converted to grayscale and resized before being passed to the model for prediction.

3. **Model Prediction and Action Mapping**: The model predicts actions based on the input image, and the corresponding keystrokes are executed in the game.

4. **Testing Loop and Control**: The model is tested in a continuous loop, with real-time performance monitored. The process can be paused and resumed as needed.

5. **Model Performance Evaluation**: Predicted actions are printed for real-time observation, and further improvements are made based on performance.

---

## Future Implementation

1. **Improved Model Architecture**: Explore deeper models like ResNet or transfer learning to improve accuracy and efficiency.

2. **Advanced Data Augmentation**: Implement synthetic data generation and real-time augmentation to enhance training robustness.

3. **Reinforcement Learning**: Use reinforcement learning to optimize driving strategies based on rewards and penalties.

---
