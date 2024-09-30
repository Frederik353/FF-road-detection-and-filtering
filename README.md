# Fuelfighter Road Detection and Filtering

This is part of what i worked on for NTNU Fuelfighter for the 2023-2024 season.
This project focuses on detecting and filtering road segments for the Fuelfighter autonomous vehicle project. The goal is to improve the accuracy and efficiency of the vehicle's road detection system, enhancing its ability to navigate safely and efficiently.

## Table of Contents

1. [Overview](#overview)
2. [Motivation](#motivation)
3. [Features](#features)
4. [Road Detection Methodology](#road-detection-methodology)
5. [Filtering Technique](#filtering-technique)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Contributing](#contributing)
9. [License](#license)

## Overview

This project implements a system for road detection and filtering, which helps the Fuelfighter autonomous vehicle understand its environment by identifying road boundaries and filtering out noise and irrelevant information. The focus is on improving the vehicle's navigation capabilities in complex road environments using computer vision and data filtering techniques.

# Before


# After



## Motivation

With autonomous vehicles, it's essential to have reliable and efficient methods for detecting roadways and filtering out obstacles or noise. This project seeks to:
- Increase road detection accuracy in different environmental conditions.
- Reduce processing time and noise to allow smoother driving decisions.
- Improve overall vehicle safety and energy efficiency through smarter road navigation.

## Features

- **Real-time road detection**: Identifies the drivable area in real-time from camera feeds.
- **Filtering of irrelevant data**: Noise and non-road elements are filtered out to provide a clear signal.
- **Environment adaptability**: Works under varying conditions such as lighting, weather, and road types.

## Road Detection Methodology

The road detection system uses a combination of computer vision techniques:
- **AI Edge detection**: Identifies the boundaries of the road.
- **Lane line detection**: Recognizes lane markings and estimates the vehicle's position relative to the road.

### Tools and Libraries
- OpenCV for image processing
- TensorFlow/PyTorch for any machine learning models used for road segmentation (if applicable)
- NumPy for data handling


## Filtering Technique

To enhance the accuracy of road detection, the system relies on AI-generated masks combined with filtering techniques. The core approach involves processing lane line estimates from YOLO PV2 (a real-time object detection algorithm) and refining them using several methods:

- **YOLO PV2 Mask**: The AI model provides an initial mask of the estimated lane lines, which serves as the starting point for further filtering.
- **Hough Transform**: The Hough Line Transform is used to detect and emphasize straight lines in the masked image, particularly for identifying lane boundaries. This step is critical for refining the lane detection and removing irrelevant or noisy line segments.
- **Smoothing Techniques**: After identifying the lane lines, an **Exponential Moving Average (EMA)** is applied to prioritize recent data points while still accounting for past information. This helps stabilize lane detection over time, reducing the impact of short-term noise and fluctuations, and ensuring smoother transitions between frames.

This combined approach ensures that the detected lane lines are accurate and reliable, providing the autonomous system with a clear understanding of the road ahead.

