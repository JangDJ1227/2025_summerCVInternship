# 2025_summerCVInternship
I made Korea car plate detection model on real time

## Project Overview

This project is part of the 2025_summerCVInternship program and focuses on implementing a real-time vehicle license plate recognition system using computer vision techniques.  
The application captures video frames from a USB camera and applies plate detection and character recognition to extract license numbers.

In addition to direct camera input, the system can also operate with RTSP video feeds via VLC, allowing flexibility in testing environments.

Two plate types were evaluated:

New License Plates  
Structured formatting and clearer typography result in highly reliable detection and recognition performance.

Old License Plates  
Variations in aging, font, and spacing lead to reduced consistency and were mainly used to validate detection feasibility.

This repository demonstrates practical application of OpenCV pipelines, character extraction, and model-based OCR in a lightweight real-time setting.

---

## Tech Stack

Python 3.x  
OpenCV (cv2) for image acquisition, preprocessing, contour extraction  
NumPy for numerical operations in image handling  
Matplotlib for optional visualization

Optional Input Extensions  
VLC for RTSP passthrough, enabling RTSP feeds to be mapped as a virtual camera device

Algorithm Components  
Plate region detection via thresholding and contour filtering  
Character segmentation pipeline  
OCR classification module that can be replaced or extended

---

## How to Run

1. Install dependencies
