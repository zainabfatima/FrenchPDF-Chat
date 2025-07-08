# Multiperson Pose Estimation using YOLOv8 & Mediapipe 🤖🕺

This project combines YOLOv8 (for people detection) and Mediapipe (for pose estimation) to detect and visualize the poses of **multiple people** in a video.

---

## 🎯 Project Overview

Since **Mediapipe Pose** only supports **single-person detection**, this project:
- Uses **YOLOv8** to detect multiple people
- Crops each detected person out of the frame
- Applies **Mediapipe Pose** on each person
- Reconstructs the annotated full-frame video with all pose estimations

---

## 🧰 Requirements

Install dependencies:

```bash
pip install mediapipe ultralytics opencv-python numpy pandas
