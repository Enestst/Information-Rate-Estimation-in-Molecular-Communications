# Welcome to Information Rate Estimation in Molecular Communications

Welcome to the Wiki for the **Information Rate Estimation in Molecular Communications** project. 

This repository serves as a comprehensive toolkit for simulating, analyzing, and ultimately predicting the performance of Molecular Communication via Diffusion (MCvD) channels. 



## 📖 Project Overview

Molecular communication is a bio-inspired paradigm where information is encoded in the release, propagation, and reception of molecules. While the physics of these channels (Brownian motion, diffusion) are well understood, calculating the theoretical limits of their digital communication performance is highly complex.

Our overarching goal is to bypass the heavy computational bottlenecks of traditional analytical formulas by developing AI-driven estimation models capable of instantly predicting channel metrics.

### Phase 1: The Starting Point (BER)
To build a strong foundation in molecular communication, we began by modeling a fundamental digital metric: the **Bit Error Rate (BER)**. Using On-Off Keying (OOK) modulation and an absorbing spherical receiver, we are currently in the developing phase for building a model that estimates BER using a dataset of theorotical values.

### The Core Problem: The Curse of Dimensionality
In a diffusion-based channel, molecules do not simply disappear after their designated symbol duration ($T_s$). They linger in the channel, causing severe **Inter-Symbol Interference (ISI)**. 

If a channel has a memory length of $L$ bits, calculating the exact BER requires computing the conditional probabilities of all $2^L$ possible interference patterns. As the memory length grows, brute-forcing these mathematical formulas becomes computationally intractable. 

### Phase 2: The End Goal (AI & Graph Networks)
Instead of relying on $O(2^L)$ brute-force calculations for environments with long channel memory, this project aims to train advanced Machine Learning models—such as deep neural networks and Graph Neural Networks (GNNs)—to estimate critical features like:
* **Bit Error Rate (BER)**
* **Channel Capacity / Information Rate**

By generating massive datasets using our rigorous physical and analytical simulators, we believe we can train AI models to understand the nonlinear mapping between physical channel coefficients (distance, diffusion rate, molecule count) and digital communication performance.

---
