# Gut Microbiome-Based Health Classification using Machine Learning

This repository contains the code and models developed for the mini project **"Gut Microbiome-Based Health Classification using Machine Learning"**, submitted by **G. Kalyani Krishna (22011102017)** as part of the B.Tech in Computer Science and Engineering (IoT) curriculum at **Shiv Nadar University Chennai**.

## Overview

This project focuses on leveraging machine learning and statistical techniques to assess human gut health through microbiome data. A user-friendly **Flask web application** has been developed that classifies microbiome samples into **healthy** or **unhealthy** based on a composite health index derived from PCA (Principal Component Analysis), alongside other microbiome-specific indicators.

## Objectives

- Build a PCA-based health index for gut microbiome classification.
- Create a web interface to allow users to upload microbiome profiles (CSV format) and receive real-time health predictions.
- Provide an interactive simulator for exploring the effects of microbial feature changes on health outcomes.
- Promote transparency via species-level contribution analysis (Bacteria-to-Health-Index Contribution - BHC).

## Key Features

- **T², Q, and Combined Indices**: Based on PCA to capture deviation from healthy microbiome distributions.
- **Gut Microbiome Health Index (GMHI)** and **Shannon Entropy**: Provide diversity and health metrics.
- **Interactive Simulator**: Modify microbial features and observe health prediction changes in real-time.
- **Species-Level Diagnosis**: Identify microbes contributing to unhealthy classifications using BHC analysis.

## Dataset

The microbiome data used in this project is from the **2025 CAMDA Challenge**, including:
- `taxonomy.txt`: Species-level microbial abundances (via MetaPhlAn)
- `pathways.txt`: Functional profiles (via HumanN)
- `metadata.txt`: Health labels, GMHI, Shannon indices, etc.

## Methodology

1. **Data Preprocessing**: Normalization (TSS, log/sqrt), missing value imputation, low-variance filtering.
2. **Feature Engineering**: Computation of GMHI, Shannon Entropy, and hiPCA-based indices.
3. **Dimensionality Reduction**: PCA on functional and taxonomic subsets.
4. **Classification**: Rule-based logic using thresholds on PCA-derived scores.
5. **Visualization**: Real-time dashboards and PCA plots.

## Installation & Setup

```bash
git clone https://github.com/yourusername/gut-microbiome-health-classifier.git
cd gut-microbiome-health-classifier
pip install -r requirements.txt
