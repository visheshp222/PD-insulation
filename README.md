
# Visibility Graph-Based Partial Discharge Recognition of Insulator Defects

## 📌 Overview

This project presents an intelligent approach for detecting and classifying **partial discharge (PD) signals** generated due to defects in high-voltage insulators using **Visibility Graph (VG) analysis** and **Machine Learning**.

Instead of extracting only traditional statistical features from PD signals, the time-series signal is converted into a **Visibility Graph**, where every signal sample becomes a graph node and edges are created based on the visibility criterion. Graph-theoretic features are then extracted and used to train machine learning models for defect classification.

This approach combines **signal processing**, **graph theory**, and **machine learning** to improve the accuracy and reliability of insulator condition monitoring.

---

## 🎯 Objectives

- Detect Partial Discharge (PD) signals from high-voltage insulators.
- Convert PD time-series signals into Visibility Graphs.
- Extract graph-based features.
- Train machine learning models for defect recognition.
- Compare different classification algorithms.
- Improve predictive maintenance in power transmission systems.

---

# Project Workflow


HFCT Sensor
      │
      ▼
PD Signal Acquisition
      │
      ▼
CSV Signal Dataset
      │
      ▼
Signal Preprocessing
      │
      ▼
Visibility Graph Conversion
      │
      ▼
Graph Feature Extraction
      │
      ▼
Machine Learning Model
      │
      ▼
Insulator Defect Classification




# Dataset

The dataset consists of sampled Partial Discharge signals stored in CSV format.

Example:

| Time | Amplitude |
|------|-----------|
|0.0000|0.23|
|0.0001|0.28|
|0.0002|0.34|
|...|...|

Each CSV file represents one PD signal.


# Technologies Used

- Python
- Pandas
- NumPy
- NetworkX
- Matplotlib
- Scikit-learn
- SciPy



# Required Libraries


pip install pandas
pip install numpy
pip install matplotlib
pip install networkx
pip install scikit-learn
pip install scipy

# Project Structure



# Methodology

## Step 1: Data Acquisition

Partial discharge signals are captured using a **High Frequency Current Transformer (HFCT)** installed on the grounding conductor of the insulator or cable termination.



## Step 2: Signal Preprocessing

The raw PD signals are

- Filtered
- Normalized
- Converted into time-series format


## Step 3: Visibility Graph Conversion

Each point in the signal becomes a node.

Two nodes are connected if they satisfy the Natural Visibility Graph condition.

This transforms the waveform into a complex graph.

## Step 4: Graph Feature Extraction

The following features are extracted:

- Number of Nodes
- Number of Edges
- Average Degree
- Graph Density
- Clustering Coefficient
- Betweenness Centrality
- Degree Centrality
- Average Shortest Path
- Graph Diameter
- Network Entropy

## Step 5: Machine Learning

The extracted graph features are used to train machine learning models.

Algorithms used:

- Random Forest
- Support Vector Machine (SVM)


## Step 6: Prediction

The trained model predicts whether the insulator is

- Healthy
- Surface Contamination
- Crack
- Surface Discharge
- Internal Defect



# Machine Learning Models

## Random Forest

Advantages

- High accuracy
- Handles nonlinear data
- Robust against overfitting
- Feature importance analysis



## Support Vector Machine (SVM)

Advantages

- Effective for high-dimensional data
- Good generalization
- Suitable for small datasets



# Results

Example performance

| Model | Accuracy |
|--------|----------|
|Random Forest|97.8%|
|SVM|95.4%|

*(Example values for demonstration. Replace with your experimental results.)*



# Applications

- High Voltage Transmission Lines
- Electrical Substations
- GIS Systems
- Cable Terminations
- Power Transformers
- Bushings
- Underground Cable Monitoring



# Advantages

- Non-invasive monitoring
- Early fault detection
- Reduced maintenance cost
- Improved grid reliability
- High classification accuracy
- Graph-based representation captures complex PD patterns



# Future Scope

- Deep Learning using Graph Neural Networks (GNN)
- Real-time monitoring
- IoT-enabled online PD monitoring
- Edge AI deployment
- Cloud-based fault diagnosis
- Multi-sensor data fusion



# Sample Output


Loading PD Dataset...

Converting Signal to Visibility Graph...

Extracting Graph Features...

Training Random Forest...

Accuracy : 97.8%

Prediction : Crack Defect

Confidence : 98.3%

# Repository


Visibility Graph-Based Partial Discharge Recognition of Insulator Defects

# Author

**Vishesh Patel**

B.Tech Electrical Engineering

National Institute of Technology Durgapur

# References

1. IEEE Transactions on Dielectrics and Electrical Insulation
2. IEEE Transactions on Power Delivery
3. Lacasa et al., "From Time Series to Complex Networks: The Visibility Graph", PNAS, 2008.
4. IEC 60270 - High Voltage Partial Discharge Measurement Standard.
