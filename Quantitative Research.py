# -------------------------------------------------------------------------------------------------------------------- #
#                                             Statistical Arbitrage Crypto                                             #
#                                                Quantitative Research                                                 #
# -------------------------------------------------------------------------------------------------------------------- #
"""
Title: Statistical Arbitrage Crypto
Script: Quantitative Research

Authors: Nicholas Wong
Creation Date: 27th January 2025
Modification Date: 27th January 2025

Purpose: This script builds a predictive model for the fatality of any given accident using various machine learning
         techniques. We use PCA for dimension reduction, SMOTE for imbalanced data, and build XGBoost and Neural Network
         models for classification.

Dependencies: pandas

Instructions: Ensure that the working directory is set to VicRoads-Fatalities

Data Sources: VicRoad Data obtained from https://discover.data.vic.gov.au/dataset/victoria-road-crash-data
- Accident Data
- Vehicle Data
- Accident Event Data
- Atmospheric Condition Data
- Sub DCA Data
- Person Data
- Node Data
- Road Surface Condition Data
- Accident Location Data

Fonts: "CMU Serif.ttf"

Table of Contents:
1. Data Integration
  1.1. Preamble
  1.2. Importing CSV Data
  1.3. Data Preparation
  1.4. Data Encoding
2. Data Cleaning
  2.1. Structuring Data
  2.2. NA Cleaning
3. Data Exploration
4. Data Transformation
  4.1. Transformation
  4.2. Dimension Reduction
"""

# ----------------------------------------------------------------------------------------------------------------------
# 1. Data Integration
# ----------------------------------------------------------------------------------------------------------------------

# 1.1. Preamble ----------------------------------------------------------------------------------------------