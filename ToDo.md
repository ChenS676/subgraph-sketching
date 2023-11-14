# Project Name Readme

## Overview

This README.md provides an overview of the implementation plan for the project, outlining key tasks related to data construction, data benchmarking, model benchmarking, visualization, and demos.

## Implementation Plan

### 1. Data Construction
| name | text | original |split | resource | visualization
|----------|----------|----------|----------|----------|----------|
| Cora | &check; | &check; | Random | &check; | &#10060; |
| pubmed | &check; | &check; | random | &check; |&#10060; |
| ogbn-arxiv | &check;  | &check;  | random for nc, default for lp | &check;  |&#10060; |
| ogbn-citation2 | &#10060;|&#10060;|&#10060;|&#10060;|&#10060;|
| ogbn-products | &#10060;|&#10060;|&#10060;|&#10060;|&#10060;|

#### Tasks
- [ ] add arxiv as a new dataset
- [ ] add your dataset as a new dataset
- [ ] visualize graph and graph statistical metrics 
- [ ] exchange with pierre

### 3. Model Benchmark
| name            | cora    | pubmed  | ogbn-arxiv | resource | visualization | analysis |
|-----------------|---------|---------|------------|----------|----------------|----------|
| load_embed      | &check; | &check; | Random     | &check;  | &#10060;        |  consistently no improvement  |
| pretrain_embed  |&#10060; |&#10060; |&#10060;  | &check;  | &#10060;        | analysis |


#### Tasks
- [ ] SEAL, SIGN, NeoGNN
- [ ] Define performance metrics for model evaluation.
- [ ] Train and validate models on benchmark datasets.
- [ ] Compare model performance and document results.

#### Timeline
Specify estimated start and end dates for each task.

### 4. Visualization

#### Tasks
- [ ] Identify key insights and visualizations needed for analysis.
- [ ] Choose appropriate tools (e.g., Matplotlib, Seaborn) for visualization.
- [ ] Develop and refine visualizations.

#### Timeline
Specify estimated start and end dates for each task.

### 5. Demos

#### Tasks
- [ ] Design interactive demos to showcase project features.
- [ ] Implement demo functionality.
- [ ] Test and debug demos.

#### Timeline
Specify estimated start and end dates for each task.

## Getting Started

Follow the steps below to get started with the project:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repository.git
