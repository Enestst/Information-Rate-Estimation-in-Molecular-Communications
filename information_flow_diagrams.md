# Information Flow Diagrams

## 1. Activity Diagram - Data Generation Process

```mermaid
flowchart TD
    Start([Start Data Generation]) --> SampleParams[Sample Physical Parameters]
    SampleParams --> ComputeHitting[Compute Hitting Probabilities]
    ComputeHitting --> DetermineL[Determine Memory Length L]
    DetermineL --> SelectThreshold[Select Detection Threshold]
    SelectThreshold --> GenerateSeqs[Generate All 2^L Sequences]
    GenerateSeqs --> ComputeBER[Compute BER via Gaussian Approx]
    ComputeBER --> SaveBuffer{Buffer Full?}
    SaveBuffer -->|No| SampleParams
    SaveBuffer -->|Yes| SaveCSV[Save to CSV]
    SaveCSV --> CheckStop{Stop Signal?}
    CheckStop -->|No| SampleParams
    CheckStop -->|Yes| End([End])
```

## 2. Activity Diagram - Model Training Process

```mermaid
flowchart TD
    Start([Start Training]) --> LoadData[Load CSV Dataset]
    LoadData --> Preprocess[Preprocess Data]
    Preprocess --> Split[Stratified Train/Val/Test Split]
    Split --> InitModel[Initialize Model Architecture]
    InitModel --> FitScalers[Fit Scalers on Training Data]
    FitScalers --> EpochStart[Start Epoch]
    
    EpochStart --> TrainBatch[Train on Batches]
    TrainBatch --> CheckStability{NaN/Inf Detected?}
    CheckStability -->|Yes| StopTraining[Stop Training]
    CheckStability -->|No| ValidateModel[Validate on Val Set]
    
    ValidateModel --> UpdateLR[Update Learning Rate]
    UpdateLR --> CheckBest{Best Val Loss?}
    CheckBest -->|Yes| SaveBest[Save Best Checkpoint]
    CheckBest -->|No| CheckEarly{Early Stop?}
    SaveBest --> CheckEarly
    
    CheckEarly -->|No| EpochStart
    CheckEarly -->|Yes| TestModel[Evaluate on Test Set]
    TestModel --> SaveArtifacts[Save Model & Scalers]
    SaveArtifacts --> End([End])
    StopTraining --> End
```

## 3. Sequence Diagram - Training Workflow

```mermaid
sequenceDiagram
    participant User
    participant DataLoader
    participant Preprocessor
    participant Model
    participant Optimizer
    participant Storage
    
    User->>DataLoader: Load CSV data
    DataLoader->>Preprocessor: Raw samples
    Preprocessor->>Preprocessor: Log transform
    Preprocessor->>Preprocessor: Feature engineering
    Preprocessor->>Preprocessor: Fit scalers
    Preprocessor->>DataLoader: Preprocessed features
    
    loop For each epoch
        DataLoader->>Model: Batch of features
        Model->>Model: Forward pass
        Model->>Optimizer: Compute loss
        Optimizer->>Optimizer: Backward pass
        Optimizer->>Model: Update weights
        Model->>User: Log metrics
        
        alt Best validation loss
            Model->>Storage: Save checkpoint
        end
        
        alt Early stopping triggered
            Model->>Storage: Save final model
            Storage->>User: Training complete
        end
    end
```

## 4. Sequence Diagram - Inference Workflow

```mermaid
sequenceDiagram
    participant User
    participant System
    participant Scalers
    participant Model
    
    User->>System: Provide channel parameters
    System->>System: Compute hitting probabilities
    System->>System: Compute variances
    System->>System: Engineer 6-channel features
    System->>Scalers: Standardize features
    Scalers-->>System: Normalized features
    System->>Model: Forward pass
    Model-->>System: Log₁₀(BER) prediction
    System->>System: Convert to raw BER
    System-->>User: Return BER estimate
```

## 5. Activity Diagram - Complete Pipeline

```mermaid
flowchart LR
    subgraph "Phase 1: Data Generation"
        A[Sample Parameters] --> B[Compute Channel Response]
        B --> C[Calculate BER]
        C --> D[Save Dataset]
    end
    
    subgraph "Phase 2: Training"
        D --> E[Load & Preprocess]
        E --> F[Train Model]
        F --> G[Validate & Save]
    end
    
    subgraph "Phase 3: Inference"
        G --> H[Load Model]
        H --> I[New Parameters]
        I --> J[Predict BER]
    end
```

## 6. Business Process Model - System Workflow

```mermaid
flowchart TD
    Start([Researcher Needs BER Estimate]) --> Decision1{Have Trained Model?}
    
    Decision1 -->|No| GenData[Generate Training Data]
    GenData --> TrainModel[Train Neural Network]
    TrainModel --> ValidateModel[Validate Performance]
    ValidateModel --> Decision2{Meets Accuracy?}
    Decision2 -->|No| TuneParams[Tune Hyperparameters]
    TuneParams --> TrainModel
    Decision2 -->|Yes| SaveModel[Save Model & Scalers]
    
    Decision1 -->|Yes| LoadModel[Load Model]
    SaveModel --> LoadModel
    
    LoadModel --> InputParams[Input Channel Parameters]
    InputParams --> Preprocess[Preprocess Features]
    Preprocess --> Inference[Run Inference]
    Inference --> Output[Get BER Prediction]
    Output --> Decision3{Satisfactory?}
    
    Decision3 -->|Yes| End([Use Results])
    Decision3 -->|No| Decision4{Need More Data?}
    Decision4 -->|Yes| GenData
    Decision4 -->|No| InputParams
```

## 7. Data Flow Diagram

```mermaid
graph LR
    A[Physical Parameters] -->|Analytical Formulas| B[Hitting Probabilities]
    B -->|2^L Evaluation| C[BER Values]
    C -->|Preprocessing| D[Feature Tensors]
    D -->|Training| E[Model Weights]
    E -->|Inference| F[BER Predictions]
    
    G[Scalers] -.->|Normalize| D
    G -.->|Normalize| F
```

## Process Descriptions

### Data Generation Process
1. Sample physical parameters from uniform distributions
2. Compute hitting probabilities using erfc-based formulas
3. Determine adaptive memory length based on 70% coverage
4. Select detection threshold from realistic range
5. Generate all 2^L bit sequences
6. Calculate BER for each sequence using Gaussian approximation
7. Average BER across all sequences
8. Save to CSV when buffer is full
9. Repeat until interrupted

### Training Process
1. Load CSV dataset and validate columns
2. Apply log transformations to handle wide ranges
3. Engineer 6-channel feature sequences
4. Fit StandardScalers on training data
5. Split data with stratification (70/15/15)
6. Initialize model architecture
7. For each epoch:
   - Train on mini-batches
   - Check for numerical instabilities
   - Validate on validation set
   - Update learning rate if plateau detected
   - Save checkpoint if best validation loss
   - Check early stopping criterion
8. Evaluate on test set
9. Save final model and scalers

### Inference Process
1. Load trained model checkpoint and scalers
2. Accept new physical channel parameters
3. Compute hitting probabilities and variances
4. Construct 6-channel feature sequence
5. Apply standardization using loaded scalers
6. Run forward pass through model
7. Convert log₁₀(BER) prediction to raw BER
8. Return prediction to user
