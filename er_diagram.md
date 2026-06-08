# Information Structure - ER Diagram

## Entity-Relationship Diagram

```mermaid
erDiagram
    CHANNEL_PARAMETERS {
        float radius
        float distance
        float diffusion_coefficient
        float symbol_duration
        int molecule_count
    }
    
    CHANNEL_RESPONSE {
        int memory_length
        float_array hitting_probabilities
        float_array variances
        float threshold
    }
    
    TRAINING_SAMPLE {
        int sample_id
        string sample_type
        float BER
    }
    
    FEATURE_SEQUENCE {
        float_array scaled_means
        float_array log_variances
        float_array position_encoding
        float_array absolute_values
        float_array snr_proxy
        float_array first_flag
    }
    
    MODEL_CHECKPOINT {
        string model_name
        datetime timestamp
        dict state_dict
        float validation_loss
    }
    
    SCALER_OBJECTS {
        string scaler_name
        object scaler_instance
    }
    
    CHANNEL_PARAMETERS ||--|| CHANNEL_RESPONSE : "generates"
    CHANNEL_RESPONSE ||--|| TRAINING_SAMPLE : "produces"
    TRAINING_SAMPLE ||--|| FEATURE_SEQUENCE : "preprocessed_to"
    FEATURE_SEQUENCE }o--|| MODEL_CHECKPOINT : "trains"
    FEATURE_SEQUENCE }o--|| SCALER_OBJECTS : "normalized_by"
```

## ER Diagram

```mermaid
erDiagram
    PhysicalParameters ||--|| ChannelResponse : computes
    ChannelResponse ||--|| TrainingSample : calculates
    TrainingSample ||--o{ Features : transforms
    Features }o--|| Model : trains
    
    PhysicalParameters {
        float radius
        float distance
        float diffusion
        float Ts
        int N
    }
    
    ChannelResponse {
        int L
        array P
        array Var
        float threshold
    }
    
    TrainingSample {
        int id
        float BER
    }
    
    Features {
        array channels_6
    }
    
    Model {
        dict weights
        float loss
    }
```

## Data Flow Description

### 1. Physical Channel Parameters
- **Attributes**: radius, distance, diffusion coefficient, symbol duration, molecule count
- **Source**: Randomly sampled from uniform distributions
- **Purpose**: Define the molecular communication channel

### 2. Channel Response
- **Attributes**: memory length (L), hitting probabilities (P[0..L-1]), variances, threshold
- **Derived from**: Physical parameters via analytical formulas
- **Purpose**: Characterize channel impulse response and ISI

### 3. Training Sample
- **Attributes**: sample ID, sample type (physics/synthetic), BER value
- **Derived from**: Channel response via $2^L$ sequence evaluation
- **Purpose**: Ground truth labels for supervised learning

### 4. Feature Sequence
- **Attributes**: 6-channel tensor (means, variances, position, abs, SNR, first-flag)
- **Derived from**: Training sample via log transformations and standardization
- **Purpose**: Model input representation

### 5. Model Checkpoint
- **Attributes**: model name, timestamp, state dictionary, validation loss
- **Produced by**: Training process
- **Purpose**: Persistent storage of trained weights

### 6. Scaler Objects
- **Attributes**: scaler name, fitted StandardScaler instance
- **Produced by**: Preprocessing on training data
- **Purpose**: Consistent normalization for inference

## Key Relationships

- **generates**: Physical parameters uniquely determine channel response
- **produces**: Channel response produces one training sample (BER value)
- **preprocessed_to**: Each training sample transforms to one feature sequence
- **trains**: Many feature sequences collectively train one model
- **normalized_by**: Feature sequences require scaler objects for standardization
