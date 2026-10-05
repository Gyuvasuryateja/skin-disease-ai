# MobileNetV2 HAM10000 Training Report

## Scope and safety

This is an AI image-classification experiment for educational and research purposes. It is not a medical diagnosis tool and must not be used as a substitute for clinical assessment.

## Model and training

- Architecture: MobileNetV2 with frozen ImageNet-pretrained feature extractor and 7-class linear head
- Device: cpu
- Parameters: 2,232,839 total; 8,967 trainable.
- Epochs completed: 7
- Best epoch: 5
- Best validation accuracy: 0.4416
- Test accuracy: 0.4458
- Training time: 37501.5 seconds
- Imbalance handling: class-weighted cross-entropy, with weights computed exclusively from the training split.
- Test-set policy: the test split was not iterated during model selection, scheduling, or early stopping.

## Class-wise untouched test performance

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| `akiec` | 0.2185 | 0.6735 | 0.3300 | 49 |
| `bcc` | 0.3163 | 0.4026 | 0.3543 | 77 |
| `bkl` | 0.2597 | 0.3593 | 0.3015 | 167 |
| `df` | 0.2593 | 0.4118 | 0.3182 | 17 |
| `mel` | 0.1908 | 0.5385 | 0.2817 | 169 |
| `nv` | 0.9135 | 0.4308 | 0.5855 | 1,005 |
| `vasc` | 0.3404 | 0.7619 | 0.4706 | 21 |

## Saved artifacts

- `best_model.pt`: MobileNetV2 state dictionary and architecture metadata for FastAPI inference.
- `class_label_mapping.json`: stable class-code-to-index mapping.
- `inference_preprocessing.json`: exact RGB, resize/padding, and normalization configuration.
- `training_history.json`, `training_validation_accuracy.png`, and `training_validation_loss.png`: training history and curves.
- `test_metrics.json`, `classification_report.json`, `confusion_matrix.csv`, and `confusion_matrix.png`: final untouched-test evaluation.
