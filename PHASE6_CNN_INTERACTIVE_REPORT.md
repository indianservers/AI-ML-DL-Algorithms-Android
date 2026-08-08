# Phase 6 CNN Interactive Report

| Feature | Interactive | Step Mode | Real Math | Visualization | Trainable | Tests | Status |
| ------- | ----------- | --------- | --------- | ------------- | --------- | ----- | ------ |
| Pixel Grid | Yes | Yes | Yes | numeric heatmap grid | N/A | Yes | Implemented |
| Kernel | Yes | Yes | Yes | editable/preset 3x3 concept | Learned concept | Yes | Implemented |
| Convolution | Yes | Yes | Yes | patch x kernel -> output cell | N/A | Yes | Implemented |
| Stride | Yes | Yes | Yes | jump distance and output size | N/A | Yes | Implemented |
| Padding | Yes | Yes | Yes | zero border and size change | N/A | Yes | Implemented |
| Feature Map | Yes | Yes | Yes | heatmap + values | N/A | Yes | Implemented |
| ReLU | Yes | Yes | Yes | raw to activated map | N/A | Yes | Implemented |
| Max Pooling | Yes | Yes | Yes | selected pooling region | N/A | Yes | Implemented |
| Multi-Channel Conv | Yes | Yes | Yes | R/G/B contributions summed | N/A | Yes | Implemented |
| Multiple Filters | Yes | Yes | Yes | three feature maps | Tiny classifier | Yes | Implemented |
| CNN Shape Tracker | Yes | Yes | Yes | layer-by-layer tensor shapes | N/A | Yes | Implemented |
| Parameter Count | Yes | Yes | Yes | live formula and count | N/A | Yes | Implemented |
| Receptive Field | Concept | Yes | Yes | highlighted input patch | N/A | Partial | Implemented conceptually |
| Tiny CNN | Yes | Yes | Yes | synthetic shape classifier | Yes | Yes | Implemented |
| Feature Map Explorer | Yes | Yes | Yes | selected feature map | Yes | Yes | Implemented |
| Learned Filters | Concept | Yes | Yes | filter weights + activations | Yes | Yes | Implemented |
| Training | Yes | Yes | Yes | loss/accuracy trace | Yes | Yes | Implemented |
| Misclassification Explorer | Concept | Yes | Probability state | prediction probabilities | Yes | Partial | Conceptual |

## Components Reused

- Learn routing and shared lab UI components
- Canvas visual design from previous phases
- Phase 5 softmax and training explanation style

## New Components

- `PixelMatrixVisualizer`
- `CalculationCard`
- `TrainingChartCnn`
- CNN shape and parameter tracker

## Simplifications

- Uses cross-correlation, with an explanatory note because common CNN libraries call this convolution.
- Tiny image classifier uses deterministic educational filters and local training trace.
- No large image tensors are rendered cell by cell.

## Phase 7 Preparation

- Extract pixel/matrix visualizers into shared components for later vision tasks.
- Add coroutine-backed training before heavier CNN architectures.
