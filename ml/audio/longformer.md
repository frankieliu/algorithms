# Longformer: The Long-Document Transformer

## Problem
How can LLM answer over long context window, example: question + huge document:
- long context window => $n^2$ scaling 

## Expensive
$O(n^2)$
![full attention](image.png)

## Global + local info
$O(n \times w)$
![local + global](image-1.png)

## Dilated sliding window
![local dilation](image-2.png)

## Receptive field
Receptive field $O(l \times w)$
![receptive field](./receptive_field.drawio.png)

## Global tokens:
eg. [CLS] for classification task
eg. Question + Document, let the question tokens attend to all of the document and vice versa

## Implementation
Two set of projections:
1. $Q_g, K_g, V_g$ - global
1. $Q_s, K_s, V_s$ - sliding window

## Window sizes
1. lower layers have smaller window size
2. larger window sizes at upper layers
