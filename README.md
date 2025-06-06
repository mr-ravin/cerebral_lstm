# Cerebral LSTM - Implementation in Pytorch
This repository provides `python package` for `pytorch` implementation of `Cerebral LSTM`, presented in the paper "Cerebral LSTM: A Better Alternative for Single- and Multi-Stacked LSTM Cell-Based RNNs". Research paper is published in SN Computer Science Springer Nature Journal.

**Paper Title**: Cerebral LSTM: A Better Alternative for Single- and Multi-Stacked LSTM Cell-Based RNNs

**Author**: [Ravin Kumar](https://mr-ravin.github.io)

**Publication**: 14th March 2020

**Published Paper**: [click here](https://link.springer.com/article/10.1007/s42979-020-0101-1)

**Doi**: [DOI Link of Paper](https://doi.org/10.1007/s42979-020-0101-1)

**Other Sources**:
- [Research Gate](https://www.researchgate.net/publication/340013877_Cerebral_LSTM_A_Better_Alternative_for_Single-_and_Multi-Stacked_LSTM_Cell-Based_RNNs), [Research Gate - Preprint](https://www.researchgate.net/publication/382380649_Cerebral_LSTM_A_Better_Alternative_for_Single-_and_Multi-Stacked_LSTM_Cell-Based_RNNs)
- [Osf.io](https://osf.io/preprints/osf/jgh7p_v1)
- [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4897569)
- [Internet Archive](https://archive.org/details/cerebral-lstm-in-deep-learning-published-paper), [Internet Archive - Preprint](https://archive.org/details/cerebral-lstm-in-deep-learning--preprint-paper)

#### Github Repositories: 
- **Github Repository** (Python Package- Pytorch Implementation): [Python Package](https://github.com/mr-ravin/cerebral_lstm)
- **Github Repository** (Sentiment Analysis LSTM vs Cerebral LSTM): [ML Experiments](https://github.com/mr-ravin/cerebral-rnn-experimental-results)

#### Cite Paper as:
```
Kumar, R. Cerebral LSTM: A Better Alternative for Single- and Multi-Stacked LSTM Cell-Based RNNs. SN COMPUT. SCI. 1, 85 (2020). https://doi.org/10.1007/s42979-020-0101-1
```

---

## Cerebral LSTM Architecture:

![image](https://github.com/mr-ravin/cerebral-lstm/blob/main/CerebralLSTM.png?raw=true)

```
Uf(t) = σ(Wuf ⋅ [h(t − 1), x(t)] + buf)
Ui(t) = σ(Wui ⋅ [h(t − 1), x(t)] + bui)
UCtmp(t) = tanh (Wuc ⋅ [h(t − 1), x(t)] + buc)
UC(t) = Uf(t) ∗ UC(t − 1) + Ui(t) ∗ UCtmp(t)
Uo(t) = σ(Wuo ⋅ [h(t − 1), x(t)] + buo)
Lf(t) = σ(Wlf ⋅ [h(t − 1), x(t)] + blf)
Li(t) = σ(Wli ⋅ [h(t − 1), x(t)] + bli)
LCtmp(t) = tanh (Wlc ⋅ [h(t − 1), x(t)] + blc)
LC(t) = Lf(t) ∗ LC(t − 1) + Li(t) ∗ LCtmp(t)
Lo(t) = σ(Wlo ⋅ [h(t − 1), x(t)] + blo)
h(t) = Uo(t) ∗ tanh(UC(t)) + Lo(t) ∗ tanh(LC(t))
```

---

### Python Package: Pytorch Implementation

###### Tested with python version: >=3.7 and <= 3.13.2

##### 📥 Installation
```python
pip install cerebral_lstm
```

Or,

```python
pip install git+https://github.com/mr-ravin/cerebral_lstm.git
```

##### 🚀 Usage
```python
import torch
from cerebral_lstm import CerebralLSTM

# Create a Cerebral LSTM model, i.e. RNN  model with Cerebral LSTM cell unit
model = CerebralLSTM(input_size=64, hidden_size=128, num_layers=2, use_xavier=True, dropout=0.5) # Default: use_xavier=True

# Input: (seq_len, batch_size, input_size)
x = torch.randn(10, 32, 64)  # Example input
output, hidden = model(x)
print(output.shape)  # (10, 32, 128)
```

##### How to access only Cerebral LSTM cell unit ?
```python
from cerebral_lstm import CerebralLSTMCell

# Get only a single Cerebral LSTM cell unit
lstm_cell_unit = CerebralLSTMCell(input_size=64, hidden_size=128, use_xavier=True) # Default: use_xavier=True
```

---
### Impact of Initialisation of Trainable Parameters in Cerebral LSTM
The initial value of trainable parameters of upper and lower parts have impact onnumber of epochs required to train Cerebral LSTM cell. Ideally, upper and lowerparts should not have same initial values for their trainable parameters. 

##### Identical initial trainable parameter values for upper and lower parts ❌

  `Initial Symmetry`: Upper and lower parts of the Cerebral LSTM process inputs identically, leading to similar cell states Uc(t) and Lc(t).

  `Redundancy`: Initial representations of upper and lower parts are redundant, potentially under-utilizing the model’s capacity.

  `Gradients`: Early training updates are similar, but divergence may occur over time,leading to different feature extraction.

##### Different initial trainable parameter values for upper and lower parts ✔️

  `Diverse Learning`: Upper and lower parts of Cerebral LSTM immediately capture different aspects of the data, enhancing representation diversity.

  `Specialization`: Faster convergence and better utilization of the dual-path architec-ture, as each path can specialize in different features.

  `Performance`: Improved performance due to richer, non-redundant representationsfrom the start. 

---
**Experimentation Repository** in [https://github.com/mr-ravin/cerebral-rnn-experimental-results](https://github.com/mr-ravin/cerebral-rnn-experimental-results)

- Comparative Study - Cerebral LSTM vs LSTM: 

  `Pytorch Implementation of Cerebral LSTM` is available in `Cerebral_LSTM/Cerebral_LSTM_Implementation_in_Pytorch.ipynb` file.

- Comparative Study Cerebral LSTM vs Stacked-LSTM vs LSTM (Logs only)
    
    For the training loss graphs present in the research paper, see the below structure:
    ```
    |
    |-data/                             # This directory contains dataset used for comparison.
    |
    |-loss_values/                      # This directory contains record of training loss for each model to perform comparative analysis.
          |
          |- 2stack_lstm.txt 
          |- proposed_model.txt
          |- single_lstm.txt
    ```
    
---
#### Conclusion
Our proposed recurrent cell ‘Cerebral LSTM’ showed the ability to better understand data and has easily outperformed both single LSTM and two-stacked LSTM based recurrent neural networks. Many variants of Cerebral LSTM can be designed using available varieties of LSTM cells such as peephole LSTM. Further research work can be conducted on designing Cerebral LSTM based stacked recurrent neural networks for designing deep learning architectures for understanding time-series data. Other recurrent cells including gated recurrent units can also be analyzed after modifying itsinternal connections similar to our cerebral structure. 

---

Copyright License
```
Copyright (c) 2025 Ravin Kumar
Website: https://mr-ravin.github.io

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation 
files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, 
modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, 
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```
