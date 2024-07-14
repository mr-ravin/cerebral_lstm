# Cerebral LSTM implementation in Pytorch
This repository contains experimental results and the comparitive study and implementation of `Cerebral LSTM`, presented in the paper "Cerebral LSTM: A Better Alternative for Single- and Multi-Stacked LSTM Cell-Based RNNs". Research paper is published in SN Computer Science Springer Nature Journal.

#### Paper Title: Cerebral LSTM: A Better Alternative for Single- and Multi-Stacked LSTM Cell-Based RNNs

#### Author: [Ravin Kumar](https://mr-ravin.github.io)

#### Publication: 14th March 2020

#### View Published Paper: [click here](https://link.springer.com/article/10.1007/s42979-020-0101-1)

#### PDF available on Research Gate: [click here](https://www.researchgate.net/publication/340013877_Cerebral_LSTM_A_Better_Alternative_for_Single-_and_Multi-Stacked_LSTM_Cell-Based_RNNs)

#### Doi: https://doi.org/10.1007/s42979-020-0101-1

## Cerebral LSTM Architecture:

![image](https://github.com/mr-ravin/cerebral-lstm/blob/master/CerebralLSTM.png?raw=true)

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

#### Cite as:
```
Kumar, R. Cerebral LSTM: A Better Alternative for Single- and Multi-Stacked LSTM Cell-Based RNNs. 
SN COMPUT. SCI. 1, 85 (2020). https://doi.org/10.1007/s42979-020-0101-1
```

### Pytorch Implementation:
- `Pytorch Implementation of Cerebral LSTM` is available in `Cerebral_LSTM/Cerebral_LSTM_Implementation_in_Pytorch.ipynb` file.
