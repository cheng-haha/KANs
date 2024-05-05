# The toy examples of KAN ( Kolmogorov-Arnold Network )

*Support MLP, KAN, and KAN derivatives*

KAN paper: [Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756)
<img width="1163" alt="326219527-695adc2d-0d0b-4e4b-bcff-db2c8070f841" src="https://github.com/cheng-haha/KANs/assets/54107313/70baa18c-5289-48bc-bfd2-50fbff89ba15">

KANs:
* [KANs](https://github.com/KindXiaoming/pykan)
* [Efficient-KAN](https://github.com/Blealtan/efficient-kan)
* [FourierKAN](https://github.com/GistNoesis/FourierKAN)
* Two-layer MLP (Toy Version)


## Run
```
python examples/mnist.py --model MLP
python examples/mnist.py --model KAN
python examples/mnist.py --model MNISTFourierKAN
```
## Experiential Settings

* For KAN, a large initial learning rate may be more effective. (You can try `lr = 1e-2`)
* In my experiments, KAN does converge faster than MLP.
