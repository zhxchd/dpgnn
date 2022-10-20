# Graph neural networks with link local differential privacy

This repository hosts the code and documents for my project on GNN with link local DP.

## Dependencies

To run the experiments in this repo, you need `numpy`, `matplotlib`, `sklearn`, `torch`, `torch_sparse`, `torch_geometric`. You can install all the dependencies is through `conda` (please use the `CUDA` version applicable to your system):

```
conda install -y pytorch cudatoolkit=11.6 -c pytorch -c conda-forge
conda install -y pyg -c pyg
```

## File structure
- `./src`: the source directory of all the mechanisms, datasets and models we have experimented with.
  - `./src/blink` implements our main result, the Blink framework.
  - `./src/rr` implements vanilla randomized response as a baseline.
  - `./src/ldpgcn` implement a LDP variant of DPGCN from [Wu et al (2022)](https://ieeexplore.ieee.org/document/9833806).
  - `./src/solitude` tries to implement Solitude from [Lin et al (2022)](https://ieeexplore.ieee.org/document/9855440).
  - `./src/data` contains all the code to download, pre-process and load graph datasets including Cora, CiteCeer and LastFM.
  - `./src/models` contains all the code to build GNN models including GCN, GraphSage and GAT.
- `./scripts` is the directory of Python scripts to run experiments.
  - `./scripts/run_blink.sh` runs the Blink framework with specified settings.
  - `./scripts/run_baselines.sh` runs baseline methods with specified settings.
  - `./scripts/log` stores all the log files when running the scripts above.
  - `./scripts/output` stores all the results (hyperparameter choices and final accuracy).
- `./doc` is the root directory for the paper describing the proposed method.

## License
The code and documents are licensed under the MIT license.
```
MIT License

Copyright (c) 2022 Xiaochen Zhu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```