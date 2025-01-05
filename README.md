# <img src="img/logo.png" style="vertical-align: -10px;" :height="40px" width="40px"> Dispider
This repository is the official implementation of Dispider.


<img align="center" src="img/pipeline.png" style="  display: block;
  margin-left: auto;
  margin-right: auto;
  width: 100%;" />

<p align="center" style="font-size: em; margin-top: 0.5em">

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)<br>
<a href="https://arxiv.org/abs/xxxx"><img src="https://img.shields.io/badge/arXiv-paper-<color>"></a>
<a href="assets/paper.pdf"><img src="https://img.shields.io/badge/PDF-red"></a>
<!-- <a href="https://mark12ding.github.io/project/SAM2Long/"><img src="https://img.shields.io/badge/Project-Homepage-green"></a> -->
<!-- <a href="https://huggingface.co/spaces/Mar2Ding/SAM2Long-Demo"><img src="https://img.shields.io/badge/🤗Hugging Demo-yellow"></a> -->
</p>



>[**Dispider: Enabling Video LLMs with Active Real-Time Interaction via Disentangled Perception, Decision, and Reaction**](https://arxiv.org/abs/xxxxx)<br>
> [Rui Qian](https://shvdiwnkozbw.github.io/), [Shuangrui Ding](https://mark12ding.github.io/), [Xiaoyi Dong](https://lightdxy.github.io/), [Pan Zhang](https://panzhang0212.github.io/)<br>
[Yuhang Zang](https://yuhangzang.github.io/), [Yuhang Cao](https://scholar.google.com/citations?user=sJkqsqkAAAAJ), [Dahua Lin](http://dahua.site/), [Jiaqi Wang](https://myownskyw7.github.io/)<br>
CUHK, Shanghai AI Lab


## 📰 News
- [2025/1/6] 🔥🔥🔥 We released the paper on arXiv!

## 🧾 ToDo Lists
- [ ] Release Inference Code and Checkpoints
- [ ] Release Training Code
- [ ] Release Demo Video


## 💡 Highlights
### 🔥 A New Paradigm for Online Video LLMs with Active Real-Time Interaction
Dispider introduces a groundbreaking approach by enabling active, real-time interactions with streaming videos. Unlike traditional offline video LLMs that require processing the entire video before generating responses, Dispider operates seamlessly in live scenarios, providing continuous and timely feedback as the video streams.

### ⚡️ Disentangled Perception, Decision, and Reaction Modules Operating Asynchronously
Dispider's architecture uniquely separates perception, decision-making, and reaction generation into distinct, asynchronous modules. This disentangled design allows each component to function in parallel, ensuring that video processing and response generation do not block each other. The perception module continuously monitors the video stream, the decision module assesses when to trigger interactions based on the processed data, and the reaction module generates detailed responses without interrupting ongoing video analysis.

### 🤯 Superior Performance on StreamingBench and Conventional Video Benchmarks
Extensive evaluations demonstrate Dispider's exceptional performance across various benchmarks. On StreamingBench, Dispider outperforms VideoLLM-online in temporal grounding, proactive response generation, and multi-step reasoning. Additionally, it surpasses traditional offline Video LLMs on long-video benchmarks such as EgoSchema, VideoMME, and MLVU, as well as time-sensitive tasks in ETBench. Dispider excels in temporal reasoning and effectively manages diverse video lengths, showcasing its robustness and versatility in real-time video understanding.



## ☎️ Contact
Shuangrui Ding: mark12ding@gmail.com


## 🔒 License
The majority of this project is released under the CC-BY-NC 4.0 license as found in the LICENSE file. 


## 👍 Acknowledgements
This codebase is built upon [LLaVA](https://github.com/haotian-liu/LLaVA) and leverages several open-source libraries. We extend our gratitude to the contributors and maintainers of these projects.

<!-- 
## ✒️ Citation
If you find our work helpful for your research, please consider giving a star ⭐ and citation 📝.
```bibtex

``` -->


