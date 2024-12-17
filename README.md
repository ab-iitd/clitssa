<h1 align="center">
<br>
[CLiTSSA] Multilingual LLMs Inherently Reward In-Language Time-Sensitive Semantic Alignment for Low-Resource Languages
</h1>


<p align="center">
  <a href="https://www.lcs2.in/"><b>[🌐 LCS2 Lab]</b></a> •
  <a href="https://arxiv.org/abs/2412.08090"><b>[📜 Paper]</b></a> •
  <a href="https://github.com/ab-iitd/clitssa/"><b>[🐱 GitHub]</b></a>
  
</p>


<p align="center">
Repo for "<a href="https://arxiv.org/abs/2412.08090" target="_blank">Multilingual LLMs Inherently Reward In-Language Time-Sensitive Semantic Alignment for Low-Resource Languages</a>"
</p>


## 🔥 News

- [2024/12/18] mTEMPREASON dataset is released over HuggingFace !
- [2024/12/09] CLiTSSA is accepted at AAAI 2025 (main track) !


## 💡 Framework

<p align="center" width="100%">
    <img src="clitssa_architecture.png" alt="img" height="360"/>
</p>


## 💡 Abstract

<details close>
<summary> Abstract of CLiTSSA</summary>

The unwavering disparity in labeled resources between resource-rich languages and those considered low-resource remains a significant impediment for Large Language Models (LLMs). Recent strides in cross-lingual in-context learning (X-ICL), mainly through semantically aligned examples retrieved from multilingual pre-trained transformers, have shown promise in mitigating this issue. However, our investigation reveals that LLMs intrinsically reward in-language semantically aligned cross-lingual instances over direct cross-lingual semantic alignments, with a pronounced disparity in handling time-sensitive queries in the X-ICL setup. Such queries demand sound temporal reasoning ability from LLMs, yet the advancements have predominantly focused on English. This study aims to bridge this gap by improving temporal reasoning capabilities in low-resource languages. To this end, we introduce mTEMPREASON a temporal reasoning dataset aimed at the varied degrees of low-resource languages and propose Cross-Lingual Time-Sensitive Semantic Alignment (CLiTSSA), a novel method to improve temporal reasoning in these contexts. To facilitate this, we construct an extension of mTEMPREASON comprising pairs of parallel cross-language temporal queries along with their anticipated in-language semantic similarity scores. Our empirical evidence underscores the superior performance of CLiTSSA compared to established baselines across three languages - Romanian, German, and French, encompassing three temporal tasks and including a diverse set of four contemporaneous LLMs. This marks a significant step forward in addressing resource disparity in the context of temporal reasoning across languages.

</details>


## 🔧 Repo Structure
This repo contains the training scripts and path for mTEMPREASON dataset. Detailed structure is as follow:
```
.
├── README.md
├── src
```
## mTEMPREASON Resource
The mTEMPREASON dataset is avaiable at HuggingFace. <a href="https://huggingface.co/datasets/ab-iitd/mTEMPREASON"><b>[Click Here]</b></a>

## Citation
If you find it helpful, please kindly cite the paper.
```
@misc{bajpai2024multilingualllmsinherentlyreward,
      title={Multilingual LLMs Inherently Reward In-Language Time-Sensitive Semantic Alignment for Low-Resource Languages}, 
      author={Ashutosh Bajpai and Tanmoy Chakraborty},
      year={2024},
      eprint={2412.08090},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.08090}, 
}
