# 🧬 RiboFlow: Conditional *De Novo* RNA Co-Design via Synergistic Flow Matching

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**RiboFlow** is a ligand-conditioned framework for **RNA sequence–structure co-design** based on **synergistic flow matching**. It jointly models **continuous RNA 3D backbone geometry** (backbone frames / torsion angles) and **discrete nucleotide identities**, enabling controllable *de novo* RNA design toward specific ligands.


---

## ✨ Highlights

- **Joint sequence–structure generation**: co-designs RNA backbone geometry and nucleotide sequence in one unified generative process.
- **RNA geometric priors**: explicitly represents RNA using **backbone frames** and **torsion angles** to capture conformational flexibility.
- **Ligand conditioning**: conditions RNA generation on ligand 3D geometry for binding-oriented design.
- **Data foundation**: introduces **RiboBind**, a curated RNA–ligand interaction dataset to alleviate structural data scarcity.

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

---

## 🗂️ Data

### RiboBind (RNA–ligand interactions)

RiboBind is curated from PDB following the procedure described in the **paper appendix**. To obtain the dataset used in the paper, please perform the data extraction exactly as described (structure filtering, ligand validation, interaction extraction, redundancy control, etc.).

### RNAsolo (pretraining structures)

For geometric pretraining, we use a filtered RNAsolo subset (paper setting): resolution ≤ **4 Å** (as of **Dec 2024**) and sequence length **30–200 nt**, yielding **7,154** training samples.

---

## 🚀 Usage

### Training

- **Pre-training** (default parameters; 4 GPUs)
```bash
accelerate launch --num_processes=4 --gpu_ids=0,1,2,3 train_pretraining.py
```

- **Ligand-binding training** (default parameters; 2 GPUs)
```bash
accelerate launch --num_processes=2 --gpu_ids=0,1 train.py
```

> Adjust dataset paths / batch sizes / checkpoints in the config section of the scripts (or project config files, if applicable).

---

## 📁 Repository Structure (high-level)

- `train_pretraining.py` — geometric pretraining on RNA structures
- `train.py` — ligand-conditioned training/fine-tuning
- `data/` — dataset folder (place processed snapshots here)
- (other modules/utilities are documented inline in the codebase)

---

## 📚 Acknowledgements

We build upon the following outstanding open-source projects:

- [protein-frame-flow](https://github.com/microsoft/protein-frame-flow)
- [rna-backbone-design](https://github.com/rish-16/rna-backbone-design)
- [se3_diffusion](https://github.com/jasonkyuyim/se3_diffusion)
- [MMDiff](https://github.com/Profluent-Internships/MMDiff)
- [geometric-rna-design](https://github.com/chaitjo/geometric-rna-design)

---

## 📝 Citation

If you find this work helpful, please consider citing:

```bibtex
@inproceedings{mariboflow,
  title={RiboFlow: Conditional De Novo RNA Co-Design via Synergistic Flow Matching},
  author={Ma, Runze and Zhang, Zhongyue and Wang, Zichen and Hua, Chenqing and Rao, Jiahua and Zhou, Zhuomin and Zheng, Shuangjia},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
}
```

---

## License

This project is released under the MIT License.
