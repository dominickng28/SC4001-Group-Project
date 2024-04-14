# SC4001-Neural Network & Deep Learning-Group-Project
Chain-of-Thought (CoT) Distillation: Outperforming Large Language Models with Task-Specific Small Language Models and Less Training Data 

## Authors
Angie Wong Mei Chi, Dominick Ng Jie En, Keith Heng Jinsheng

## Installing Dependencies
Pytorch is required for the code to work. Install it here: [Pytorch Website](https://pytorch.org/)
```bash
pip install -r requirements.txt
```

## Command Usage
Use `--pretrained` to select pretrained T5 models to use. Default: `google/t5-v1_1-small`

Use `--subsample` to select the size of the dataset to use. Default: `1.0`

## Example
- To run Standard Finetuning, with 75% of data on T5 small
```bash
python finetuning.py --subsample 0.75 --pretrained google/t5-v1_1-small
```

- To run Distilling Step-By-Step using the human labeled data, with 75% of data on T5 small
```bash
python distill-step-by-step.py --subsample 0.75 --pretrained google/t5-v1_1-small
```

- To run Standard Distillation, with 100% of data on T5 base
```bash
python distilling.py --subsample 1.0 --pretrained google/t5-v1_1-base
```

- To run Distilling Step-By-Step using the LLM annotated data, with 100% of data on T5 base
```bash
python distill-step-by-step2.py --subsample 1.0 --pretrained google/t5-v1_1-base
```

## Reference
```bash
@article{hsieh2023distilling,
  title={Distilling step-by-step! outperforming larger language models with less training data and smaller model sizes},
  author={Hsieh, Cheng-Yu and Li, Chun-Liang and Yeh, Chih-Kuan and Nakhost, Hootan and Fujii, Yasuhisa and Ratner, Alexander and Krishna, Ranjay and Lee, Chen-Yu and Pfister, Tomas},
  journal={arXiv preprint arXiv:2305.02301},
  year={2023}
}
```
