# Bayesian Parameter-Efficient Fine-Tuning for Overcoming Catastrophic Forgetting

This repository contains the source code for the following [paper](https://arxiv.org/abs/2402.12220) by Haolin Chen and Philip N. Garner:

```
@misc{chen2024bayesian,
      title={Bayesian Parameter-Efficient Fine-Tuning for Overcoming Catastrophic Forgetting}, 
      author={Haolin Chen and Philip N. Garner},
      year={2024},
      eprint={2402.12220},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```

It comprises three components:

1. peft: a customized Python package based on Hugging Face [PEFT](https://github.com/huggingface/peft/tree/v0.6.0) version 0.6.0. It includes the implementation of the Bayesian transfer learning techniques with LoRA and supports the language modeling experiments.
2. lm: codes and scripts for language modeling experiments adapted from the Hugging Face [Transformers](https://github.com/huggingface/transformers/tree/v4.34.0/examples/pytorch) version 4.34.0. This is dependent on the customized peft package.
3. tts: codes and scripts for speech synthesis experiments based on the official implementation of [StyleTTS 2](https://github.com/yl4579/StyleTTS2).

Please refer to the README.md in each directory for instructions.
