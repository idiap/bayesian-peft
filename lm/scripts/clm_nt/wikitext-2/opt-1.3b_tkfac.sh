#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Haolin Chen <haolin.chen@idiap.ch>
#
# SPDX-License-Identifier: Apache-2.0
#
python language-modeling/run_clm_no_trainer.py \
--model_name_or_path facebook/opt-1.3b \
--dataset_name wikitext \
--dataset_config_name wikitext-2-v1 \
--per_device_train_batch_size 8 \
--learning_rate 2e-4 \
--num_train_epochs 3 \
--weight_decay 0 \
--seed 2023 \
--output_dir ./output/clm_nt/wikitext-2/opt-1.3b_lora_r16a32_tkfac_1e5 \
--with_tracking \
--report_to tensorboard \
--gradient_checkpointing \
--low_cpu_mem_usage \
--fp16 \
--apply_lora \
--lora_alpha 32 \
--lora_r 16 \
--lora_path ./output/opt_pile/20000/opt-1.3b_hessian_tkfac \
--hessian_method tkfac \
--hessian_lambda 1e5