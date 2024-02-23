#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Haolin Chen <haolin.chen@idiap.ch>
#
# SPDX-License-Identifier: Apache-2.0
#
python language-modeling/run_clm_no_trainer.py \
--model_name_or_path facebook/opt-350m \
--train_file data/opt_pile_20000_train.json \
--validation_file data/opt_pile_2000_test.json \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--learning_rate 1e-5 \
--num_train_epochs 1 \
--seed 2023 \
--output_dir ./output/opt_pile/20000/opt-350m_hessian_kfac \
--with_tracking \
--report_to tensorboard \
--gradient_checkpointing \
--low_cpu_mem_usage \
--apply_lora \
--lora_alpha 32 \
--lora_r 16 \
--compute_hessian \
--hessian_method kfac