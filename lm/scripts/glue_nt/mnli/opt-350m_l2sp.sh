#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Haolin Chen <haolin.chen@idiap.ch>
#
# SPDX-License-Identifier: Apache-2.0
#
python text-classification/run_glue_no_trainer.py \
--model_name_or_path facebook/opt-350m \
--task_name mnli \
--max_length 256 \
--per_device_train_batch_size 32 \
--learning_rate 5e-4 \
--num_train_epochs 3 \
--weight_decay 0 \
--seed 2023 \
--output_dir ./output/glue_nt/mnli/opt-350m_lora_r16a32_l2sp_1e-3 \
--with_tracking \
--report_to tensorboard \
--fp16 \
--apply_lora \
--lora_alpha 32 \
--lora_r 16 \
--hessian_method l2sp \
--hessian_lambda 1e-3