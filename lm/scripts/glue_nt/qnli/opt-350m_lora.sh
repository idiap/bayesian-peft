#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Haolin Chen <haolin.chen@idiap.ch>
#
# SPDX-License-Identifier: Apache-2.0
#
python text-classification/run_glue_no_trainer.py \
--model_name_or_path facebook/opt-350m \
--task_name qnli \
--max_length 512 \
--per_device_train_batch_size 32 \
--learning_rate 5e-4 \
--num_train_epochs 3 \
--weight_decay 0 \
--seed 2023 \
--output_dir ./output/glue_nt/qnli/opt-350m_lora_r16a32 \
--with_tracking \
--report_to tensorboard \
--fp16 \
--gradient_checkpointing \
--apply_lora \
--lora_alpha 32 \
--lora_r 16