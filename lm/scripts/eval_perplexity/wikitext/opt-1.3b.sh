#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Haolin Chen <haolin.chen@idiap.ch>
#
# SPDX-License-Identifier: Apache-2.0
#
MODEL_PATH=
python language-modeling/run_clm.py \
--model_name_or_path facebook/opt-1.3b \
--dataset_name wikitext \
--dataset_config_name wikitext-2-v1 \
--per_device_eval_batch_size 8 \
--do_eval \
--seed 2023 \
--output_dir $MODEL_PATH/eval_wikitext \
--overwrite_output_dir \
--report_to tensorboard \
--low_cpu_mem_usage \
--apply_lora \
--lora_alpha 32 \
--lora_r 16 \
--lora_path $MODEL_PATH
