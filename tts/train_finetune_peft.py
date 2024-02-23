#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Haolin Chen <haolin.chen@idiap.ch>
#
# SPDX-License-Identifier: MIT
#
# load packages
import random
import yaml
import time
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
import click
import shutil
import warnings
warnings.simplefilter('ignore')
from torch.utils.tensorboard import SummaryWriter

from meldataset import build_dataloader

from Utils.ASR.models import ASRCNN
from Utils.JDC.model import JDCNet
from Utils.PLBERT.util import load_plbert

from models import *
from losses import *
from utils import *

from Modules.slmadv import SLMAdversarialLoss
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

from optimizers import build_optimizer

# simple fix for dataparallel that allows access to class attributes
class MyDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
        
import logging
from logging import StreamHandler
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)


@click.command()
@click.option('-p', '--config_path', default='Configs/config_ft.yml', type=str)
@click.option('-s', '--seed', default=1, type=int)
@click.option('--log_dir', default='Models/finetune_peft', type=str)
@click.option('--lora_r', default=16, type=int)
@click.option('--lora_alpha', default=32, type=int)
@click.option('--hessian_method', default='none', type=str)
@click.option('--hessian_lambda', default=1e4, type=float)
@click.option('--compute_hessian', default=False, is_flag=True)
@click.option('--lora_path', default='', type=str)
def main(config_path, seed, log_dir, lora_r, lora_alpha, hessian_method, hessian_lambda, compute_hessian, lora_path):
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    np.random.seed(seed)

    config = yaml.safe_load(open(config_path))
    
    # log_dir = config['log_dir']
    if not osp.exists(log_dir): os.makedirs(log_dir, exist_ok=True)
    shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))
    writer = SummaryWriter(log_dir + "/tensorboard")

    # write logs
    file_handler = logging.FileHandler(osp.join(log_dir, 'train.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))
    logger.addHandler(file_handler)

    
    batch_size = config.get('batch_size', 10)

    epochs = config.get('epochs', 200)
    save_freq = config.get('save_freq', 2)
    log_interval = config.get('log_interval', 10)
    saving_epoch = config.get('save_freq', 2)

    data_params = config.get('data_params', None)
    sr = config['preprocess_params'].get('sr', 24000)
    train_path = data_params['train_data']
    val_path = data_params['val_data']
    root_path = data_params['root_path']
    min_length = data_params['min_length']
    OOD_data = data_params['OOD_data']

    max_len = config.get('max_len', 200)
    
    loss_params = Munch(config['loss_params'])
    diff_epoch = loss_params.diff_epoch
    joint_epoch = loss_params.joint_epoch
    
    optimizer_params = Munch(config['optimizer_params'])
    
    train_list, val_list = get_data_path_list(train_path, val_path)
    device = 'cuda'

    train_dataloader = build_dataloader(train_list,
                                        root_path,
                                        OOD_data=OOD_data,
                                        min_length=min_length,
                                        batch_size=batch_size,
                                        num_workers=2,
                                        dataset_config={"sr": 22500},
                                        device=device)

    val_dataloader = build_dataloader(val_list,
                                      root_path,
                                      OOD_data=OOD_data,
                                      min_length=min_length,
                                      batch_size=batch_size,
                                      validation=True,
                                      num_workers=0,
                                      device=device,
                                      dataset_config={"sr": 22500})
    
    # load pretrained ASR model
    ASR_config = config.get('ASR_config', False)
    ASR_path = config.get('ASR_path', False)
    text_aligner = load_ASR_models(ASR_path, ASR_config)
    
    # load pretrained F0 model
    F0_path = config.get('F0_path', False)
    pitch_extractor = load_F0_models(F0_path)
    
    # load PL-BERT model
    BERT_path = config.get('PLBERT_dir', False)
    plbert = load_plbert(BERT_path)
    
    # build model
    model_params = recursive_munch(config['model_params'])
    multispeaker = model_params.multispeaker
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    _ = [model[key].to(device) for key in model]
    
    # DP
    for key in model:
        if key != "mpd" and key != "msd" and key != "wd":
            model[key] = MyDataParallel(model[key])
            
    start_epoch = 0
    iters = 0

    load_pretrained = config.get('pretrained_model', '') != '' and config.get('second_stage_load_pretrained', False)
    
    if not load_pretrained:
        if config.get('first_stage_path', '') != '':
            first_stage_path = osp.join(log_dir, config.get('first_stage_path', 'first_stage.pth'))
            print('Loading the first stage model at %s ...' % first_stage_path)
            model, _, start_epoch, iters = load_checkpoint(model, 
                None, 
                first_stage_path,
                load_only_params=True,
                ignore_modules=['bert', 'bert_encoder', 'predictor', 'predictor_encoder', 'msd', 'mpd', 'wd', 'diffusion']) # keep starting epoch for tensorboard log

            # these epochs should be counted from the start epoch
            diff_epoch += start_epoch
            joint_epoch += start_epoch
            epochs += start_epoch
            
            model.predictor_encoder = copy.deepcopy(model.style_encoder)
        else:
            raise ValueError('You need to specify the path to the first stage model.') 

    gl = GeneratorLoss(model.mpd, model.msd).to(device)
    dl = DiscriminatorLoss(model.mpd, model.msd).to(device)
    wl = WavLMLoss(model_params.slm.model, 
                   model.wd, 
                   sr, 
                   model_params.slm.sr).to(device)

    gl = MyDataParallel(gl)
    dl = MyDataParallel(dl)
    wl = MyDataParallel(wl)
    
    sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
        clamp=False
    )
    
    scheduler_params = {
        "max_lr": optimizer_params.lr,
        "pct_start": float(0),
        "epochs": epochs,
        "steps_per_epoch": len(train_dataloader),
    }
    scheduler_params_dict= {key: scheduler_params.copy() for key in model}
    scheduler_params_dict['bert']['max_lr'] = optimizer_params.bert_lr * 2
    scheduler_params_dict['decoder']['max_lr'] = optimizer_params.ft_lr * 2
    scheduler_params_dict['style_encoder']['max_lr'] = optimizer_params.ft_lr * 2


    # PEFT: Apply LoRA to Linear layers
    from peft import LoraConfig, inject_adapter_in_model
    from peft.tuners.lora.layer import LoraLayer

    ekfac_init = False

    # Set requires_grad=False except full finetune models
    # mpd, msd, wd are discriminator params, text_aligner is optimized during training but not used during inference
    full_finetune_model_names = ['mpd', 'msd', 'wd', 'text_aligner']
    for model_name, m in model.items():
        if model_name not in full_finetune_model_names:
            m.requires_grad_(False)

    # Inject LoRA adapters, excluding static models
    # pitch_extractor is a static model (not optimized), text_encoder does not contain any linear module
    static_model_names = ['pitch_extractor', 'text_encoder']
    for model_name, m in model.items():
        if model_name in static_model_names + full_finetune_model_names:
            continue
        target_modules = []
        for k, v in m.named_modules():
            if isinstance(v, torch.nn.modules.linear.Linear):
                target_modules.append(k)
        if target_modules:
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
            )
            model[model_name] = inject_adapter_in_model(lora_config, m)

    # Record LoRA modules
    modules_to_adapt = {}
    for model_name, m in model.items():
        for k, v in m.named_modules():
            if isinstance(v, LoraLayer):
                modules_to_adapt[f'{model_name}:{k}'] = v

    # Init Hessian buffer
    if hessian_method != 'none':
        for k, v in modules_to_adapt.items():
            dim_o, dim_i = (v.weight.shape[0], v.weight.shape[1]) if not v.fan_in_fan_out else (v.weight.shape[1], v.weight.shape[0])
            # Set dtype=torch.float32 following PEFT's LoRA dtype to ensure loading correctly in 8bit mode
            if hessian_method in ['ewc', 'rewc', 'aewc']:
                # EWC: Overcoming Catastrophic Forgetting in Neural Networks
                # REWC, AEWC: Mitigating the Diminishing Effect of Elastic Weight Consolidation
                v.register_buffer('lora_hessian_ewc', torch.zeros(v.weight.shape, dtype=torch.float32, device=v.weight.device))
            elif hessian_method in ['kfac', 'ekfac']:
                # KFAC: Online Structured Laplace Approximations For Overcoming Catastrophic Forgetting
                # EKFAC: Fast Approximate Natural Gradient Descent in a Kronecker-factored Eigenbasis
                v.register_buffer('lora_hessian_kfac_G', torch.zeros((dim_o, dim_o), dtype=torch.float32, device=v.weight.device))
                v.register_buffer('lora_hessian_kfac_A', torch.zeros((dim_i, dim_i), dtype=torch.float32, device=v.weight.device))
            elif hessian_method == 'tkfac':
                # TKFAC: A Trace-restricted Kronecker-Factored Approximation to Natural Gradient
                v.register_buffer('lora_hessian_tkfac_Psi', torch.zeros((dim_o, dim_o), dtype=torch.float32, device=v.weight.device))
                v.register_buffer('lora_hessian_tkfac_Phi', torch.zeros((dim_i, dim_i), dtype=torch.float32, device=v.weight.device))
                v.register_buffer('lora_hessian_tkfac_delta', torch.zeros(1, dtype=torch.float32, device=v.weight.device))

    # prepare_compute_hessian
    if compute_hessian:
        backward_hook, forward_hook = None, None
        backward_hook_handles, forward_hook_handles = {}, {}
        # disable_adapter_layers
        for module_name, module in modules_to_adapt.items():
            module.enable_adapters(False)

        ewc_ver = 'v2'
        if hessian_method in ['ewc', 'rewc', 'aewc']:
            assert ewc_ver in ['v1', 'v2']
            if ewc_ver == 'v1':
                def backward_hook_ewc(module, grad_input, grad_output):
                    module.lora_hessian_ewc += module.weight.grad.detach() ** 2
                
                def backward_hook_aewc(module, grad_input, grad_output):
                    module.lora_hessian_ewc += torch.abs(module.weight.grad.detach())

                if hessian_method in ['ewc', 'rewc']:
                    backward_hook = backward_hook_ewc
                elif hessian_method == 'aewc':
                    backward_hook = backward_hook_aewc
            elif ewc_ver == 'v2':
                def backward_hook_ewc(module, grad_input, grad_output):
                    gy = grad_output[0].detach()
                    x = module.lora_temp_ewc_x
                    grad = gy.unsqueeze(-1) @ x.unsqueeze(-2)
                    # cope with different input dims
                    assert len(x.shape) in [2, 3], len(x.shape)
                    if len(x.shape) == 2:
                        grad = torch.sum(grad, (0))
                    elif len(x.shape) == 3:
                        grad = torch.sum(grad, (0, 1))
                    grad = grad.T if module.fan_in_fan_out else grad
                    assert grad.shape == module.weight.shape
                    module.lora_hessian_ewc += grad ** 2

                def backward_hook_aewc(module, grad_input, grad_output):
                    gy = grad_output[0].detach()
                    x = module.lora_temp_ewc_x
                    grad = gy.unsqueeze(-1) @ x.unsqueeze(-2)
                    # cope with different input dims
                    assert len(x.shape) in [2, 3], len(x.shape)
                    if len(x.shape) == 2:
                        grad = torch.sum(grad, (0))
                    elif len(x.shape) == 3:
                        grad = torch.sum(grad, (0, 1))
                    grad = grad.T if module.fan_in_fan_out else grad
                    module.lora_hessian_ewc += torch.abs(grad)

                def forward_hook_ewc(module, args, output):
                    x = args[0].detach()
                    if not hasattr(module, 'lora_temp_ewc_x'):
                        module.register_buffer('lora_temp_ewc_x', x, persistent=False)
                    else:
                        module.lora_temp_ewc_x = x

                if hessian_method in ['ewc', 'rewc']:
                    backward_hook = backward_hook_ewc
                elif hessian_method == 'aewc':
                    backward_hook = backward_hook_aewc
                forward_hook = forward_hook_ewc
        elif hessian_method in ['kfac', 'ekfac']:
            def backward_hook_kfac(module, grad_input, grad_output):
                g = grad_output[0].detach()
                G = g[..., None] @ g[..., None, :]
                # cope with different input dims
                assert len(g.shape) in [2, 3], len(g.shape)
                if len(g.shape) == 2:
                    G = torch.mean(G, (0))
                elif len(g.shape) == 3:
                    G = torch.mean(G, (0, 1))
                module.lora_hessian_kfac_G += G

            def forward_hook_kfac(module, args, output):
                a = args[0].detach()
                A = a[..., None] @ a[..., None, :]
                # cope with different input dims
                assert len(a.shape) in [2, 3], len(a.shape)
                if len(a.shape) == 2:
                    A = torch.mean(A, (0))
                elif len(a.shape) == 3:
                    A = torch.mean(A, (0, 1))
                module.lora_hessian_kfac_A += A

            backward_hook = backward_hook_kfac
            forward_hook = forward_hook_kfac
        elif hessian_method == 'tkfac':
            def backward_hook_tkfac(module, grad_input, grad_output):
                g = grad_output[0].detach()
                G = g[..., None] @ g[..., None, :]
                # cope with different input dims
                assert len(g.shape) in [2, 3], len(g.shape)
                if len(g.shape) == 2:
                    G = torch.mean(G, (0))
                elif len(g.shape) == 3:
                    G = torch.mean(G, (0, 1))
                A = module.lora_temp_tkfac_A
                trace_G = torch.trace(G)
                trace_A = torch.trace(A)
                module.lora_hessian_tkfac_Psi += trace_A * G
                module.lora_hessian_tkfac_Phi += trace_G * A
                module.lora_hessian_tkfac_delta += trace_A * trace_G

            def forward_hook_tkfac(module, args, output):
                a = args[0].detach()
                A = a[..., None] @ a[..., None, :]
                # cope with different input dims
                assert len(a.shape) in [2, 3], len(a.shape)
                if len(a.shape) == 2:
                    A = torch.mean(A, (0))
                elif len(a.shape) == 3:
                    A = torch.mean(A, (0, 1))
                if not hasattr(module, 'lora_temp_tkfac_A'):
                    module.register_buffer('lora_temp_tkfac_A', A, persistent=False)
                else:
                    module.lora_temp_tkfac_A = A

            backward_hook = backward_hook_tkfac
            forward_hook = forward_hook_tkfac

        for k, v in modules_to_adapt.items():
            v.weight.requires_grad_(True)
            if backward_hook:
                backward_hook_handles[k] = v.register_full_backward_hook(backward_hook)
            if forward_hook:
                forward_hook_handles[k] = v.register_forward_hook(forward_hook)

    # Load Hessian
    if not compute_hessian and lora_path:
        if os.path.exists(lora_path):
            model_state_dict = torch.load(lora_path, map_location=torch.device(device))
            for model_name, state_dict in model_state_dict.items():
                model[model_name].load_state_dict(state_dict, strict=False)

    # Print active parameters
    print('******************************* Parameter Stats *******************************')
    total_params, trainable_params = 0, 0
    for model_name, m in model.items():
        if model_name in static_model_names + full_finetune_model_names:
            continue
        m_total_params, m_trainable_params = 0, 0
        for k, v in m.named_parameters():
            total_params += v.numel()
            m_total_params += v.numel()
            if v.requires_grad:
                trainable_params += v.numel()
                m_trainable_params += v.numel()
        print(
            f"{model_name} || trainable params: {m_trainable_params:,d} || all params: {m_total_params:,d} || trainable%: {100 * m_trainable_params / m_total_params}"
        )
    print(
        f"total || trainable params: {trainable_params:,d} || all params: {total_params:,d} || trainable%: {100 * trainable_params / total_params}"
    )


    optimizer = build_optimizer({key: model[key].parameters() for key in model},
                                          scheduler_params_dict=scheduler_params_dict, lr=optimizer_params.lr)
    
    # adjust BERT learning rate
    for g in optimizer.optimizers['bert'].param_groups:
        g['betas'] = (0.9, 0.99)
        g['lr'] = optimizer_params.bert_lr
        g['initial_lr'] = optimizer_params.bert_lr
        g['min_lr'] = 0
        g['weight_decay'] = 0.01
        
    # adjust acoustic module learning rate
    for module in ["decoder", "style_encoder"]:
        for g in optimizer.optimizers[module].param_groups:
            g['betas'] = (0.0, 0.99)
            g['lr'] = optimizer_params.ft_lr
            g['initial_lr'] = optimizer_params.ft_lr
            g['min_lr'] = 0
            g['weight_decay'] = 1e-4
        
    # load models if there is a model
    if load_pretrained:
        model, optimizer, start_epoch, iters = load_checkpoint(model,  optimizer, config['pretrained_model'],
                                    load_only_params=config.get('load_only_params', True))
        
    n_down = model.text_aligner.n_down

    best_loss = float('inf')  # best test loss
    loss_train_record = list([])
    loss_test_record = list([])
    iters = 0
    
    criterion = nn.L1Loss() # F0 loss (regression)
    torch.cuda.empty_cache()
    
    stft_loss = MultiResolutionSTFTLoss().to(device)
    
    print('BERT', optimizer.optimizers['bert'])
    print('decoder', optimizer.optimizers['decoder'])

    start_ds = False
    
    running_std = []
    
    slmadv_params = Munch(config['slmadv_params'])
    slmadv = SLMAdversarialLoss(model, wl, sampler, 
                                slmadv_params.min_len, 
                                slmadv_params.max_len,
                                batch_percentage=slmadv_params.batch_percentage,
                                skip_update=slmadv_params.iter, 
                                sig=slmadv_params.sig
                               )
    
    for epoch in range(start_epoch, epochs):
        running_loss = 0
        start_time = time.time()

        _ = [model[key].eval() for key in model]
        
        model.text_aligner.train()
        model.text_encoder.train()
        
        model.predictor.train()
        model.bert_encoder.train()
        model.bert.train()
        model.msd.train()
        model.mpd.train()

        for i, batch in enumerate(train_dataloader):
            waves = batch[0]
            batch = [b.to(device) for b in batch[1:]]
            texts, input_lengths, ref_texts, ref_lengths, mels, mel_input_length, ref_mels = batch
            with torch.no_grad():
                mask = length_to_mask(mel_input_length // (2 ** n_down)).to(device)
                mel_mask = length_to_mask(mel_input_length).to(device)
                text_mask = length_to_mask(input_lengths).to(texts.device)

                # compute reference styles
                if multispeaker and epoch >= diff_epoch:
                    ref_ss = model.style_encoder(ref_mels.unsqueeze(1))
                    ref_sp = model.predictor_encoder(ref_mels.unsqueeze(1))
                    ref = torch.cat([ref_ss, ref_sp], dim=1)
                
            try:
                ppgs, s2s_pred, s2s_attn = model.text_aligner(mels, mask, texts)
                s2s_attn = s2s_attn.transpose(-1, -2)
                s2s_attn = s2s_attn[..., 1:]
                s2s_attn = s2s_attn.transpose(-1, -2)
            except:
                continue

            mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** n_down))
            s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

            # encode
            t_en = model.text_encoder(texts, input_lengths, text_mask)
            
            # 50% of chance of using monotonic version
            if bool(random.getrandbits(1)):
                asr = (t_en @ s2s_attn)
            else:
                asr = (t_en @ s2s_attn_mono)

            d_gt = s2s_attn_mono.sum(axis=-1).detach()

            # compute the style of the entire utterance
            # this operation cannot be done in batch because of the avgpool layer (may need to work on masked avgpool)
            ss = []
            gs = []
            for bib in range(len(mel_input_length)):
                mel_length = int(mel_input_length[bib].item())
                mel = mels[bib, :, :mel_input_length[bib]]
                s = model.predictor_encoder(mel.unsqueeze(0).unsqueeze(1))
                ss.append(s)
                s = model.style_encoder(mel.unsqueeze(0).unsqueeze(1))
                gs.append(s)

            # s_dur = torch.stack(ss).squeeze()  # global prosodic styles
            # gs = torch.stack(gs).squeeze() # global acoustic styles
            s_dur = torch.stack(ss).squeeze(1)  # global prosodic styles
            gs = torch.stack(gs).squeeze(1) # global acoustic styles
            s_trg = torch.cat([gs, s_dur], dim=-1).detach() # ground truth for denoiser

            bert_dur = model.bert(texts, attention_mask=(~text_mask).int())
            d_en = model.bert_encoder(bert_dur).transpose(-1, -2) 
            
            # denoiser training
            if epoch >= diff_epoch:
                num_steps = np.random.randint(3, 5)
                
                if model_params.diffusion.dist.estimate_sigma_data:
                    model.diffusion.module.diffusion.sigma_data = s_trg.std(axis=-1).mean().item() # batch-wise std estimation
                    running_std.append(model.diffusion.module.diffusion.sigma_data)
                    
                if multispeaker:
                    s_preds = sampler(noise = torch.randn_like(s_trg).unsqueeze(1).to(device), 
                          embedding=bert_dur,
                          embedding_scale=1,
                                   features=ref, # reference from the same speaker as the embedding
                             embedding_mask_proba=0.1,
                             num_steps=num_steps).squeeze(1)
                    loss_diff = model.diffusion(s_trg.unsqueeze(1), embedding=bert_dur, features=ref).mean() # EDM loss
                    loss_sty = F.l1_loss(s_preds, s_trg.detach()) # style reconstruction loss
                else:
                    s_preds = sampler(noise = torch.randn_like(s_trg).unsqueeze(1).to(device), 
                          embedding=bert_dur,
                          embedding_scale=1,
                             embedding_mask_proba=0.1,
                             num_steps=num_steps).squeeze(1)                    
                    loss_diff = model.diffusion.module.diffusion(s_trg.unsqueeze(1), embedding=bert_dur).mean() # EDM loss
                    loss_sty = F.l1_loss(s_preds, s_trg.detach()) # style reconstruction loss
            else:
                loss_sty = 0
                loss_diff = 0

                
            s_loss = 0
            

            d, p = model.predictor(d_en, s_dur, 
                                                    input_lengths, 
                                                    s2s_attn_mono, 
                                                    text_mask)
                
            mel_len_st = int(mel_input_length.min().item() / 2 - 1)
            mel_len = min(int(mel_input_length.min().item() / 2 - 1), max_len // 2)
            en = []
            gt = []
            p_en = []
            wav = []
            st = []
            
            for bib in range(len(mel_input_length)):
                mel_length = int(mel_input_length[bib].item() / 2)

                random_start = np.random.randint(0, mel_length - mel_len)
                en.append(asr[bib, :, random_start:random_start+mel_len])
                p_en.append(p[bib, :, random_start:random_start+mel_len])
                gt.append(mels[bib, :, (random_start * 2):((random_start+mel_len) * 2)])
                
                y = waves[bib][(random_start * 2) * 300:((random_start+mel_len) * 2) * 300]
                wav.append(torch.from_numpy(y).to(device))
                
                # style reference (better to be different from the GT)
                random_start = np.random.randint(0, mel_length - mel_len_st)
                st.append(mels[bib, :, (random_start * 2):((random_start+mel_len_st) * 2)])
                
            wav = torch.stack(wav).float().detach()

            en = torch.stack(en)
            p_en = torch.stack(p_en)
            gt = torch.stack(gt).detach()
            st = torch.stack(st).detach()
            
            
            if gt.size(-1) < 80:
                continue
            
            s = model.style_encoder(gt.unsqueeze(1))           
            s_dur = model.predictor_encoder(gt.unsqueeze(1))
                
            with torch.no_grad():
                F0_real, _, F0 = model.pitch_extractor(gt.unsqueeze(1))
                if len(F0_real.shape) == 1:
                    F0_real = F0_real.unsqueeze(0)
                F0 = F0.reshape(F0.shape[0], F0.shape[1] * 2, F0.shape[2], 1).squeeze()

                N_real = log_norm(gt.unsqueeze(1)).squeeze(1)
                
                y_rec_gt = wav.unsqueeze(1)
                y_rec_gt_pred = model.decoder(en, F0_real, N_real, s)

                wav = y_rec_gt

            F0_fake, N_fake = model.predictor.F0Ntrain(p_en, s_dur)

            y_rec = model.decoder(en, F0_fake, N_fake, s)

            loss_F0_rec =  (F.smooth_l1_loss(F0_real, F0_fake)) / 10
            loss_norm_rec = F.smooth_l1_loss(N_real, N_fake)

            optimizer.zero_grad()
            d_loss = dl(wav.detach(), y_rec.detach()).mean()
            d_loss.backward()
            if not compute_hessian:
                optimizer.step('msd')
                optimizer.step('mpd')

            # generator loss
            optimizer.zero_grad()

            loss_mel = stft_loss(y_rec, wav)
            loss_gen_all = gl(wav, y_rec).mean()
            # loss_lm = wl(wav.detach().squeeze(), y_rec.squeeze()).mean()
            loss_lm = wl(wav.detach().squeeze(1), y_rec.squeeze(1)).mean()

            loss_ce = 0
            loss_dur = 0
            for _s2s_pred, _text_input, _text_length in zip(d, (d_gt), input_lengths):
                _s2s_pred = _s2s_pred[:_text_length, :]
                _text_input = _text_input[:_text_length].long()
                _s2s_trg = torch.zeros_like(_s2s_pred)
                for p in range(_s2s_trg.shape[0]):
                    _s2s_trg[p, :_text_input[p]] = 1
                _dur_pred = torch.sigmoid(_s2s_pred).sum(axis=1)

                loss_dur += F.l1_loss(_dur_pred[1:_text_length-1], 
                                       _text_input[1:_text_length-1])
                loss_ce += F.binary_cross_entropy_with_logits(_s2s_pred.flatten(), _s2s_trg.flatten())

            loss_ce /= texts.size(0)
            loss_dur /= texts.size(0)
            
            loss_s2s = 0
            for _s2s_pred, _text_input, _text_length in zip(s2s_pred, texts, input_lengths):
                loss_s2s += F.cross_entropy(_s2s_pred[:_text_length], _text_input[:_text_length])
            loss_s2s /= texts.size(0)

            loss_mono = F.l1_loss(s2s_attn, s2s_attn_mono) * 10

            # Compute Hessian loss
            loss_hessian = 0
            if not compute_hessian and hessian_method not in ['none']:
                if hessian_method == 'l2sp':
                    for k, v in modules_to_adapt.items():
                        loss_hessian += torch.sum(v.get_delta_weight("default") ** 2)
                elif hessian_method in ['ewc', 'rewc']:
                    for k, v in modules_to_adapt.items():
                        loss_hessian += torch.sum(v.lora_hessian_ewc * v.get_delta_weight("default") ** 2)
                    if hessian_method == 'rewc':
                        loss_hessian = (loss_hessian + 1e-8) ** 0.5
                elif hessian_method == 'aewc':
                    for k, v in modules_to_adapt.items():
                        loss_hessian += torch.sum(v.lora_hessian_ewc * torch.abs(v.get_delta_weight("default")))
                elif hessian_method == 'kfac':
                    for k, v in modules_to_adapt.items():
                        delta_weight = v.get_delta_weight("default")
                        delta_weight = delta_weight if not v.fan_in_fan_out else delta_weight.T
                        G = v.lora_hessian_kfac_G
                        A = v.lora_hessian_kfac_A
                        loss_hessian += torch.sum(delta_weight * (G @ delta_weight @ A))
                elif hessian_method == 'ekfac':
                    # compute_ekfac
                    def compute_ekfac():
                        if ekfac_init:
                            return
                        
                        def kronecker(A, B):
                            sA = A.size()
                            sB = B.size()
                            return (A.view(sA[0], 1, sA[1], 1) * B.view(1, sB[0], 1, sB[1])) \
                                .contiguous().view(sA[0] * sB[0], sA[1] * sB[1])
                        
                        for k, v in modules_to_adapt.items():
                            G, A = v.lora_hessian_kfac_G, v.lora_hessian_kfac_A
                            evals_G, evecs_G = torch.linalg.eigh(G)
                            evals_A, evecs_A = torch.linalg.eigh(A)
                            diag = kronecker(evals_G.view(-1, 1), evals_A.view(-1, 1))
                            v.register_buffer('lora_ekfac_evecs_G', evecs_G)
                            v.register_buffer('lora_ekfac_evecs_A', evecs_A)
                            v.register_buffer('lora_ekfac_diag', diag)
                            del v.lora_hessian_kfac_G
                            del v.lora_hessian_kfac_A
                        ekfac_init = True

                    compute_ekfac()
                    for k, v in modules_to_adapt.items():
                        evecs_G, evecs_A = v.lora_ekfac_evecs_G, v.lora_ekfac_evecs_A
                        diag = v.lora_ekfac_diag
                        delta_weight = v.get_delta_weight("default")
                        delta_weight = delta_weight if not v.fan_in_fan_out else delta_weight.T
                        v_kfe = torch.mm(torch.mm(evecs_G.t(), delta_weight), evecs_A)
                        loss_hessian += torch.dot(v_kfe.view(-1)**2, diag.view(-1))
                elif hessian_method == 'tkfac':
                    for k, v in modules_to_adapt.items():
                        delta_weight = v.get_delta_weight("default")
                        delta_weight = delta_weight if not v.fan_in_fan_out else delta_weight.T
                        Psi = v.lora_hessian_tkfac_Psi
                        Phi = v.lora_hessian_tkfac_Phi
                        delta = v.lora_hessian_tkfac_delta.squeeze()
                        loss_hessian += delta * torch.sum(delta_weight * (Psi @ delta_weight @ Phi))

                loss_hessian *= hessian_lambda

            g_loss = loss_params.lambda_mel * loss_mel + \
                     loss_params.lambda_F0 * loss_F0_rec + \
                     loss_params.lambda_ce * loss_ce + \
                     loss_params.lambda_norm * loss_norm_rec + \
                     loss_params.lambda_dur * loss_dur + \
                     loss_params.lambda_gen * loss_gen_all + \
                     loss_params.lambda_slm * loss_lm + \
                     loss_params.lambda_sty * loss_sty + \
                     loss_params.lambda_diff * loss_diff + \
                    loss_params.lambda_mono * loss_mono + \
                    loss_params.lambda_s2s * loss_s2s + \
                    loss_hessian
            
            running_loss += loss_mel.item()
            g_loss.backward()
            if torch.isnan(g_loss):
                from IPython.core.debugger import set_trace
                set_trace()

            if not compute_hessian:
                optimizer.step('bert_encoder')
                optimizer.step('bert')
                optimizer.step('predictor')
                optimizer.step('predictor_encoder')
                optimizer.step('style_encoder')
                optimizer.step('decoder')
            
                # PEFT: no active parameters in text_encoder
                # optimizer.step('text_encoder')
                optimizer.step('text_aligner')
            
            if epoch >= diff_epoch:
                if not compute_hessian:
                    optimizer.step('diffusion')

            d_loss_slm, loss_gen_lm = 0, 0
            if epoch >= joint_epoch:
                # randomly pick whether to use in-distribution text
                if np.random.rand() < 0.5:
                    use_ind = True
                else:
                    use_ind = False

                if use_ind:
                    ref_lengths = input_lengths
                    ref_texts = texts
                    
                slm_out = slmadv(i, 
                                 y_rec_gt, 
                                 y_rec_gt_pred, 
                                 waves, 
                                 mel_input_length,
                                 ref_texts, 
                                 ref_lengths, use_ind, s_trg.detach(), ref if multispeaker else None)

                if slm_out is not None:
                    d_loss_slm, loss_gen_lm, y_pred = slm_out

                    # SLM generator loss
                    optimizer.zero_grad()
                    loss_gen_lm.backward()

                    # compute the gradient norm
                    total_norm = {}
                    for key in model.keys():
                        total_norm[key] = 0
                        parameters = [p for p in model[key].parameters() if p.grad is not None and p.requires_grad]
                        for p in parameters:
                            param_norm = p.grad.detach().data.norm(2)
                            total_norm[key] += param_norm.item() ** 2
                        total_norm[key] = total_norm[key] ** 0.5

                    # gradient scaling
                    if total_norm['predictor'] > slmadv_params.thresh:
                        for key in model.keys():
                            for p in model[key].parameters():
                                if p.grad is not None:
                                    p.grad *= (1 / total_norm['predictor'])

                    for p in model.predictor.duration_proj.parameters():
                        if p.grad is not None:
                            p.grad *= slmadv_params.scale

                    for p in model.predictor.lstm.parameters():
                        if p.grad is not None:
                            p.grad *= slmadv_params.scale

                    for p in model.diffusion.parameters():
                        if p.grad is not None:
                            p.grad *= slmadv_params.scale
                    
                    if not compute_hessian:
                        optimizer.step('bert_encoder')
                        optimizer.step('bert')
                        optimizer.step('predictor')
                        optimizer.step('diffusion')

                    # SLM discriminator loss
                    if d_loss_slm != 0:
                        optimizer.zero_grad()
                        d_loss_slm.backward(retain_graph=True)
                        if not compute_hessian:
                            optimizer.step('wd')

            iters = iters + 1
            
            if (i+1)%log_interval == 0:
                logger.info ('Epoch [%d/%d], Step [%d/%d], Loss: %.5f, Disc Loss: %.5f, Dur Loss: %.5f, CE Loss: %.5f, Norm Loss: %.5f, F0 Loss: %.5f, LM Loss: %.5f, Gen Loss: %.5f, Sty Loss: %.5f, Diff Loss: %.5f, DiscLM Loss: %.5f, GenLM Loss: %.5f, SLoss: %.5f, S2S Loss: %.5f, Mono Loss: %.5f, Hessian Loss: %.5f, Hessian Loss Real: %.5f'
                    %(epoch+1, epochs, i+1, len(train_list)//batch_size, running_loss / log_interval, d_loss, loss_dur, loss_ce, loss_norm_rec, loss_F0_rec, loss_lm, loss_gen_all, loss_sty, loss_diff, d_loss_slm, loss_gen_lm, s_loss, loss_s2s, loss_mono, loss_hessian, loss_hessian / hessian_lambda))
                
                writer.add_scalar('train/mel_loss', running_loss / log_interval, iters)
                writer.add_scalar('train/gen_loss', loss_gen_all, iters)
                writer.add_scalar('train/d_loss', d_loss, iters)
                writer.add_scalar('train/ce_loss', loss_ce, iters)
                writer.add_scalar('train/dur_loss', loss_dur, iters)
                writer.add_scalar('train/slm_loss', loss_lm, iters)
                writer.add_scalar('train/norm_loss', loss_norm_rec, iters)
                writer.add_scalar('train/F0_loss', loss_F0_rec, iters)
                writer.add_scalar('train/sty_loss', loss_sty, iters)
                writer.add_scalar('train/diff_loss', loss_diff, iters)
                writer.add_scalar('train/d_loss_slm', d_loss_slm, iters)
                writer.add_scalar('train/gen_loss_slm', loss_gen_lm, iters)
                writer.add_scalar('train/hessian_loss', loss_hessian, iters)
                writer.add_scalar('train/hessian_loss_real', loss_hessian / hessian_lambda, iters)
                
                running_loss = 0
                
                print('Time elasped:', time.time()-start_time)

        if compute_hessian:
            # finalize_hessian
            # remove hook handles
            for k, handle in backward_hook_handles.items():
                handle.remove()
            for k, handle in forward_hook_handles.items():
                handle.remove()
            # normalize hessian
            for k, v in modules_to_adapt.items():
                if hessian_method in ['ewc', 'rewc', 'aewc']:
                    v.lora_hessian_ewc /= iters
                    if hasattr(v, 'lora_temp_ewc_x'):
                        del v.lora_temp_ewc_x
                elif hessian_method in ['kfac', 'ekfac']:
                    v.lora_hessian_kfac_G /= iters
                    v.lora_hessian_kfac_A /= iters
                elif hessian_method == 'tkfac':
                    v.lora_hessian_tkfac_Psi /= iters
                    v.lora_hessian_tkfac_Phi /= iters
                    v.lora_hessian_tkfac_delta /= iters
                    v.lora_hessian_tkfac_Psi /= v.lora_hessian_tkfac_delta
                    v.lora_hessian_tkfac_Phi /= v.lora_hessian_tkfac_delta
                    del v.lora_temp_tkfac_A

        loss_test = 0
        loss_align = 0
        loss_f = 0
        _ = [model[key].eval() for key in model]

        with torch.no_grad():
            iters_test = 0
            for batch_idx, batch in enumerate(val_dataloader):
                optimizer.zero_grad()

                try:
                    waves = batch[0]
                    batch = [b.to(device) for b in batch[1:]]
                    texts, input_lengths, ref_texts, ref_lengths, mels, mel_input_length, ref_mels = batch
                    with torch.no_grad():
                        mask = length_to_mask(mel_input_length // (2 ** n_down)).to('cuda')
                        text_mask = length_to_mask(input_lengths).to(texts.device)

                        _, _, s2s_attn = model.text_aligner(mels, mask, texts)
                        s2s_attn = s2s_attn.transpose(-1, -2)
                        s2s_attn = s2s_attn[..., 1:]
                        s2s_attn = s2s_attn.transpose(-1, -2)

                        mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** n_down))
                        s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

                        # encode
                        t_en = model.text_encoder(texts, input_lengths, text_mask)
                        asr = (t_en @ s2s_attn_mono)

                        d_gt = s2s_attn_mono.sum(axis=-1).detach()

                    ss = []
                    gs = []

                    for bib in range(len(mel_input_length)):
                        mel_length = int(mel_input_length[bib].item())
                        mel = mels[bib, :, :mel_input_length[bib]]
                        s = model.predictor_encoder(mel.unsqueeze(0).unsqueeze(1))
                        ss.append(s)
                        s = model.style_encoder(mel.unsqueeze(0).unsqueeze(1))
                        gs.append(s)

                    s = torch.stack(ss).squeeze(1)
                    gs = torch.stack(gs).squeeze(1)
                    s_trg = torch.cat([s, gs], dim=-1).detach()

                    bert_dur = model.bert(texts, attention_mask=(~text_mask).int())
                    d_en = model.bert_encoder(bert_dur).transpose(-1, -2) 
                    d, p = model.predictor(d_en, s, 
                                                        input_lengths, 
                                                        s2s_attn_mono, 
                                                        text_mask)
                    # get clips
                    mel_len = int(mel_input_length.min().item() / 2 - 1)
                    en = []
                    gt = []

                    p_en = []
                    wav = []

                    for bib in range(len(mel_input_length)):
                        mel_length = int(mel_input_length[bib].item() / 2)

                        random_start = np.random.randint(0, mel_length - mel_len)
                        en.append(asr[bib, :, random_start:random_start+mel_len])
                        p_en.append(p[bib, :, random_start:random_start+mel_len])

                        gt.append(mels[bib, :, (random_start * 2):((random_start+mel_len) * 2)])
                        y = waves[bib][(random_start * 2) * 300:((random_start+mel_len) * 2) * 300]
                        wav.append(torch.from_numpy(y).to(device))

                    wav = torch.stack(wav).float().detach()

                    en = torch.stack(en)
                    p_en = torch.stack(p_en)
                    gt = torch.stack(gt).detach()
                    s = model.predictor_encoder(gt.unsqueeze(1))

                    F0_fake, N_fake = model.predictor.F0Ntrain(p_en, s)

                    loss_dur = 0
                    for _s2s_pred, _text_input, _text_length in zip(d, (d_gt), input_lengths):
                        _s2s_pred = _s2s_pred[:_text_length, :]
                        _text_input = _text_input[:_text_length].long()
                        _s2s_trg = torch.zeros_like(_s2s_pred)
                        for bib in range(_s2s_trg.shape[0]):
                            _s2s_trg[bib, :_text_input[bib]] = 1
                        _dur_pred = torch.sigmoid(_s2s_pred).sum(axis=1)
                        loss_dur += F.l1_loss(_dur_pred[1:_text_length-1], 
                                               _text_input[1:_text_length-1])

                    loss_dur /= texts.size(0)

                    s = model.style_encoder(gt.unsqueeze(1))

                    y_rec = model.decoder(en, F0_fake, N_fake, s)
                    loss_mel = stft_loss(y_rec.squeeze(), wav.detach())

                    F0_real, _, F0 = model.pitch_extractor(gt.unsqueeze(1)) 

                    loss_F0 = F.l1_loss(F0_real, F0_fake) / 10

                    loss_test += (loss_mel).mean()
                    loss_align += (loss_dur).mean()
                    loss_f += (loss_F0).mean()

                    iters_test += 1
                except:
                    continue

        print('Epochs:', epoch + 1)
        logger.info('Validation loss: %.3f, Dur loss: %.3f, F0 loss: %.3f' % (loss_test / iters_test, loss_align / iters_test, loss_f / iters_test) + '\n\n\n')
        print('\n\n\n')
        writer.add_scalar('eval/mel_loss', loss_test / iters_test, epoch + 1)
        writer.add_scalar('eval/dur_loss', loss_align / iters_test, epoch + 1)
        writer.add_scalar('eval/F0_loss', loss_f / iters_test, epoch + 1)
        
        
        if (epoch + 1) % save_freq == 0 :
            if (loss_test / iters_test) < best_loss:
                best_loss = loss_test / iters_test
            print('Saving..')

            # Save PEFT weights
            from collections import defaultdict
            
            state = defaultdict(dict)
            if compute_hessian:
                for model_name, m in model.items():
                    m_state_dict = m.state_dict()
                    for n, p in m_state_dict.items():
                        if 'hessian' in n:
                            state[model_name][n] = p
            else:
                for model_name, m in model.items():
                    m_state_dict = m.state_dict()
                    for n, p in m_state_dict.items():
                        if 'lora' in n and not 'hessian' in n:
                            state[model_name][n] = p
            
            save_path = osp.join(log_dir, 'epoch_2nd_%05d.pth' % epoch)
            torch.save(state, save_path)

            # if estimate sigma, save the estimated simga
            if model_params.diffusion.dist.estimate_sigma_data:
                config['model_params']['diffusion']['dist']['sigma_data'] = float(np.mean(running_std))

                with open(osp.join(log_dir, osp.basename(config_path)), 'w') as outfile:
                    yaml.dump(config, outfile, default_flow_style=True)

                            
if __name__=="__main__":
    main()
