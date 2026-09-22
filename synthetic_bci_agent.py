#!/usr/bin/env python3
"""
🧠 SYNTHETIC BCI AGENT (100% GPU, REAR-TOP VIEW CARTESIAN ALIGNED)
Строгая физика:
+fy = ВПЕРЕД / ВВЕРХ (Frontal / Anterior)
-fy = НАЗАД / ВНИЗ (Occipital / Posterior)
+fx = ВПРАВО (Right Hemisphere)
-fx = ВЛЕВО (Left Hemisphere)
"""

import time
import math
import numpy as np
import multiprocessing as mp
from pylsl import StreamInfo, StreamOutlet
import torch

from neuro_heterarchy_core import DEVICE, COORDS_X, COORDS_Y

class SyntheticAgentProcess(mp.Process):
    def __init__(self, shm):
        super().__init__()
        self.daemon = True
        self.shm = shm

    def run(self):
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Координаты электродов: +Y - лоб, -Y - затылок, +X - правая сторона, -X - левая
        cx_gpu = torch.tensor(COORDS_X, device=DEVICE, dtype=torch.float32).view(1, 16)
        cy_gpu = torch.tensor(COORDS_Y, device=DEVICE, dtype=torch.float32).view(1, 16)

        info = StreamInfo('FreeEEG_Synthetic', 'EEG', 16, 250, 'float32', 'synth_001')
        outlet = StreamOutlet(info)
        print(f"🚀 [SYNTHETIC AGENT] Rear-Top Projection Active on {DEVICE}. 100% GPU Math.")
        
        start_time = time.time()
        
        while self.shm['is_running'].value:
            # Декартовы координаты: fx > 0 вправо, fy > 0 вперед/вверх
            fx = self.shm['target_fx'].value
            fy = self.shm['target_fy'].value
            t_now = time.time() - start_time
            
            with torch.no_grad():
                # Вектор времени 10 сэмплов (40 мс)
                t_vec = torch.arange(10, device=DEVICE, dtype=torch.float32).view(10, 1) * 0.004 + t_now
                
                # Тета несущая 6 Hz
                theta_phase = 2.0 * math.pi * 6.0 * t_vec
                theta_wave = torch.sin(theta_phase)
                
                # Фаза теты [0..1]
                theta_norm = (theta_phase % (2.0 * math.pi)) / (2.0 * math.pi)
                
                # Огибающая гаммы: всплеск на пике теты
                gamma_env = torch.exp(-((theta_phase % (2.0 * math.pi) - math.pi)**2) / 0.4)
                
                # Физический градиент бегущей волны (компенсирует внутренний знак I-J в ядре)
                spatial_gradient = -(cx_gpu * fx + cy_gpu * fy) * 0.35 # [1, 16]
                phase_shift = theta_norm * spatial_gradient # [10, 16]
                
                gamma_wave = torch.sin(2.0 * math.pi * 90.0 * t_vec + phase_shift)
                
                # Детерминированный сигнал без рандомного шума
                eeg_chunk_gpu = theta_wave + (gamma_env * gamma_wave * 2.0)

            outlet.push_chunk(eeg_chunk_gpu.cpu().tolist())
            time.sleep(0.04)

class SyntheticBCIAgent:
    def __init__(self):
        ctx = mp.get_context('spawn')
        self.shm = {'is_running': ctx.Value('b', True), 'target_fx': ctx.Value('d', 0.0), 'target_fy': ctx.Value('d', 0.0)}
        self.process = SyntheticAgentProcess(self.shm)

    def start(self): self.process.start()
    def update_target(self, fx: float, fy: float):
        self.shm['target_fx'].value = float(fx)
        self.shm['target_fy'].value = float(fy)
    def stop(self):
        self.shm['is_running'].value = False
        self.process.join(timeout=2.0)
