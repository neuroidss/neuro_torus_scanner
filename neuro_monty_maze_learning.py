#!/usr/bin/env python3
"""
🧠 NEUROCANVAS × VLA-JEPA: EGOCENTRIC ROTATING MAP & 3D ACTIVE INFERENCE
- Эгоцентрическая вращающаяся 2D-карта: Аватар всегда смотрит ВВЕРХ, мир крутится вокруг него.
- Управление в координатах тела: Вперед = туда, куда смотрит аватар; Вбок = стрейф.
- Поворот влево вращает карту вправо (как в Quake / авиагоризонте).
- 32-точечный фазовый кометный шлейф (Lisman & Jensen 2013) ориентирован по телу аватара.
- Высокоскоростной GPU Raycaster (CUDA) питает V-JEPA 3D-видеопотоком.
"""

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import sys
import math
import time
from collections import deque
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import pygame

from neuro_heterarchy_core import (
    HeterarchicalBrainEngine, DEVICE
)
from vla_jepa_wrapper import VLA_JEPA_Wrapper

DIM = 11
CELL_SIZE = 56
MAZE_W = DIM * CELL_SIZE
MAZE_H = DIM * CELL_SIZE

CAM_W, CAM_H = 256, 256 

ELECTRODE_X = np.array([10.14, 7.43, 2.75, 2.72, -2.72, -2.75, -7.42, -10.14,
                        -10.14, -7.43, -2.75, -2.72, 2.72, 2.75, 7.43, 10.14], dtype=np.float32)
ELECTRODE_Y = np.array([-2.72, -7.43, -4.77, -10.15,-10.14, -4.77, -7.42,  -2.73,
                         2.72,  7.43,  4.76, 10.14, 10.15,  4.77,  7.42,   2.71], dtype=np.float32)

# ==============================================================================
# ТОПОЛОГИЧЕСКИЙ ЛАБИРИНТ (2D GRID)
# ==============================================================================
class TopoMaze:
    def __init__(self, dim=DIM):
        self.dim = dim if dim % 2 != 0 else dim + 1
        self.grid = [[1 for _ in range(self.dim)] for _ in range(self.dim)]
        self._gen(1, 1)
        
        self.exit_pos = (self.dim - 2, self.dim - 2)
        self.grid[self.exit_pos[1]][self.exit_pos[0]] = 2
        self.grid[self.exit_pos[1] - 1][self.exit_pos[0]] = 0
        self.grid[self.exit_pos[1]][self.exit_pos[0] - 1] = 0

    def _gen(self, x, y):
        self.grid[y][x] = 0
        dirs = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        np.random.shuffle(dirs)
        for dx, dy in dirs:
            nx, ny = x + dx * 2, y + dy * 2
            if 0 < nx < self.dim - 1 and 0 < ny < self.dim - 1 and self.grid[ny][nx] == 1:
                self.grid[y + dy][x + dx] = 0
                self._gen(nx, ny)

    def is_wall(self, gx: float, gy: float) -> bool:
        ix, iy = int(math.floor(gx)), int(math.floor(gy))
        if ix < 0 or ix >= self.dim or iy < 0 or iy >= self.dim: return True
        return self.grid[iy][ix] == 1

# ==============================================================================
# GPU ТЕКСТУРИРОВАННЫЙ 3D РЕЙКАСТЕР (CUDA)
# ==============================================================================
class GPURaycaster(nn.Module):
    def __init__(self, dim=DIM, cam_w=CAM_W, cam_h=CAM_H, device=DEVICE):
        super().__init__()
        self.dim, self.w, self.h, self.device = dim, cam_w, cam_h, device
        self.fov = math.pi / 3.0
        self.d_steps = torch.linspace(0.04, 11.0, 180, device=device)
        self.ray_offsets = torch.linspace(-self.fov / 2.0, self.fov / 2.0, self.w, device=device)
        self.cos_offsets = torch.cos(self.ray_offsets)
        self.y_coords = torch.arange(self.h, device=device)[:, None]
        
        self.bg = torch.zeros((self.h, self.w, 3), dtype=torch.uint8, device=device)
        self.bg[:self.h // 2, :, :] = 35
        self.bg[self.h // 2:, :, :] = 55

    def render(self, grid_list: list[list[int]], x: float, y: float, angle: float) -> np.ndarray:
        with torch.no_grad():
            grid_gpu = torch.tensor(grid_list, dtype=torch.uint8, device=self.device)
            angles = angle + self.ray_offsets
            dx, dy = torch.cos(angles), torch.sin(angles)
            
            sample_x = (x + dx[:, None] * self.d_steps[None, :]).clamp(0, self.dim - 1.001)
            sample_y = (y + dy[:, None] * self.d_steps[None, :]).clamp(0, self.dim - 1.001)
            
            ix, iy = sample_x.long(), sample_y.long()
            cell_types = grid_gpu[iy, ix]
            hits = (cell_types > 0)
            has_hit = hits.any(dim=1)
            
            first_hit_idx = torch.argmax(hits.float(), dim=1)
            dist = torch.where(has_hit, self.d_steps[first_hit_idx], torch.tensor(11.0, device=self.device))
            dist_corr = dist * self.cos_offsets
            
            hit_x = torch.where(has_hit, sample_x[torch.arange(self.w, device=self.device), first_hit_idx], torch.zeros(self.w, device=self.device))
            hit_y = torch.where(has_hit, sample_y[torch.arange(self.w, device=self.device), first_hit_idx], torch.zeros(self.w, device=self.device))
            tex_u = ((hit_x + hit_y) * 3.0) % 1.0
            
            hit_types = torch.where(has_hit, cell_types[torch.arange(self.w, device=self.device), first_hit_idx], torch.zeros(self.w, dtype=torch.uint8, device=self.device))
            
            wall_heights = (self.h / (dist_corr + 1e-4)).long().clamp(0, self.h)
            wall_top = ((self.h - wall_heights) // 2).clamp(0, self.h)
            wall_bot = (wall_top + wall_heights).clamp(0, self.h)
            
            shades = (255.0 / (1.0 + dist**2 * 0.12)).clamp(0, 255).byte()
            
            frame = torch.zeros((self.h, self.w, 3), dtype=torch.uint8, device=self.device)
            floor_y = self.y_coords[self.h // 2:] - (self.h // 2) + 1
            floor_stripes = (((30.0 / floor_y.float()) % 1.0 < 0.18).byte() * 40).unsqueeze(-1)
            frame[self.h // 2:, :, :] = 40 + floor_stripes
            frame[:self.h // 2, :, :] = 25
            
            wall_mask = (self.y_coords >= wall_top[None, :]) & (self.y_coords < wall_bot[None, :])
            tex_v = ((self.y_coords.float() - wall_top[None, :].float()) / (wall_heights.float()[None, :] + 1e-4) * 4.0) % 1.0
            is_mortar = (tex_u[None, :] < 0.12) | (tex_v < 0.12)
            
            r_wall = torch.where(hit_types == 2, torch.zeros_like(shades), shades)
            g_wall = shades
            b_wall = torch.where(hit_types == 2, (shades // 2), torch.clamp(shades.float() * 1.15, 0, 255).byte())
            base_col = torch.stack([r_wall, g_wall, b_wall], dim=-1)
            textured_col = torch.where(is_mortar[:, :, None], (base_col.float() * 0.35).byte(), base_col)
            
            frame = torch.where(wall_mask[:, :, None], textured_col, frame)
            return frame.cpu().numpy()

# ==============================================================================
# АВАТАР С ЭГОЦЕНТРИЧЕСКОЙ КИНЕМАТИКОЙ (BODY-FRAME)
# ==============================================================================
class Avatar3D:
    def __init__(self):
        self.x, self.y = 1.5, 1.5
        self.angle = math.pi / 2.0  # Смотрит на Север (+Y)
        self.vx, self.vy = 0.0, 0.0
        self.persistence = 0.0
        self.last_fwd, self.last_str = 0.0, 0.0
        self.trail = []
        
    def update_motion(self, dt, fwd_drive, strafe_drive, turn_rate, maze):
        # 1. Поворот камеры (угол тела)
        self.angle += turn_rate * dt
        self.angle = (self.angle + math.pi) % (2.0 * math.pi) - math.pi
        
        # 2. Персистентность движения
        mag = math.hypot(fwd_drive, strafe_drive)
        alignment = max(0.0, (fwd_drive * self.last_fwd + strafe_drive * self.last_str) / (mag * math.hypot(self.last_fwd, self.last_str) + 1e-6)) if mag > 0.05 else 0.0
        self.persistence = self.persistence * 0.95 + 0.05 * alignment * math.tanh(mag * 2.0)
        self.last_fwd, self.last_str = fwd_drive, strafe_drive

        speed = 3.2 * (1.0 + self.persistence * 2.5)
        
        # 3. Перевод из координат тела (Forward/Strafe) в мировые координаты (X/Y)
        # При angle = pi/2 (вверх): Forward = (0, 1), Strafe = (1, 0)
        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)
        
        target_vx = (fwd_drive * cos_a + strafe_drive * sin_a) * speed
        target_vy = (fwd_drive * sin_a - strafe_drive * cos_a) * speed
        
        self.vx = self.vx * 0.82 + target_vx * 0.18
        self.vy = self.vy * 0.82 + target_vy * 0.18

        # 4. Движение с проверкой стен
        move_dist = math.hypot(self.vx, self.vy) * dt
        steps = max(1, int(math.ceil(move_dist / 0.04)))
        sdx, sdy = (self.vx * dt) / steps, (self.vy * dt) / steps
        r = 0.25

        for _ in range(steps):
            if not maze.is_wall(self.x + sdx + math.copysign(r, sdx), self.y): self.x += sdx
            if not maze.is_wall(self.x, self.y + sdy + math.copysign(r, sdy)): self.y += sdy
            
        self.trail.append((self.x, self.y))
        if len(self.trail) > 35: self.trail.pop(0)

# ==============================================================================
# CORTICAL HTM ENGINE
# ==============================================================================
class CanonicalHTMColumn(nn.Module):
    def __init__(self, num_columns=4096, k_active=80, num_slots=32):
        super().__init__()
        self.num_columns, self.k_active, self.num_slots = num_columns, k_active, num_slots
        ex, ey = [], []
        for i in range(16):
            for j in range(i + 1, 16):
                ex.append((ELECTRODE_X[i] + ELECTRODE_X[j]) / 2.0)
                ey.append((ELECTRODE_Y[i] + ELECTRODE_Y[j]) / 2.0)
        self.register_buffer("edge_x", torch.tensor(ex, device=DEVICE))
        self.register_buffer("edge_y", torch.tensor(ey, device=DEVICE))
        grid_dim = int(math.isqrt(num_columns))
        cy = torch.linspace(-11.0, 11.0, grid_dim, device=DEVICE).view(grid_dim, 1, 1)
        cx = torch.linspace(-11.0, 11.0, grid_dim, device=DEVICE).view(1, grid_dim, 1)
        d_sq = (cx - self.edge_x.view(1, 1, 120))**2 + (cy - self.edge_y.view(1, 1, 120))**2
        self.register_buffer("permanence", torch.exp(-d_sq / 40.0).view(-1, 120)[:num_columns])
        self.perm_threshold = 0.25

    def compute_sdr(self, pac_iplv_32x120: torch.Tensor) -> torch.Tensor:
        connected = (self.permanence >= self.perm_threshold).float()
        x_clean = torch.relu(pac_iplv_32x120)
        if torch.max(x_clean) < 1e-5:
            return torch.full((self.num_slots, self.num_columns), 1e-4, device=DEVICE)
        overlap = torch.matmul(x_clean, connected.T) / torch.sum(connected, dim=1).clamp(min=1.0)
        _, active_indices = torch.topk(overlap, self.k_active, dim=-1)
        sdr_seq = torch.full((self.num_slots, self.num_columns), 0.01, device=DEVICE)
        sdr_seq.scatter_(1, active_indices, 1.0)
        return sdr_seq

class FrontalExecutiveHeterarchy(nn.Module):
    def __init__(self, num_nodes=1, max_capacity=16, num_columns_per_node=4096, k_active_per_node=80, num_slots=32):
        super().__init__()
        self.max_capacity, self.num_slots = max_capacity, num_slots
        self.total_dim = num_columns_per_node * num_nodes
        self.nodes = nn.ModuleList([CanonicalHTMColumn(k_active=k_active_per_node, num_slots=num_slots) for _ in range(num_nodes)])
        self.register_buffer("trajectory_weights", torch.full((max_capacity, num_slots, self.total_dim), 0.05, device=DEVICE))
        self.register_buffer("calcium_trajectory", torch.zeros((num_slots, self.total_dim), device=DEVICE))
        self.register_buffer("membrane_potential", torch.zeros(max_capacity, device=DEVICE))

    def get_current_sdr(self, iplv_gamma_nodes: list[torch.Tensor]):
        sdrs = [self.nodes[i].compute_sdr(iplv_gamma_nodes[i]) if i < len(iplv_gamma_nodes) else torch.zeros((self.num_slots, 4096), device=DEVICE) for i in range(len(self.nodes))]
        return torch.cat(sdrs, dim=-1), sdrs[0]

    def contrastive_learn(self, target_idx: int, num_active: int, lr=0.05, ltd_factor=0.5):
        if torch.max(self.calcium_trajectory) > 1e-5 and abs(lr) > 1e-6:
            self.trajectory_weights[target_idx] = torch.clamp(self.trajectory_weights[target_idx] + lr * self.calcium_trajectory, 0.01, 1.0)
            for o_idx in range(num_active):
                if o_idx != target_idx:
                    self.trajectory_weights[o_idx] = torch.clamp(self.trajectory_weights[o_idx] - (lr * ltd_factor) * self.calcium_trajectory, 0.01, 1.0)

    def predict_evidence(self, cur_sdr_seq: torch.Tensor, active_count: int, dt=0.016, tau=0.250):
        self.calcium_trajectory = torch.max(self.calcium_trajectory * 0.88, cur_sdr_seq)
        w_flat = self.trajectory_weights[:active_count].view(active_count, -1)
        s_flat = self.calcium_trajectory.view(-1)
        current = torch.mv(F.normalize(w_flat, p=2, dim=1), F.normalize(s_flat, p=2, dim=0))
        alpha = dt / tau
        self.membrane_potential[:active_count] = (1.0 - alpha) * self.membrane_potential[:active_count] + alpha * current
        wm_scores = torch.clamp(self.membrane_potential[:active_count], 0.0, 1.0) * 100.0
        weights = torch.softmax(self.membrane_potential[:active_count] * 8.0, dim=0).cpu().numpy()
        return weights, wm_scores.cpu().numpy()

# ==============================================================================
# VLA-JEPA MONTY (БЕЗ АДАМА, ЧИСТАЯ СИНАПТИЧЕСКАЯ ПЛАСТИЧНОСТЬ)
# ==============================================================================
class MontyVLA:
    def __init__(self, jepa_dim=2048, htm_dim=16):
        self.max_capacity = htm_dim
        self.jepa_dim = jepa_dim
        self.heterarchy = FrontalExecutiveHeterarchy(num_nodes=1, max_capacity=htm_dim).to(DEVICE)
        
        self.concept_to_latent = torch.randn((htm_dim, jepa_dim), device=DEVICE)
        self.concept_to_latent = F.normalize(self.concept_to_latent, p=2, dim=1)
        
        # Синапсы моторного вывода (Body Frame: Forward, Strafe)
        self.action_weights = torch.zeros((2, jepa_dim + htm_dim), device=DEVICE)
        
        self.autonomy_score = 0.0
        self.total_learned_steps = 0
        self.match_history = deque(maxlen=150)
        
        self.last_jepa_emb = None
        self.latent_velocity = 0.0
        self.jepa_cosine_sim = 0.0
        self.wm_scores = np.zeros(htm_dim)

    def learn_from_human(self, iplv_gamma_np: np.ndarray, jepa_emb: torch.Tensor, human_fwd: float, human_str: float, dt: float):
        mag = math.hypot(human_fwd, human_str)
        jepa_emb_flat = jepa_emb.view(-1)
        
        if self.last_jepa_emb is not None:
            self.latent_velocity = torch.norm(jepa_emb_flat - self.last_jepa_emb).item()
        self.last_jepa_emb = jepa_emb_flat.clone()
        
        if mag < 0.01: return self.autonomy_score

        t_iplv = torch.tensor(iplv_gamma_np, dtype=torch.float32, device=DEVICE)
        full_sdr_seq, _ = self.heterarchy.get_current_sdr([t_iplv])
        
        # 1. Ассоциация HTM
        similarities = torch.matmul(self.concept_to_latent, jepa_emb_flat)
        target_idx = int(torch.argmax(similarities).item())
        self.heterarchy.contrastive_learn(target_idx, self.max_capacity, lr=0.05, ltd_factor=0.5)
        
        # 2. Выравнивание Ойя
        self.concept_to_latent[target_idx] += 0.05 * (jepa_emb_flat - self.concept_to_latent[target_idx])
        self.concept_to_latent[target_idx] = F.normalize(self.concept_to_latent[target_idx], p=2, dim=0)
        
        # 3. Дельта-правило для моторных синапсов (Widrow-Hoff)
        weights, self.wm_scores = self.heterarchy.predict_evidence(full_sdr_seq, self.max_capacity, dt=dt)
        htm_intent = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
        
        pred_jepa = torch.matmul(htm_intent, self.concept_to_latent)
        self.jepa_cosine_sim = F.cosine_similarity(pred_jepa.unsqueeze(0), jepa_emb_flat.unsqueeze(0)).item()
        
        vla_state = torch.cat([jepa_emb_flat, htm_intent], dim=0)
        pred_action = torch.matmul(self.action_weights, vla_state)
        target_action = torch.tensor([human_fwd, human_str], dtype=torch.float32, device=DEVICE)
        
        error = target_action - pred_action
        self.action_weights += 0.02 * torch.outer(error, vla_state)
        
        self.total_learned_steps += 1
        loss_val = torch.sum(error**2).item()
        self.match_history.append(1.0 if loss_val < 0.18 else 0.0)
        if len(self.match_history) >= 15:
            self.autonomy_score = float(np.mean(self.match_history)) * min(1.0, self.total_learned_steps / 80.0) * 100.0
            
        return self.autonomy_score

    def step_inference(self, iplv_gamma_np: np.ndarray, jepa_emb: torch.Tensor, dt: float) -> tuple[float, float]:
        t_iplv = torch.tensor(iplv_gamma_np, dtype=torch.float32, device=DEVICE)
        full_sdr_seq, _ = self.heterarchy.get_current_sdr([t_iplv])
        
        weights, self.wm_scores = self.heterarchy.predict_evidence(full_sdr_seq, self.max_capacity, dt=dt)
        htm_intent = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
        
        jepa_emb_flat = jepa_emb.view(-1)
        vla_state = torch.cat([jepa_emb_flat, htm_intent], dim=0)
        
        pred_action = torch.matmul(self.action_weights, vla_state)
        return float(pred_action[0].item()), float(pred_action[1].item())

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    pygame.init()
    flags = pygame.HWSURFACE | pygame.DOUBLEBUF
    screen = pygame.display.set_mode((1420, 820), flags, vsync=0)
    pygame.display.set_caption("NeuroCanvas × VLA-JEPA | Rotating Egocentric Map (125 FPS)")
    clock = pygame.time.Clock()

    engine = HeterarchicalBrainEngine()
    engine.start()

    jepa = VLA_JEPA_Wrapper(port=6001, device=DEVICE)

    maze = TopoMaze(DIM)
    avatar = Avatar3D()
    raycaster = GPURaycaster(dim=maze.dim, cam_w=CAM_W, cam_h=CAM_H, device=DEVICE)
    monty = MontyVLA(jepa_dim=jepa.jepa_dim, htm_dim=16)

    AUTOPILOT_MODE = False

    try:
        while True:
            dt = clock.tick(120) / 1000.0
            dt = min(0.05, dt)

            turn_rate_key = 0.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT: raise KeyboardInterrupt
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        AUTOPILOT_MODE = not AUTOPILOT_MODE
                    elif event.key == pygame.K_r:
                        maze = TopoMaze(DIM)
                        avatar.x, avatar.y = 1.5, 1.5

            keys = pygame.key.get_pressed()
            if keys[pygame.K_q] or keys[pygame.K_LEFT]:  turn_rate_key += 2.8 # Поворот влево
            if keys[pygame.K_e] or keys[pygame.K_RIGHT]: turn_rate_key -= 2.8 # Поворот вправо

            frame = engine.get_frame()
            lead_node = frame.fcz_macro
            axes = lead_node.gamepad_axes

            # 1. 3D Текстурированный рендер на CUDA
            frame_3d_np = raycaster.render(maze.grid, avatar.x, avatar.y, avatar.angle)
            
            # 2. V-JEPA Эмбеддинг
            jepa_tensor = jepa.encode_world_state(frame_3d_np)
            if isinstance(jepa_tensor, torch.Tensor):
                real_jepa_emb = jepa_tensor.mean(dim=1).squeeze(0).to(DEVICE)
            else:
                real_jepa_emb = torch.tensor(jepa_tensor).mean(dim=1).squeeze(0).to(DEVICE)
            real_jepa_emb = F.normalize(real_jepa_emb, p=2, dim=0)

            # 3. УПРАВЛЕНИЕ В BODY-FRAME (КООРДИНАТЫ ТЕЛА АВАТАРА)
            # Человек: Вперед/Назад (-ly), Стрейф (lx), Поворот (сагитта rx + клавиши)
            human_fwd = -axes.ly
            human_str = axes.lx
            human_turn = -axes.rx * 3.2 + turn_rate_key

            # Клавиатурный дубляж движения (WASD)
            if keys[pygame.K_w] or keys[pygame.K_UP]:    human_fwd += 1.0
            if keys[pygame.K_s] or keys[pygame.K_DOWN]:  human_fwd -= 1.0
            if keys[pygame.K_d]:                         human_str += 1.0
            if keys[pygame.K_a]:                         human_str -= 1.0

            human_mag = math.hypot(human_fwd, human_str)

            # 4. ОБУЧЕНИЕ ИЛИ АВТОПИЛОТ
            if not AUTOPILOT_MODE:
                drive_fwd, drive_str = human_fwd, human_str
                monty.learn_from_human(lead_node.iplv_gamma, real_jepa_emb, human_fwd, human_str, dt)
                monty_fwd, monty_str = monty.step_inference(lead_node.iplv_gamma, real_jepa_emb, dt)
            else:
                monty_fwd, monty_str = monty.step_inference(lead_node.iplv_gamma, real_jepa_emb, dt)
                drive_fwd, drive_str = monty_fwd, monty_str

            avatar.update_motion(dt, drive_fwd, drive_str, human_turn, maze)

            # ==================================================================
            # РЕНДЕРИНГ ЭГОЦЕНТРИЧЕСКОЙ ВРАЩАЮЩЕЙСЯ КАРТЫ
            # ==================================================================
            screen.fill((8, 11, 16))
            
            # Размеры вьюпорта 2D-карты
            ox, oy = 30, 45
            cx_ui = MAZE_W // 2
            cy_ui = MAZE_H // 2
            
            maze_surface_ui = pygame.Surface((MAZE_W, MAZE_H), pygame.SRCALPHA)
            
            # Математическая проекция мира во вращающийся экран:
            # Аватар всегда смотрит ВВЕРХ (Forward), мир поворачивается на угол (-angle)
            def to_ego(wx, wy):
                dx = wx - avatar.x
                dy = wy - avatar.y
                # Скалярное произведение с локальным Right и Forward
                # sin(angle) и cos(angle)
                sx = cx_ui + int((dx * math.sin(avatar.angle) - dy * math.cos(avatar.angle)) * CELL_SIZE)
                sy = cy_ui + int((-dx * math.cos(avatar.angle) - dy * math.sin(avatar.angle)) * CELL_SIZE)
                return (sx, sy)

            # Отрисовка вращающихся полигонов лабиринта (11x11 = 121 клетка, мгновенно)
            for gy in range(maze.dim):
                for gx in range(maze.dim):
                    if maze.grid[gy][gx] == 1:
                        p0 = to_ego(gx, gy)
                        p1 = to_ego(gx + 1, gy)
                        p2 = to_ego(gx + 1, gy + 1)
                        p3 = to_ego(gx, gy + 1)
                        pygame.draw.polygon(maze_surface_ui, (16, 22, 32), [p0, p1, p2, p3])
                        pygame.draw.polygon(maze_surface_ui, (30, 42, 60), [p0, p1, p2, p3], 1)
                    elif maze.grid[gy][gx] == 2:
                        exit_center = to_ego(gx + 0.5, gy + 0.5)
                        pygame.draw.circle(maze_surface_ui, (0, 255, 120), exit_center, 16)
                        pygame.draw.circle(maze_surface_ui, (255, 255, 255), exit_center, 5)

            # Шлейф аватара во вращающихся координатах
            if len(avatar.trail) > 1:
                trail_pts = [to_ego(tx, ty) for tx, ty in avatar.trail]
                t_col = (255, 80, 200) if AUTOPILOT_MODE else (0, 180, 255)
                pygame.draw.lines(maze_surface_ui, t_col, False, trail_pts, 2)

            # -------------------------------------------------------------
            # 32-ТОЧЕЧНАЯ КОМЕТА ТЕТА-ГАММА РАБОЧЕЙ ПАМЯТИ (Lisman-Jensen)
            # -------------------------------------------------------------
            traj_2d = lead_node.traj_32 # [32, 2]
            base_tx, base_ty = traj_2d[0, 0], traj_2d[0, 1]
            comet_pts = []
            for k in range(32):
                # В координатах тела: X = strafe, Y = forward
                c_str = (traj_2d[k, 0] - base_tx) * 12.0
                c_fwd = (traj_2d[k, 1] - base_ty) * 12.0
                comet_pts.append((cx_ui + int(c_str), cy_ui - int(c_fwd)))
            
            for k in range(31):
                col_c = int(255 * (k / 31.0))
                pygame.draw.line(maze_surface_ui, (col_c, 255 - col_c, 255), comet_pts[k], comet_pts[k+1], 3)
            pygame.draw.circle(maze_surface_ui, (255, 255, 255), comet_pts[-1], 4)

            # ДИНАМИЧЕСКИЙ ВЕКТОР ЧЕЛОВЕКА (БИРЮЗОВЫЙ)
            VEC_SCALE = 80.0
            if human_mag > 0.02:
                h_len = min(1.5, human_mag) * VEC_SCALE * (1.0 + avatar.persistence * 0.4)
                h_end_x = cx_ui + int((human_str / human_mag) * h_len)
                h_end_y = cy_ui - int((human_fwd / human_mag) * h_len)
                pygame.draw.line(maze_surface_ui, (0, 255, 255), (cx_ui, cy_ui), (h_end_x, h_end_y), 4)
                pygame.draw.circle(maze_surface_ui, (255, 255, 255), (h_end_x, h_end_y), 5)

            # ДИНАМИЧЕСКИЙ ВЕКТОР MONTY (ПУРПУРНЫЙ)
            m_mag = math.hypot(monty_fwd, monty_str)
            if m_mag > 0.02:
                m_len = min(1.5, m_mag) * VEC_SCALE * (monty.autonomy_score / 100.0 + 0.3)
                m_end_x = cx_ui + int((monty_str / m_mag) * m_len)
                m_end_y = cy_ui - int((monty_fwd / m_mag) * m_len)
                pygame.draw.line(maze_surface_ui, (255, 60, 180), (cx_ui, cy_ui), (m_end_x, m_end_y), 3)
                pygame.draw.circle(maze_surface_ui, (255, 120, 220), (m_end_x, m_end_y), 4)

            # Аватар (всегда в центре и смотрит строго ВВЕРХ)
            av_col = (255, 50, 200) if AUTOPILOT_MODE else (0, 255, 200)
            pygame.draw.circle(maze_surface_ui, av_col, (cx_ui, cy_ui), 13)
            pygame.draw.circle(maze_surface_ui, (255, 255, 255), (cx_ui, cy_ui), 4)
            # Желтая стрелочка носа аватара (всегда вверх)
            pygame.draw.line(maze_surface_ui, (255, 255, 0), (cx_ui, cy_ui), (cx_ui, cy_ui - 26), 3)
            pygame.draw.circle(maze_surface_ui, (255, 255, 0), (cx_ui, cy_ui - 26), 3)

            # Стрелка компаса: указывает на мировой СЕВЕР (+Y)
            north_pt = to_ego(avatar.x + 0.0, avatar.y + 1.2)
            pygame.draw.line(maze_surface_ui, (255, 60, 60), (cx_ui, cy_ui), north_pt, 2)
            f_xs = pygame.font.SysFont("consolas", 11, bold=True)
            maze_surface_ui.blit(f_xs.render("N", True, (255, 80, 80)), (north_pt[0] - 4, north_pt[1] - 12))

            screen.blit(maze_surface_ui, (ox, oy))
            pygame.draw.rect(screen, (40, 60, 80), (ox, oy, MAZE_W, MAZE_H), 2)

            # --- 3D SENSOR VIEW СПРАВА ---
            surf_3d = pygame.surfarray.make_surface(frame_3d_np.swapaxes(0, 1))
            surf_3d_scaled = pygame.transform.scale(surf_3d, (384, 384))
            screen.blit(surf_3d_scaled, (670, 45))
            pygame.draw.rect(screen, (0, 255, 200), (670, 45, 384, 384), 2)

            # --- ПАНЕЛИ ТЕЛЕМЕТРИИ ---
            f_b = pygame.font.SysFont("consolas", 13, bold=True)
            f_s = pygame.font.SysFont("consolas", 11)
            
            # Панель 1: V-JEPA World Model
            dx_p, dy_p = 1070, 45
            pygame.draw.rect(screen, (14, 18, 26), (dx_p, dy_p, 325, 384), border_radius=6)
            pygame.draw.rect(screen, (0, 200, 255), (dx_p, dy_p, 325, 384), 1, border_radius=6)
            
            screen.blit(f_b.render("V-JEPA 2 WORLD EMBEDDINGS", True, (0, 200, 255)), (dx_p + 12, dy_p + 12))
            screen.blit(f_s.render(f"Port: 6001 | Dim: {jepa.jepa_dim}D", True, (150, 180, 210)), (dx_p + 12, dy_p + 35))
            screen.blit(f_s.render(f"Ego-Velocity ||dz||: {monty.latent_velocity:.3f}", True, (255, 220, 50)), (dx_p + 12, dy_p + 58))
            screen.blit(f_s.render("  (>0.1 = Оптический поток детектирован)", True, (120, 140, 160)), (dx_p + 12, dy_p + 75))
            
            sim_col = (100, 255, 100) if monty.jepa_cosine_sim > 0.70 else (255, 180, 50)
            screen.blit(f_b.render(f"World Prediction Sim: {monty.jepa_cosine_sim:.3f}", True, sim_col), (dx_p + 12, dy_p + 105))

            screen.blit(f_b.render("TOP HTM WORKING MEMORY (L2/3):", True, (255, 120, 220)), (dx_p + 12, dy_p + 155))
            top_concepts = np.argsort(monty.wm_scores)[-4:][::-1]
            for idx, c_i in enumerate(top_concepts):
                sc = monty.wm_scores[c_i]
                screen.blit(f_s.render(f"Concept #{c_i:02d}: {sc:5.1f}%", True, (200, 200, 220)), (dx_p + 12, dy_p + 180 + idx * 22))
                pygame.draw.rect(screen, (30, 40, 50), (dx_p + 135, dy_p + 182 + idx * 22, 120, 10))
                pygame.draw.rect(screen, (255, 60, 180), (dx_p + 135, dy_p + 182 + idx * 22, int((sc/100.0)*120), 10))

            # Панель 2: ЭЭГ Кинематика и Автономия
            by_p = 445
            pygame.draw.rect(screen, (14, 18, 26), (670, by_p, 725, 340), border_radius=6)
            pygame.draw.rect(screen, (100, 255, 120) if AUTOPILOT_MODE else (60, 80, 100), (670, by_p, 725, 340), 2 if AUTOPILOT_MODE else 1, border_radius=6)

            auto_txt = "► АВТОПИЛОТ MONTY [АКТИВЕН]" if AUTOPILOT_MODE else "► ОБУЧЕНИЕ: ЧЕЛОВЕК-УЧИТЕЛЬ (FCz BCI)"
            screen.blit(f_b.render(auto_txt, True, (255, 60, 180) if AUTOPILOT_MODE else (0, 255, 200)), (690, by_p + 15))
            screen.blit(f_b.render(f"ГОТОВНОСТЬ АВТОНОМИИ: {monty.autonomy_score:.1f}%", True, (255, 255, 255)), (690, by_p + 40))

            pygame.draw.rect(screen, (30, 40, 50), (690, by_p + 65, 400, 16), border_radius=4)
            pygame.draw.rect(screen, (255, 60, 180) if AUTOPILOT_MODE else (0, 255, 200), (690, by_p + 65, int((monty.autonomy_score / 100.0) * 400), 16), border_radius=4)

            screen.blit(f_s.render(f"• Координаты тела: Fwd={drive_fwd:+.2f} | Strafe={drive_str:+.2f} | Угол={math.degrees(avatar.angle):.0f}°", True, (200, 220, 255)), (690, by_p + 95))
            screen.blit(f_s.render(f"• Намерение Человека (Бирюзовый): Fwd={human_fwd:+.2f}, Str={human_str:+.2f}", True, (0, 255, 255)), (690, by_p + 115))
            screen.blit(f_s.render(f"• Вывод Action Head (Пурпурный): Fwd={monty_fwd:+.2f}, Str={monty_str:+.2f}", True, (255, 60, 180)), (690, by_p + 135))
            screen.blit(f_s.render("• Управление: Вперед/Назад (W/S), Стрейф (A/D), Поворот (Q/E или сагитта rx)", True, (160, 180, 200)), (690, by_p + 155))

            # Спектр 120 ребер ciPLV
            screen.blit(f_s.render("120-EDGE DIRECTED ciPLV СЕТЬ (FCz):", True, (150, 160, 170)), (690, by_p + 185))
            g120 = np.abs(lead_node.iplv_gamma[-1])
            max_g = np.max(g120) + 1e-6
            for p in range(120):
                bh = int((g120[p] / max_g) * 65)
                col_b = (255, 80, 80) if p < 6 else ((80, 255, 120) if p < 72 else (80, 150, 255))
                pygame.draw.rect(screen, col_b, (690 + p * 5.8, by_p + 270 - bh, 4, bh))

            fps_val = clock.get_fps()
            screen.blit(f_b.render(f"FPS: {fps_val:.0f} | [SPACE] Автопилот | [Q/E] Поворот | [R] Ресет", True, (255, 255, 255)), (30, 15))

            pygame.display.flip()

    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
        pygame.quit()

if __name__ == '__main__':
    main()
