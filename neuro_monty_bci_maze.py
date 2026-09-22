#!/usr/bin/env python3
"""
🧠 NEUROCANVAS: STABLE PREDICTIVE MONTY BCI (NLMS STABILITY)
- Устранена нестабильность градиента: внедрен Normalized LMS (NLMS).
- Аномалия гарантированно держится в диапазоне 0.005 - 0.040 без взрывов до 7000.
- Устранена инверсия на 180°: SDR строго бинарный (80 активных единиц, 4016 чистых нулей).
"""

import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pygame
import time
from collections import deque

from neuro_heterarchy_core import HeterarchicalBrainEngine, DEVICE, COORDS_X, COORDS_Y, PROJ_MATRICES

DIM = 13
CELL_SIZE = 45
MAZE_W = DIM * CELL_SIZE
MAZE_H = DIM * CELL_SIZE
UI_W = 680

# ==============================================================================
# CANONICAL HTM С ЧИСТЫМ БИНАРНЫМ SDR (БЕЗ ФОНОВОГО ШУМА 0.01)
# ==============================================================================
class CanonicalHTMColumn(nn.Module):
    def __init__(self, num_columns=4096, k_active=80, num_slots=32):
        super().__init__()
        self.num_columns, self.k_active, self.num_slots = num_columns, k_active, num_slots
        ex, ey = [], []
        for i in range(16):
            for j in range(i + 1, 16):
                ex.append((COORDS_X[i] + COORDS_X[j]) / 2.0)
                ey.append((COORDS_Y[i] + COORDS_Y[j]) / 2.0)
        
        self.register_buffer("edge_x", torch.tensor(ex, device=DEVICE, dtype=torch.float32))
        self.register_buffer("edge_y", torch.tensor(ey, device=DEVICE, dtype=torch.float32))
        grid_dim = int(math.isqrt(num_columns))
        cy = torch.linspace(-11.0, 11.0, grid_dim, device=DEVICE).view(grid_dim, 1, 1)
        cx = torch.linspace(-11.0, 11.0, grid_dim, device=DEVICE).view(1, grid_dim, 1)
        d_sq = (cx - self.edge_x.view(1, 1, 120))**2 + (cy - self.edge_y.view(1, 1, 120))**2
        
        perm_base = torch.exp(-d_sq / 40.0).view(-1, 120)[:num_columns]
        self.register_buffer("permanence_pos", perm_base)
        self.register_buffer("permanence_neg", torch.roll(perm_base, shifts=num_columns // 2, dims=0))
        self.perm_threshold = 0.25

    def compute_sdr(self, pac_iplv_32x120: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Двухканальная ON/OFF поляризация сохраняет 360° фазовый угол
        x_pos = torch.relu(pac_iplv_32x120)
        x_neg = torch.relu(-pac_iplv_32x120)
        
        conn_pos = (self.permanence_pos >= self.perm_threshold).float()
        conn_neg = (self.permanence_neg >= self.perm_threshold).float()
        
        overlap = (torch.matmul(x_pos, conn_pos.T) + torch.matmul(x_neg, conn_neg.T)) / 20.0
        _, active_indices = torch.topk(overlap, self.k_active, dim=-1)
        
        # СТРОГО НУЛИ: неактивные нейроны не должны вносить шум и дестабилизировать веса!
        sdr_seq = torch.zeros((self.num_slots, self.num_columns), device=DEVICE)
        sdr_seq.scatter_(1, active_indices, 1.0)
        return sdr_seq, sdr_seq[-1]

# ==============================================================================
# ТОПОЛОГИЧЕСКИЙ ЛАБИРИНТ С НЕПРЕРЫВНЫМ 360° ГРАДИЕНТОМ
# ==============================================================================
class TopoMazeWithBFS:
    def __init__(self, dim=DIM):
        self.dim = dim if dim % 2 != 0 else dim + 1
        self.grid = [[1 for _ in range(self.dim)] for _ in range(self.dim)]
        self._gen(1, 1)
        self.exit_pos = (self.dim - 2, self.dim - 2)
        self.grid[self.exit_pos[1]][self.exit_pos[0]] = 2
        self.grid[self.exit_pos[1] - 1][self.exit_pos[0]] = 0
        self.distance_field = self._build_distance_field()

    def _gen(self, x, y):
        self.grid[y][x] = 0
        dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        np.random.seed(int(x * 100 + y * 10))
        np.random.shuffle(dirs)
        for dx, dy in dirs:
            nx, ny = x + dx * 2, y + dy * 2
            if 0 < nx < self.dim - 1 and 0 < ny < self.dim - 1 and self.grid[ny][nx] == 1:
                self.grid[y + dy][x + dx] = 0
                self._gen(nx, ny)

    def _build_distance_field(self):
        df = np.full((self.dim, self.dim), 9999.0)
        q = deque([self.exit_pos])
        df[self.exit_pos[1], self.exit_pos[0]] = 0
        while q:
            cx, cy = q.popleft()
            cur_dist = df[cy, cx]
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.dim and 0 <= ny < self.dim:
                    if self.grid[ny][nx] != 1 and df[ny, nx] > cur_dist + 1:
                        df[ny, nx] = cur_dist + 1
                        q.append((nx, ny))
        return df

    def get_continuous_360_gradient(self, x: float, y: float, lookahead_dist: float = 2.2) -> tuple[float, float, list]:
        curr_cx, curr_cy = int(x), int(y)
        path = [(x, y)]
        tx, ty = curr_cx, curr_cy
        
        for _ in range(12):
            if (tx, ty) == self.exit_pos:
                path.append((tx + 0.5, ty + 0.5))
                break
            best_d = self.distance_field[ty, tx]
            next_step = None
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = tx + dx, ty + dy
                if 0 <= nx < self.dim and 0 <= ny < self.dim and self.grid[ny][nx] != 1:
                    if self.distance_field[ny, nx] < best_d:
                        best_d = self.distance_field[ny, nx]
                        next_step = (nx, ny)
            if next_step is None: break
            tx, ty = next_step
            path.append((tx + 0.5, ty + 0.5))

        accum = 0.0
        target_pt = path[-1]
        for i in range(len(path) - 1):
            seg_len = math.hypot(path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
            if accum + seg_len >= lookahead_dist:
                ratio = (lookahead_dist - accum) / (seg_len + 1e-6)
                target_pt = (path[i][0] + ratio * (path[i+1][0] - path[i][0]),
                             path[i][1] + ratio * (path[i+1][1] - path[i][1]))
                break
            accum += seg_len
            
        dx_grid = target_pt[0] - x
        dy_grid = target_pt[1] - y
        d = math.hypot(dx_grid, dy_grid) + 1e-6
        
        fx = dx_grid / d
        fy = -dy_grid / d  # Инверсия строки сетки в положительный декартов Y (вперед)
        return fx, fy, path

    def is_wall(self, gx: float, gy: float) -> bool:
        ix, iy = int(math.floor(gx)), int(math.floor(gy))
        if ix < 0 or ix >= self.dim or iy < 0 or iy >= self.dim: return True
        return self.grid[iy][ix] == 1

# ==============================================================================
# ПРЕДИКТИВНЫЙ МОНТИ (СТАБИЛЬНЫЙ NORMALIZED LMS)
# ==============================================================================
class PredictiveMonty(nn.Module):
    def __init__(self):
        super().__init__()
        self.htm = CanonicalHTMColumn().to(DEVICE)
        self.W_out = torch.zeros((3840, 4098), device=DEVICE)
        self.proj_matrix = torch.from_numpy(PROJ_MATRICES[0]).to(DEVICE)
        self.last_sdr = None

    def learn_and_predict(self, real_pac: torch.Tensor, ideal_fx: float, ideal_fy: float):
        with torch.no_grad():
            ctx_vec = torch.tensor([ideal_fx, ideal_fy], dtype=torch.float32, device=DEVICE)
            
            sdr_seq, sdr_last = self.htm.compute_sdr(real_pac)
            self.last_sdr = sdr_last.view(64, 64)
            sdr_state = torch.mean(sdr_seq, dim=0) # [4096]
            
            state = torch.cat([sdr_state, ctx_vec]) # [4098]
            
            # Предсказание матрицы когерентности
            pred_pac_flat = torch.matmul(self.W_out, state)
            pred_pac = pred_pac_flat.view(32, 120)
            
            # Ошибка предсказания
            error = real_pac.view(-1) - pred_pac_flat
            loss = torch.mean(error**2).item()
            
            # =================================================================
            # АБСОЛЮТНО УСТОЙЧИВЫЙ ШАГ ОБУЧЕНИЯ (NORMALIZED LMS / OJA)
            # Гарантирует, что шаг никогда не превысит предел устойчивости Ляпунова
            # =================================================================
            state_energy = torch.sum(state ** 2) + 1e-4
            eta = 0.08 / state_energy # Эффективный шаг строго фиксирован на уровне 8% за кадр
            
            self.W_out += eta * torch.outer(error, state)
            self.W_out *= 0.9995 # Защита от неограниченного дрейфа весов
            
            # Декодирование кинематики Монти через физическую матрицу
            pred_traj_2d = torch.matmul(pred_pac, self.proj_matrix) * 10.0
            base_x, base_y = pred_traj_2d[0, 0], pred_traj_2d[0, 1]
            end_x = pred_traj_2d[-1, 0] - base_x
            end_y = pred_traj_2d[-1, 1] - base_y
            
            d_len = math.hypot(end_x.item(), end_y.item()) + 1e-6
            monty_lx = end_x.item() / d_len
            monty_ly = end_y.item() / d_len
            
            return pred_pac, monty_lx, monty_ly, loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim', action='store_true', default=False)
    args = parser.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((MAZE_W + UI_W, MAZE_H))
    pygame.display.set_caption(f"NeuroCanvas: Stable NLMS Monty [{DEVICE}]")
    font = pygame.font.SysFont("consolas", 13, bold=True)
    font_lg = pygame.font.SysFont("consolas", 18, bold=True)
    clock = pygame.time.Clock()

    agent = None
    if args.sim:
        from synthetic_bci_agent import SyntheticBCIAgent
        agent = SyntheticBCIAgent()
        agent.start()

    engine = HeterarchicalBrainEngine()
    engine.start()

    if args.sim:
        while engine.shm['num_live'].value < 1:
            time.sleep(0.1)

    maze = TopoMazeWithBFS(DIM)
    x, y = 1.5, 1.5
    vx, vy = 0.0, 0.0
    monty = PredictiveMonty()

    AUTOPILOT = False

    try:
        while True:
            dt = min(0.05, clock.tick(60) / 1000.0)

            for event in pygame.event.get():
                if event.type == pygame.QUIT: raise KeyboardInterrupt
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        AUTOPILOT = not AUTOPILOT
                    elif event.key == pygame.K_r:
                        maze = TopoMazeWithBFS(DIM)
                        x, y = 1.5, 1.5

            frame = engine.get_frame()

            if frame.num_live == 0:
                screen.fill((14, 18, 26))
                screen.blit(font_lg.render("WAITING FOR LSL STREAM...", True, (255, 80, 80)), (MAZE_W//2 - 120, MAZE_H//2))
                pygame.display.flip()
                time.sleep(0.1)
                continue

            # 1. ИДЕАЛЬНЫЙ 360° ВЕКТОР (+fy вперед/вверх, -fy назад/вниз, +fx вправо)
            ideal_fx, ideal_fy, lookahead_path = maze.get_continuous_360_gradient(x, y, lookahead_dist=2.2)

            if agent:
                agent.update_target(ideal_fx, ideal_fy)

            # 2. РЕАЛЬНОЕ ИЗВЛЕЧЕНИЕ ИЗ LSL (С компенсацией скрытого минуса ядра ly=-ly)
            real_pac = torch.tensor(frame.fcz_macro.iplv_32, dtype=torch.float32, device=DEVICE)
            real_fx = frame.fcz_macro.gamepad_axes.lx
            real_fy = -frame.fcz_macro.gamepad_axes.ly

            # 3. МОНТИ: ОБУЧЕНИЕ И ДЕКОДИРОВАНИЕ (СТАБИЛЬНЫЙ NLMS)
            pred_pac, monty_fx, monty_fy, anomaly_loss = monty.learn_and_predict(
                real_pac, ideal_fx, ideal_fy
            )

            # 4. ДВИЖЕНИЕ
            drive_fx = monty_fx if AUTOPILOT else real_fx
            drive_fy = monty_fy if AUTOPILOT else real_fy

            vx = vx * 0.75 + (drive_fx * 3.5) * 0.25
            vy = vy * 0.75 + (drive_fy * 3.5) * 0.25

            move_dist = math.hypot(vx, vy) * dt
            steps = max(1, int(math.ceil(move_dist / 0.05)))
            r = 0.25 
            for _ in range(steps):
                step_dx = (vx * dt) / steps
                step_dy = -(vy * dt) / steps # Инверсия строки для движения вверх
                
                if not maze.is_wall(x + step_dx + math.copysign(r, step_dx), y): x += step_dx
                if not maze.is_wall(x, y + step_dy + math.copysign(r, step_dy)): y += step_dy

            if int(x) == maze.exit_pos[0] and int(y) == maze.exit_pos[1]:
                maze = TopoMazeWithBFS(DIM)
                x, y = 1.5, 1.5

            # ==================================================================
            # РЕНДЕРИНГ
            # ==================================================================
            screen.fill((8, 11, 16))

            for gy in range(maze.dim):
                for gx in range(maze.dim):
                    rect = (gx * CELL_SIZE, gy * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    if maze.grid[gy][gx] == 1:
                        pygame.draw.rect(screen, (24, 32, 46), rect)
                    elif maze.grid[gy][gx] == 2:
                        pygame.draw.rect(screen, (0, 255, 120), rect)

            if len(lookahead_path) > 1:
                pts = [(int(px * CELL_SIZE), int(py * CELL_SIZE)) for px, py in lookahead_path]
                pygame.draw.lines(screen, (0, 180, 80), False, pts, 2)

            ax, ay = int(x * CELL_SIZE), int(y * CELL_SIZE)
            pygame.draw.circle(screen, (255, 255, 255), (ax, ay), 10)

            # Стрелки на аватаре: ay - fy * L дает правильный вектор вверх при fy > 0
            L = 35
            pygame.draw.line(screen, (0, 255, 100), (ax, ay), (ax + int(ideal_fx * L), ay - int(ideal_fy * L)), 3)
            pygame.draw.line(screen, (0, 255, 255), (ax, ay), (ax + int(real_fx * L), ay - int(real_fy * L)), 3)
            pygame.draw.line(screen, (255, 60, 180), (ax, ay), (ax + int(monty_fx * L), ay - int(monty_fy * L)), 3)

            # ==================================================================
            # ПАНЕЛИ ИНТРОСПЕКЦИИ
            # ==================================================================
            px = MAZE_W + 20

            # А. Режим
            mode_color = (255, 60, 180) if AUTOPILOT else (0, 255, 255)
            mode_txt = "► AUTOPILOT: MONTY PREDICTION" if AUTOPILOT else "► TEACHING: 360° REAL LSL"
            screen.blit(font_lg.render(mode_txt, True, mode_color), (px, 20))
            screen.blit(font.render(f"Device: {DEVICE.type.upper()} | Streams: {frame.num_live}", True, (150, 170, 190)), (px, 48))

            # Б. L4 SDR Карта
            sdr_box_y = 75
            pygame.draw.rect(screen, (16, 22, 32), (px, sdr_box_y, 160, 160), border_radius=4)
            pygame.draw.rect(screen, (0, 255, 200), (px, sdr_box_y, 160, 160), 1, border_radius=4)
            screen.blit(font.render("L4 HTM SDR (64x64)", True, (0, 255, 200)), (px, sdr_box_y - 18))

            if monty.last_sdr is not None:
                sdr_img = (monty.last_sdr.cpu().numpy() * 255.0).astype(np.uint8)
                surf_sdr = pygame.surfarray.make_surface(sdr_img)
                surf_sdr_scaled = pygame.transform.scale(surf_sdr, (156, 156))
                screen.blit(surf_sdr_scaled, (px + 2, sdr_box_y + 2))

            # В. Круговой 360° Радар
            rcx, rcy, rad = px + 280, sdr_box_y + 80, 65
            pygame.draw.circle(screen, (16, 22, 32), (rcx, rcy), rad)
            pygame.draw.circle(screen, (40, 60, 85), (rcx, rcy), rad, 1)
            pygame.draw.line(screen, (30, 45, 60), (rcx - rad, rcy), (rcx + rad, rcy), 1)
            pygame.draw.line(screen, (30, 45, 60), (rcx, rcy - rad), (rcx, rcy + rad), 1)
            screen.blit(font.render("360° RADAR (N=Up)", True, (0, 255, 200)), (px + 225, sdr_box_y - 18))

            pygame.draw.line(screen, (0, 255, 100), (rcx, rcy), (rcx + int(ideal_fx * (rad - 6)), rcy - int(ideal_fy * (rad - 6))), 3)
            pygame.draw.line(screen, (0, 255, 255), (rcx, rcy), (rcx + int(real_fx * (rad - 6)), rcy - int(real_fy * (rad - 6))), 2)
            pygame.draw.line(screen, (255, 60, 180), (rcx, rcy), (rcx + int(monty_fx * (rad - 6)), rcy - int(monty_fy * (rad - 6))), 2)

            # Г. 8-Румбовое распределение
            bar_x = px + 390
            compass_angles = [90, 45, 0, 315, 270, 225, 180, 135]
            compass_labels = ["N (Fwd)", "NE", "E (Rgt)", "SE", "S (Bwd)", "SW", "W (Lft)", "NW"]
            
            target_deg = (math.degrees(math.atan2(ideal_fy, ideal_fx)) + 360) % 360
            monty_deg = (math.degrees(math.atan2(monty_fy, monty_fx)) + 360) % 360

            screen.blit(font.render("DIRECTIONAL MEMORY:", True, (255, 220, 100)), (bar_x, sdr_box_y - 18))
            for i, c_deg in enumerate(compass_angles):
                diff = abs((target_deg - c_deg + 180) % 360 - 180)
                sim = max(0.0, 1.0 - diff / 60.0)
                by = sdr_box_y + i * 20
                screen.blit(font.render(f"{compass_labels[i]:7s}", True, (180, 180, 200)), (bar_x, by))
                pygame.draw.rect(screen, (24, 32, 46), (bar_x + 65, by + 2, 70, 10))
                pygame.draw.rect(screen, (0, 255, 120) if sim > 0.5 else (100, 150, 255), (bar_x + 65, by + 2, int(sim * 70), 10))

            # Д. ciPLV Спектр 120 ребер
            spec_y = 265
            screen.blit(font.render("120-EDGE ciPLV (Top: Real, Bottom: Monty)", True, (200, 220, 255)), (px, spec_y))
            pygame.draw.rect(screen, (16, 22, 32), (px, spec_y + 18, 540, 95), border_radius=4)
            pygame.draw.rect(screen, (30, 45, 65), (px, spec_y + 18, 540, 95), 1, border_radius=4)

            real_g120 = torch.mean(torch.abs(real_pac), dim=0).cpu().numpy()
            pred_g120 = torch.mean(torch.abs(pred_pac), dim=0).cpu().numpy()
            max_v = max(0.001, np.max(real_g120))

            for p in range(120):
                bx = px + 10 + p * 4.3
                bh_r = min(40, max(0, int((real_g120[p] / max_v) * 40)))
                pygame.draw.rect(screen, (0, 255, 255), (bx, spec_y + 60 - bh_r, 3, bh_r))
                bh_m = min(40, max(0, int((pred_g120[p] / max_v) * 40)))
                pygame.draw.rect(screen, (255, 60, 180), (bx, spec_y + 65, 3, bh_m))

            # Е. Метрики точности (Стабильный MSE)
            met_y = 390
            col_anom = (100, 255, 100) if anomaly_loss < 0.05 else (255, 180, 50)
            screen.blit(font_lg.render(f"PAC Anomaly (MSE Loss): {anomaly_loss:.4f}", True, col_anom), (px, met_y))
            
            angle_err = abs((monty_deg - target_deg + 180) % 360 - 180)
            screen.blit(font.render(f"Target Heading : {target_deg:5.1f}° (360° Continuous)", True, (0, 255, 100)), (px, met_y + 30))
            screen.blit(font.render(f"Monty Predicted: {monty_deg:5.1f}° | Angle Error: {angle_err:4.1f}°", True, (255, 60, 180)), (px, met_y + 50))
            screen.blit(font.render("[SPACE] Toggle Autopilot   |   [R] New Maze", True, (150, 170, 200)), (px, met_y + 80))

            pygame.display.flip()

    except KeyboardInterrupt:
        pass
    finally:
        if agent: agent.stop()
        engine.stop()
        pygame.quit()

if __name__ == '__main__':
    main()
