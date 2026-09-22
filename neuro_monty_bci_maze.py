#!/usr/bin/env python3
"""
🧠 NEUROCANVAS: STABLE PREDICTIVE MONTY BCI (EXACT WEB ENGINE PORT)
- Полный порт физики, масштабов и сглаживания из EngineConfig.ts и BrainMazeScene.tsx.
- Лабиринт динамически центрируется и гарантированно 100% влезает в окно.
- Генерация лабиринта и выхода 1:1 из maze.ts (алгоритм findHardestExit, 200 попыток).
- Все настройки вынесены в CLI-параметры.
- Зафиксированы значения по умолчанию: --hide-path True, --vec-scale 80.0.
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

# Геометрия окна
DIM = 13
MAZE_AREA_W = 620
MAZE_AREA_H = 620
CELL_SIZE = int((MAZE_AREA_W - 40) / DIM)  # ~44 пикселя на клетку
GRID_PX_W = DIM * CELL_SIZE
GRID_PX_H = DIM * CELL_SIZE
OFFSET_X = (MAZE_AREA_W - GRID_PX_W) // 2
OFFSET_Y = (MAZE_AREA_H - GRID_PX_H) // 2
UI_W = 680

# ==============================================================================
# CANONICAL HTM С ЧИСТЫМ БИНАРНЫМ SDR
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
        x_pos = torch.relu(pac_iplv_32x120)
        x_neg = torch.relu(-pac_iplv_32x120)
        conn_pos = (self.permanence_pos >= self.perm_threshold).float()
        conn_neg = (self.permanence_neg >= self.perm_threshold).float()
        overlap = (torch.matmul(x_pos, conn_pos.T) + torch.matmul(x_neg, conn_neg.T)) / 20.0
        _, active_indices = torch.topk(overlap, self.k_active, dim=-1)
        sdr_seq = torch.zeros((self.num_slots, self.num_columns), device=DEVICE)
        sdr_seq.scatter_(1, active_indices, 1.0)
        return sdr_seq, sdr_seq[-1]

# ==============================================================================
# ТОПОЛОГИЧЕСКИЙ ЛАБИРИНТ (ТОЧНЫЙ ПОРТ ИЗ maze.ts)
# ==============================================================================
class TopoMazeWithBFS:
    def __init__(self, dim=DIM):
        self.dim = dim if dim % 2 != 0 else dim + 1
        self.grid = []
        self.exit_pos = (1, 1)
        self.optimal_dist = 0
        
        attempts = 0
        is_valid = False
        best_exit = None
        best_grid = None

        # Точный алгоритм findHardestExit из maze.ts (200 попыток)
        while not is_valid and attempts < 200:
            attempts += 1
            self.grid = [[1 for _ in range(self.dim)] for _ in range(self.dim)]
            self._gen(1, 1)
            
            exit_params = self._find_hardest_exit()
            if not best_exit or (exit_params['d'] + exit_params['turns'] > best_exit['d'] + best_exit['turns']):
                best_exit = exit_params
                best_grid = [row[:] for row in self.grid]
            
            if exit_params['d'] >= 20 and exit_params['turns'] >= 5:
                is_valid = True

        self.grid = best_grid
        self.exit_pos = (best_exit['x'], best_exit['y'])
        self.optimal_dist = best_exit['d']
        self.grid[self.exit_pos[1]][self.exit_pos[0]] = 2
        
        self.distance_field = self._build_distance_field()

    def _gen(self, x, y):
        self.grid[y][x] = 0
        dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        np.random.shuffle(dirs)
        for dx, dy in dirs:
            nx = x + dx * 2
            ny = y + dy * 2
            if 0 < nx < self.dim - 1 and 0 < ny < self.dim - 1 and self.grid[ny][nx] == 1:
                self.grid[y + dy][x + dx] = 0
                self._gen(nx, ny)

    def _find_hardest_exit(self):
        q = deque([{'x': 1, 'y': 1, 'd': 0, 'dx': 0, 'dy': 0, 'turns': 0}])
        visited = [[False]*self.dim for _ in range(self.dim)]
        visited[1][1] = True
        best = {'x': 1, 'y': 1, 'd': 0, 'turns': 0}
        max_score = 0

        while q:
            curr = q.popleft()
            score = curr['d'] + curr['turns'] * 3
            if score > max_score and (curr['x'] != 1 or curr['y'] != 1):
                max_score = score
                best = curr

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = curr['x'] + dx, curr['y'] + dy
                if 0 < nx < self.dim - 1 and 0 < ny < self.dim - 1:
                    if not visited[ny][nx] and self.grid[ny][nx] == 0:
                        visited[ny][nx] = True
                        is_turn = (curr['dx'] != 0 or curr['dy'] != 0) and (curr['dx'] != dx or curr['dy'] != dy)
                        q.append({
                            'x': nx, 'y': ny, 
                            'd': curr['d'] + 1, 
                            'dx': dx, 'dy': dy, 
                            'turns': curr['turns'] + (1 if is_turn else 0)
                        })
        return best

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
        # +X = Вправо, +Y = Вниз по экрану
        return dx_grid / d, dy_grid / d, path

    def is_wall(self, gx: float, gy: float) -> bool:
        ix, iy = int(math.floor(gx)), int(math.floor(gy))
        if ix < 0 or ix >= self.dim or iy < 0 or iy >= self.dim: return True
        return self.grid[iy][ix] == 1

# ==============================================================================
# АВАТАР (ТОЧНЫЙ ПОРТ КИНЕМАТИКИ ИЗ BrainMazeScene.tsx + EngineConfig.ts)
# ==============================================================================
class Avatar3D:
    def __init__(self, sensitivity=0.05, max_speed=0.15, fwd_scale=0.2, strafe_scale=0.2, turn_scale=0.5, intent_gain=1.5):
        self.x, self.y = 1.5, 1.5
        self.angle = 0.0
        
        # Настройки из EngineConfig.Maze
        self.sensitivity = sensitivity
        self.max_speed = max_speed
        self.forward_speed_scale = fwd_scale
        self.strafe_speed_scale = strafe_scale
        self.turn_speed_scale = turn_scale
        self.intent_gain = intent_gain
        
        # Leaky-интеграторы (BrainMazeScene.tsx)
        self.ctrl_moveX = 0.0
        self.ctrl_moveY = 0.0
        self.ctrl_torque = 0.0
        
        self.last_ix = 0.0
        self.last_iy = 0.0
        self.persistence = 0.0
        self.trail = []
        
    def update_motion(self, dt, intent_x, intent_y, intent_tq, maze, rotating_view):
        # 1. Расчет Persistence (BleService.ts)
        mag = math.hypot(intent_x, intent_y)
        dot = intent_x * self.last_ix + intent_y * self.last_iy
        cos_th = dot / (mag * math.hypot(self.last_ix, self.last_iy) + 1e-6)
        
        if mag > 0.05 and cos_th > 0.8:
            self.persistence = min(1.0, self.persistence + 0.05)
        else:
            self.persistence *= 0.95
            
        self.last_ix = intent_x
        self.last_iy = intent_y

        # 2. Мягкое сглаживание в точности как в BrainMazeScene.tsx
        skill_level = self.sensitivity
        smooth = 0.98 - (skill_level * 0.1)
        gain = skill_level * self.intent_gain
        active_boost = 1.0 + self.persistence * 4.0
        
        self.ctrl_moveX = self.ctrl_moveX * smooth + intent_x * gain * (1.0 - smooth)
        self.ctrl_moveY = self.ctrl_moveY * smooth + intent_y * gain * (1.0 - smooth)
        self.ctrl_torque = self.ctrl_torque * smooth + intent_tq * gain * 0.5 * (1.0 - smooth)

        # 3. Вращение
        if rotating_view:
            self.angle += self.ctrl_torque * active_boost * self.turn_speed_scale
            self.angle = (self.angle + math.pi) % (2.0 * math.pi) - math.pi
            
            forward_speed = -self.ctrl_moveY * self.forward_speed_scale * active_boost
            strafe_speed = self.ctrl_moveX * self.strafe_speed_scale * active_boost

            raw_dx = math.sin(self.angle) * forward_speed + math.cos(self.angle) * strafe_speed
            raw_dy = -math.cos(self.angle) * forward_speed + math.sin(self.angle) * strafe_speed
        else:
            self.angle += self.ctrl_torque * active_boost * 0.5
            self.angle = (self.angle + math.pi) % (2.0 * math.pi) - math.pi
            
            raw_dx = self.ctrl_moveX * self.strafe_speed_scale * active_boost
            raw_dy = self.ctrl_moveY * self.forward_speed_scale * active_boost

        # 4. Ограничение предельной скорости (maxSpeed из EngineConfig)
        intended_move = math.hypot(raw_dx, raw_dy)
        target_dx, target_dy = raw_dx, raw_dy
        if intended_move > self.max_speed:
            target_dx = (raw_dx / intended_move) * self.max_speed
            target_dy = (raw_dy / intended_move) * self.max_speed

        # 5. Проход по шагам с коллизиями (строка 157 BrainMazeScene.tsx)
        steps = max(1, int(math.ceil(max(abs(target_dx), abs(target_dy)) / 0.05)))
        sdx = target_dx / steps
        sdy = target_dy / steps

        for _ in range(steps):
            if not maze.is_wall(self.x + sdx + math.copysign(0.2, sdx), self.y): self.x += sdx
            if not maze.is_wall(self.x, self.y + sdy + math.copysign(0.2, sdy)): self.y += sdy
            
        self.trail.append((self.x, self.y))
        if len(self.trail) > 30: self.trail.pop(0)

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

    def learn_and_predict(self, real_pac: torch.Tensor, ideal_x: float, ideal_y: float):
        with torch.no_grad():
            ctx_vec = torch.tensor([ideal_x, ideal_y], dtype=torch.float32, device=DEVICE)
            sdr_seq, sdr_last = self.htm.compute_sdr(real_pac)
            self.last_sdr = sdr_last.view(64, 64)
            sdr_state = torch.mean(sdr_seq, dim=0)
            state = torch.cat([sdr_state, ctx_vec])
            
            pred_pac_flat = torch.matmul(self.W_out, state)
            pred_pac = pred_pac_flat.view(32, 120)
            
            error = real_pac.view(-1) - pred_pac_flat
            loss = torch.mean(error**2).item()
            
            state_energy = torch.sum(state ** 2) + 1e-4
            eta = 0.08 / state_energy
            self.W_out += eta * torch.outer(error, state)
            self.W_out *= 0.9995
            
            pred_traj_2d = torch.matmul(pred_pac, self.proj_matrix) * 10.0
            base_x, base_y = pred_traj_2d[0, 0], pred_traj_2d[0, 1]
            end_x = pred_traj_2d[-1, 0] - base_x
            end_y = pred_traj_2d[-1, 1] - base_y
            
            d_len = math.hypot(end_x.item(), end_y.item()) + 1e-6
            monty_x = end_x.item() / d_len
            monty_y = -end_y.item() / d_len
            
            return pred_pac, monty_x, monty_y, loss

def get_screen_coords(wx, wy, avatar, rotating_view):
    """ Проекция: лабиринт всегда центрирован и полностью виден в окне. """
    if not rotating_view:
        return (int(OFFSET_X + wx * CELL_SIZE), int(OFFSET_Y + wy * CELL_SIZE))
        
    cx_ui = MAZE_AREA_W // 2
    cy_ui = MAZE_AREA_H // 2
    dx = wx - avatar.x
    dy = wy - avatar.y
    
    rot = -avatar.angle
    sx = dx * math.cos(rot) - dy * math.sin(rot)
    sy = dx * math.sin(rot) + dy * math.cos(rot)
    
    return (int(cx_ui + sx * CELL_SIZE), int(cy_ui + sy * CELL_SIZE))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim', action='store_true', default=False)
    parser.add_argument('--rotate', action='store_true', default=False, help="Включить вращающийся мир по умолчанию")
    parser.add_argument('--hide-path', action='store_true', default=True, help="Скрыть зеленую линию пути и вектор")
    parser.add_argument('--vec-scale', type=float, default=80.0, help="Длина векторов нейрофидбека (по умолчанию 80)")
    parser.add_argument('--hide-trail', action='store_true', default=True, help="Скрыть синий хвост")
    parser.add_argument('--hide-monty', action='store_true', default=True, help="Скрыть розовый вектор Монти")
    
    # Настройки из EngineConfig.ts
    parser.add_argument('--sensitivity', type=float, default=0.05, help="moveSensitivity из веба (по умолчанию 0.05)")
    parser.add_argument('--max-speed', type=float, default=0.15, help="maxSpeed из EngineConfig (по умолчанию 0.15)")
    parser.add_argument('--fwd-scale', type=float, default=0.2, help="forwardSpeedScale из EngineConfig (по умолчанию 0.2)")
    parser.add_argument('--strafe-scale', type=float, default=0.2, help="strafeSpeedScale из EngineConfig (по умолчанию 0.2)")
    parser.add_argument('--turn-scale', type=float, default=0.5, help="turnSpeedScale из EngineConfig (по умолчанию 0.5)")
    parser.add_argument('--intent-gain', type=float, default=1.5, help="intentGain из EngineConfig (по умолчанию 1.5)")
    parser.add_argument('--intent-mag', type=float, default=15.0, help="intentMoveMagnitude для клавиатуры (по умолчанию 15.0)")
    args = parser.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((MAZE_AREA_W + UI_W, MAZE_AREA_H))
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
    avatar = Avatar3D(
        sensitivity=args.sensitivity,
        max_speed=args.max_speed,
        fwd_scale=args.fwd_scale,
        strafe_scale=args.strafe_scale,
        turn_scale=args.turn_scale,
        intent_gain=args.intent_gain
    )
    monty = PredictiveMonty()

    AUTOPILOT = False
    ROTATING_VIEW = args.rotate
    SHOW_PATH = not args.hide_path
    SHOW_TRAIL = not args.hide_trail
    SHOW_MONTY = not args.hide_monty
    VEC_SCALE = args.vec_scale

    try:
        while True:
            dt = min(0.05, clock.tick(60) / 1000.0)

            for event in pygame.event.get():
                if event.type == pygame.QUIT: raise KeyboardInterrupt
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        AUTOPILOT = not AUTOPILOT
                    elif event.key == pygame.K_F3:
                        ROTATING_VIEW = not ROTATING_VIEW
                    elif event.key == pygame.K_F4:
                        maze = TopoMazeWithBFS(DIM)
                        avatar.x, avatar.y = 1.5, 1.5
                        avatar.angle = 0.0
                        avatar.trail.clear()
                    elif event.key == pygame.K_F5:
                        SHOW_PATH = not SHOW_PATH
                    elif event.key == pygame.K_F6:
                        SHOW_TRAIL = not SHOW_TRAIL
                    elif event.key == pygame.K_F7:
                        SHOW_MONTY = not SHOW_MONTY

            frame = engine.get_frame()

            if frame.num_live == 0:
                screen.fill((14, 18, 26))
                screen.blit(font_lg.render("WAITING FOR LSL STREAM...", True, (255, 80, 80)), (MAZE_AREA_W//2 - 120, MAZE_AREA_H//2))
                pygame.display.flip()
                time.sleep(0.1)
                continue

            # 1. ИДЕАЛЬНЫЙ ВЕКТОР (dx вправо, dy вниз)
            ideal_x, ideal_y, lookahead_path = maze.get_continuous_360_gradient(avatar.x, avatar.y, lookahead_dist=2.2)

            if agent:
                agent.update_target(ideal_x, -ideal_y)

            # 2. ИЗВЛЕЧЕНИЕ СЫРЫХ LSL ПАРАМЕТРОВ (Только Тета-Гамма)
            real_pac = torch.tensor(frame.fcz_macro.iplv_32, dtype=torch.float32, device=DEVICE)
            real_x = frame.fcz_macro.gamepad_axes.lx
            real_y = frame.fcz_macro.gamepad_axes.ly  # Синхронизированный знак: -1 это ВВЕРХ
            real_tq = frame.fcz_macro.gamepad_axes.rx

            # 3. МОНТИ: ОБУЧЕНИЕ И ДЕКОДИРОВАНИЕ
            pred_pac, monty_x, monty_y, anomaly_loss = monty.learn_and_predict(
                real_pac, ideal_x, ideal_y
            )

            # 4. ВЫБОР ИСТОЧНИКА ДВИЖЕНИЯ
            # В вебе сырой интент умножается на intentMoveMagnitude для клавиатуры
            drive_x = (monty_x if AUTOPILOT else real_x) * args.intent_mag
            drive_y = (monty_y if AUTOPILOT else real_y) * args.intent_mag
            drive_tq = real_tq * args.turn_scale

            # Ручной override со стрелок и точек/запятых
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:     drive_y -= args.intent_mag
            if keys[pygame.K_DOWN]:   drive_y += args.intent_mag
            if keys[pygame.K_RIGHT]:  drive_x += args.intent_mag
            if keys[pygame.K_LEFT]:   drive_x -= args.intent_mag
            if keys[pygame.K_PERIOD]: drive_tq += 2.0
            if keys[pygame.K_COMMA]:  drive_tq -= 2.0

            # 5. ДВИЖЕНИЕ АВАТАРА (через сглаживание BrainMazeScene.tsx)
            avatar.update_motion(dt, drive_x, drive_y, drive_tq, maze, ROTATING_VIEW)

            # 6. ПРОВЕРКА ВЫХОДА (строка 317 BrainMazeScene.tsx)
            px_cur = int(math.floor(avatar.x))
            py_cur = int(math.floor(avatar.y))
            if 0 <= py_cur < maze.dim and 0 <= px_cur < maze.dim:
                if maze.grid[py_cur][px_cur] == 2:
                    maze = TopoMazeWithBFS(DIM)
                    avatar.x, avatar.y = 1.5, 1.5
                    avatar.angle = 0.0
                    avatar.trail.clear()

            # ==================================================================
            # РЕНДЕРИНГ
            # ==================================================================
            screen.fill((8, 11, 16))

            # Рамка лабиринта
            pygame.draw.rect(screen, (16, 24, 34), (OFFSET_X - 2, OFFSET_Y - 2, GRID_PX_W + 4, GRID_PX_H + 4), 1)

            for gy in range(maze.dim):
                for gx in range(maze.dim):
                    if maze.grid[gy][gx] == 1:
                        p0 = get_screen_coords(gx, gy, avatar, ROTATING_VIEW)
                        p1 = get_screen_coords(gx + 1, gy, avatar, ROTATING_VIEW)
                        p2 = get_screen_coords(gx + 1, gy + 1, avatar, ROTATING_VIEW)
                        p3 = get_screen_coords(gx, gy + 1, avatar, ROTATING_VIEW)
                        
                        # Эффект свечения стен из веба (от persistence)
                        gb_val = int(40 + avatar.persistence * 110)
                        pygame.draw.polygon(screen, (10, gb_val, int(70 + avatar.persistence * 120)), [p0, p1, p2, p3])
                        pygame.draw.polygon(screen, (34, 211, 238), [p0, p1, p2, p3], 1)
                    elif maze.grid[gy][gx] == 2:
                        c = get_screen_coords(gx + 0.5, gy + 0.5, avatar, ROTATING_VIEW)
                        pygame.draw.circle(screen, (0, 255, 102), c, int(CELL_SIZE * 0.45))
                        pygame.draw.circle(screen, (255, 255, 255), c, int(CELL_SIZE * 0.15))

            if SHOW_PATH and len(lookahead_path) > 1:
                pts = [get_screen_coords(px, py, avatar, ROTATING_VIEW) for px, py in lookahead_path]
                pygame.draw.lines(screen, (0, 180, 80), False, pts, 3)

            # Синий след (скрыт по умолчанию, переключается по F6)
            if SHOW_TRAIL and len(avatar.trail) > 1:
                trail_pts = [get_screen_coords(tx, ty, avatar, ROTATING_VIEW) for tx, ty in avatar.trail]
                t_col = (255, 80, 200) if AUTOPILOT else (0, 180, 255)
                pygame.draw.lines(screen, t_col, False, trail_pts, 2)

            a_sx, a_sy = get_screen_coords(avatar.x, avatar.y, avatar, ROTATING_VIEW)
            
            # Аватар
            av_col = (0, int(150 + avatar.persistence * 105), 255) if not AUTOPILOT else (255, 50, 200)
            pygame.draw.circle(screen, av_col, (a_sx, a_sy), int(CELL_SIZE * 0.28))
            
            # Направление взгляда (Нос)
            if ROTATING_VIEW:
                pygame.draw.line(screen, (255, 255, 255), (a_sx, a_sy), (a_sx, a_sy - 16), 3)
            else:
                nose_x = a_sx + int(math.sin(avatar.angle) * 16)
                nose_y = a_sy - int(math.cos(avatar.angle) * 16)
                pygame.draw.line(screen, (255, 255, 255), (a_sx, a_sy), (nose_x, nose_y), 3)

            # Векторы намерений на экране
            L = VEC_SCALE
            if SHOW_PATH:
                pygame.draw.line(screen, (0, 255, 100), (a_sx, a_sy), (a_sx + int(ideal_x * L), a_sy + int(ideal_y * L)), 4)
                
            pygame.draw.line(screen, (0, 255, 255), (a_sx, a_sy), (a_sx + int(real_x * L), a_sy + int(real_y * L)), 4)
            
            if SHOW_MONTY:
                pygame.draw.line(screen, (255, 60, 180), (a_sx, a_sy), (a_sx + int(monty_x * L), a_sy + int(monty_y * L)), 4)

            # ==================================================================
            # ПАНЕЛИ ИНТРОСПЕКЦИИ
            # ==================================================================
            px = MAZE_AREA_W + 20

            mode_color = (255, 60, 180) if AUTOPILOT else (0, 255, 255)
            mode_txt = "► AUTOPILOT: MONTY PREDICTION" if AUTOPILOT else "► TEACHING: 360° REAL LSL"
            screen.blit(font_lg.render(mode_txt, True, mode_color), (px, 20))
            
            cam_mode = "EGO [ROTATING]" if ROTATING_VIEW else "WORLD [FIXED]"
            screen.blit(font.render(f"Device: {DEVICE.type.upper()} | Streams: {frame.num_live} | CAM: {cam_mode}", True, (150, 170, 190)), (px, 48))

            sdr_box_y = 75
            pygame.draw.rect(screen, (16, 22, 32), (px, sdr_box_y, 160, 160), border_radius=4)
            pygame.draw.rect(screen, (0, 255, 200), (px, sdr_box_y, 160, 160), 1, border_radius=4)
            screen.blit(font.render("L4 HTM SDR (64x64)", True, (0, 255, 200)), (px, sdr_box_y - 18))

            if monty.last_sdr is not None:
                sdr_img = (monty.last_sdr.cpu().numpy() * 255.0).astype(np.uint8)
                surf_sdr = pygame.surfarray.make_surface(sdr_img)
                surf_sdr_scaled = pygame.transform.scale(surf_sdr, (156, 156))
                screen.blit(surf_sdr_scaled, (px + 2, sdr_box_y + 2))

            rcx, rcy, rad = px + 280, sdr_box_y + 80, 65
            pygame.draw.circle(screen, (16, 22, 32), (rcx, rcy), rad)
            pygame.draw.circle(screen, (40, 60, 85), (rcx, rcy), rad, 1)
            pygame.draw.line(screen, (30, 45, 60), (rcx - rad, rcy), (rcx + rad, rcy), 1)
            pygame.draw.line(screen, (30, 45, 60), (rcx, rcy - rad), (rcx, rcy + rad), 1)
            screen.blit(font.render("360° RADAR (N=Up)", True, (0, 255, 200)), (px + 225, sdr_box_y - 18))

            radar_scale = rad - 6
            if SHOW_PATH:
                pygame.draw.line(screen, (0, 255, 100), (rcx, rcy), (rcx + int(ideal_x * radar_scale), rcy + int(ideal_y * radar_scale)), 3)
            pygame.draw.line(screen, (0, 255, 255), (rcx, rcy), (rcx + int(real_x * radar_scale), rcy + int(real_y * radar_scale)), 2)
            if SHOW_MONTY:
                pygame.draw.line(screen, (255, 60, 180), (rcx, rcy), (rcx + int(monty_x * radar_scale), rcy + int(monty_y * radar_scale)), 2)

            bar_x = px + 390
            compass_angles = [90, 45, 0, 315, 270, 225, 180, 135]
            compass_labels = ["N (Fwd)", "NE", "E (Rgt)", "SE", "S (Bwd)", "SW", "W (Lft)", "NW"]
            
            target_deg = (math.degrees(math.atan2(-ideal_y, ideal_x)) + 360) % 360
            monty_deg = (math.degrees(math.atan2(-monty_y, monty_x)) + 360) % 360

            screen.blit(font.render("DIRECTIONAL MEMORY:", True, (255, 220, 100)), (bar_x, sdr_box_y - 18))
            for i, c_deg in enumerate(compass_angles):
                by = sdr_box_y + i * 20
                screen.blit(font.render(f"{compass_labels[i]:7s}", True, (180, 180, 200)), (bar_x, by))
                pygame.draw.rect(screen, (24, 32, 46), (bar_x + 65, by + 2, 70, 10))
                
                if SHOW_PATH:
                    diff = abs((target_deg - c_deg + 180) % 360 - 180)
                    sim = max(0.0, 1.0 - diff / 60.0)
                    pygame.draw.rect(screen, (0, 255, 120) if sim > 0.5 else (100, 150, 255), (bar_x + 65, by + 2, int(sim * 70), 10))

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

            met_y = 390
            col_anom = (100, 255, 100) if anomaly_loss < 0.05 else (255, 180, 50)
            screen.blit(font_lg.render(f"PAC Anomaly (MSE Loss): {anomaly_loss:.4f}", True, col_anom), (px, met_y))
            
            if SHOW_PATH:
                angle_err = abs((monty_deg - target_deg + 180) % 360 - 180)
                screen.blit(font.render(f"Target Heading : {target_deg:5.1f}°", True, (0, 255, 100)), (px, met_y + 30))
                screen.blit(font.render(f"Monty Predicted: {monty_deg:5.1f}° | Angle Error: {angle_err:4.1f}°", True, (255, 60, 180)), (px, met_y + 50))
            else:
                screen.blit(font.render("Target Heading : [HIDDEN BY F5]", True, (100, 150, 100)), (px, met_y + 30))
                screen.blit(font.render(f"Monty Predicted: {monty_deg:5.1f}°", True, (255, 60, 180)), (px, met_y + 50))
            
            screen.blit(font.render(f"Persistence (Speed Boost): {avatar.persistence:.2f}", True, (255, 255, 100)), (px, met_y + 75))
            screen.blit(font.render("[SPACE] Autopilot  | [F3] Cam | [F4] New Maze", True, (150, 170, 200)), (px, met_y + 95))
            screen.blit(font.render("[F5] Path  | [F6] Trail  | [F7] Monty Vec", True, (150, 170, 200)), (px, met_y + 115))

            pygame.display.flip()

    except KeyboardInterrupt:
        pass
    finally:
        if agent: agent.stop()
        engine.stop()
        pygame.quit()

if __name__ == '__main__':
    main()
