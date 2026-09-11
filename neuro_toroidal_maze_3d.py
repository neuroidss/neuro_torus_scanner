#!/usr/bin/env python3
"""
🧠 NEUROCANVAS: TRUE 3D TOROIDAL MAZE RUNNER (PAC-MAN WRAPAROUND T^2)
- Лабиринт физически развернут на замкнутой поверхности 3D Тора T^2 = S^1 x S^1.
- Бесшовная тороидальная топология: выход за границу возвращает с противоположной стороны.
- Аватар движется непрерывным фазовым потоком рабочей памяти из FCz.
- 100% Real-Time, 60 FPS, нативное ядро CUDA.
"""

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import multiprocessing as mp

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

import math
import time
import random
import numpy as np
import torch
import pygame

from neuro_heterarchy_core import (
    HeterarchicalBrainEngine, NUM_CHANNELS, NUM_FREQS, NUM_PAIRS, 
    COORDS_X, COORDS_Y, I_IDX, J_IDX
)

WIDTH, HEIGHT = 1320, 780
CENTER_X, CENTER_Y = 400, 390
R_MAJOR, R_MINOR = 200.0, 80.0

DIM_TH = 16  # 16 секторов по главному кругу Тора (Theta)
DIM_PH = 12  # 12 секторов по малому кругу Тора (Phi)

I_GPU = torch.tensor(I_IDX, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), dtype=torch.long)
J_GPU = torch.tensor(J_IDX, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), dtype=torch.long)

# ==============================================================================
# 1. ГЕНЕРАТОР БЕСШОВНОГО ТОРОИДАЛЬНОГО ЛАБИРИНТА (T^2 PAC-MAN TOPOLOGY)
# ==============================================================================
class ToroidalMaze:
    def __init__(self, dim_th=DIM_TH, dim_ph=DIM_PH):
        self.dim_th = dim_th
        self.dim_ph = dim_ph
        # Стены: 0 = проход, 1 = стена
        # walls_v: вертикальные стены между (th, ph) и ((th+1)%dim_th, ph)
        # walls_h: горизонтальные стены между (th, ph) и (th, (ph+1)%dim_ph)
        self.walls_v = [[1 for _ in range(dim_ph)] for _ in range(dim_th)]
        self.walls_h = [[1 for _ in range(dim_ph)] for _ in range(dim_th)]
        self.visited = [[False for _ in range(dim_ph)] for _ in range(dim_th)]
        self._generate_toroidal_dfs(0, 0)
        
        # Размещаем целевые кристаллы на поверхности Тора
        self.target_th = random.randint(0, dim_th - 1) + 0.5
        self.target_ph = random.randint(0, dim_ph - 1) + 0.5

    def _generate_toroidal_dfs(self, th, ph):
        self.visited[th][ph] = True
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        random.shuffle(dirs)
        
        for dth, dph in dirs:
            nth = (th + dth) % self.dim_th
            nph = (ph + dph) % self.dim_ph
            
            if not self.visited[nth][nph]:
                # Ломаем тороидальную стену
                if dth == 1: self.walls_v[th][ph] = 0
                elif dth == -1: self.walls_v[nth][ph] = 0
                elif dph == 1: self.walls_h[th][ph] = 0
                elif dph == -1: self.walls_h[th][nph] = 0
                self._generate_toroidal_dfs(nth, nph)

    def is_blocked(self, cur_th, cur_ph, next_th, next_ph):
        """Проверка коллизии со стенами с учетом тороидального переноса"""
        c_cell_th = int(cur_th) % self.dim_th
        c_cell_ph = int(cur_ph) % self.dim_ph
        n_cell_th = int(next_th) % self.dim_th
        n_cell_ph = int(next_ph) % self.dim_ph
        
        if c_cell_th == n_cell_th and c_cell_ph == n_cell_ph:
            return False # Внутри той же ячейки коллизий нет
            
        # Переход по горизонтали (Theta)
        if (c_cell_th + 1) % self.dim_th == n_cell_th:
            if self.walls_v[c_cell_th][c_cell_ph] == 1: return True
        elif (n_cell_th + 1) % self.dim_th == c_cell_th:
            if self.walls_v[n_cell_th][c_cell_ph] == 1: return True
            
        # Переход по вертикали (Phi)
        if (c_cell_ph + 1) % self.dim_ph == n_cell_ph:
            if self.walls_h[c_cell_th][c_cell_ph] == 1: return True
        elif (n_cell_ph + 1) % self.dim_ph == c_cell_ph:
            if self.walls_h[c_cell_th][n_cell_ph] == 1: return True
            
        return False

# ==============================================================================
# 2. ФИЗИКА ТОРОИДАЛЬНОГО ПИЛОТА
# ==============================================================================
class ToroidalPilotAvatar:
    def __init__(self):
        self.th = 0.5  # Положение на торе: 0 .. DIM_TH
        self.ph = 0.5  # Положение на торе: 0 .. DIM_PH
        self.v_th = 0.0
        self.v_ph = 0.0
        self.persistence = 0.0
        self.trail = []

    def update_motion(self, dt, force_x, force_y, temp_bias, maze):
        mag = math.hypot(force_x, force_y)
        self.persistence = self.persistence * 0.94 + 0.06 * math.tanh(mag * 2.5)
        
        speed = (2.6 + self.persistence * 3.2) * (1.0 + temp_bias * 0.4)
        
        target_v_th = force_x * speed
        target_v_ph = -force_y * speed # Вверх по экрану = крутим Phi вперед

        self.v_th = self.v_th * 0.84 + target_v_th * 0.16
        self.v_ph = self.v_ph * 0.84 + target_v_ph * 0.16

        # Шаг движения с проверкой коллизий
        steps = 4
        dth = (self.v_th * dt) / steps
        dph = (self.v_ph * dt) / steps

        for _ in range(steps):
            next_th = (self.th + dth) % maze.dim_th
            if not maze.is_blocked(self.th, self.ph, next_th, self.ph):
                self.th = next_th
                
            next_ph = (self.ph + dph) % maze.dim_ph
            if not maze.is_blocked(self.th, self.ph, self.th, next_ph):
                self.ph = next_ph

        # Сохранение следа на торе
        self.trail.append((self.th, self.ph))
        if len(self.trail) > 40: self.trail.pop(0)

# ==============================================================================
# 3. 3D ТОПОЛОГИЧЕСКИЙ РЕНДЕРЕР ТОРА И ЛАБИРИНТА
# ==============================================================================
def torus_to_3d(theta, phi):
    x = (R_MAJOR + R_MINOR * math.cos(phi)) * math.cos(theta)
    y = (R_MAJOR + R_MINOR * math.cos(phi)) * math.sin(theta)
    z = R_MINOR * math.sin(phi)
    return x, y, z

def project_3d(x, y, z, yaw=0.0, pitch=0.82):
    x1, y1 = x * math.cos(yaw) - y * math.sin(yaw), x * math.sin(yaw) + y * math.cos(yaw)
    y2, z2 = y1 * math.cos(pitch) - z * math.sin(pitch), y1 * math.sin(pitch) + z * math.cos(pitch)
    return int(CENTER_X + x1), int(CENTER_Y - y2), z2

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("NeuroCanvas: 3D Toroidal Maze Runner (T^2 Manifold)")
    font_xs = pygame.font.SysFont("consolas", 11)
    font_sm = pygame.font.SysFont("consolas", 13, bold=True)
    font_med = pygame.font.SysFont("consolas", 16, bold=True)
    font_lg = pygame.font.SysFont("consolas", 20, bold=True)
    font_huge = pygame.font.SysFont("consolas", 36, bold=True)
    clock = pygame.time.Clock()

    engine = HeterarchicalBrainEngine()
    engine.start()

    maze = ToroidalMaze(DIM_TH, DIM_PH)
    avatar = ToroidalPilotAvatar()

    running = True
    torus_yaw = 0.0
    crystals_collected = 0

    try:
        while running:
            dt = min(0.05, clock.tick(60) / 1000.0)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        maze = ToroidalMaze(DIM_TH, DIM_PH)
                        avatar.th, avatar.ph = 0.5, 0.5
                        crystals_collected = 0

            # 1. Получаем чистый вектор намерения из FCz
            frame = engine.get_frame()
            node = frame.fcz_macro
            ax = node.gamepad_axes
            traj_32 = node.traj_32

            # 2. Физика тороидального перемещения
            force_x = ax.lx
            force_y = -ax.ly
            avatar.update_motion(dt, force_x, force_y, ax.ry, maze)

            # 3. Проверка взятия кристалла цели
            d_th_t = abs(avatar.th - maze.target_th)
            d_ph_t = abs(avatar.ph - maze.target_ph)
            if min(d_th_t, DIM_TH - d_th_t) < 0.6 and min(d_ph_t, DIM_PH - d_ph_t) < 0.6:
                crystals_collected += 1
                maze.target_th = random.randint(0, DIM_TH - 1) + 0.5
                maze.target_ph = random.randint(0, DIM_PH - 1) + 0.5

            # ==================================================================
            # 4. ОТРИСОВКА
            # ==================================================================
            screen.fill((6, 8, 12))

            # --- A. 3D ТОР С НАТЯНУТЫМ ЛАБИРИНТОМ (СЛЕВА) ---
            # 1. Прозрачный каркас ячеек лабиринта
            for th_i in range(DIM_TH):
                th_val = (th_i / DIM_TH) * 2.0 * math.pi
                for ph_i in range(DIM_PH):
                    ph_val = (ph_i / DIM_PH) * 2.0 * math.pi
                    
                    # Рисуем вертикальные стены (разделители между секторами Theta)
                    if maze.walls_v[th_i][ph_i] == 1:
                        th_next = ((th_i + 1) / DIM_TH) * 2.0 * math.pi
                        p1 = project_3d(*torus_to_3d(th_next, ph_val), yaw=torus_yaw)
                        p2 = project_3d(*torus_to_3d(th_next, ph_val + (2.0*math.pi/DIM_PH)), yaw=torus_yaw)
                        pygame.draw.line(screen, (30, 48, 75), p1[:2], p2[:2], 2)
                        
                    # Рисуем горизонтальные стены (разделители между секторами Phi)
                    if maze.walls_h[th_i][ph_i] == 1:
                        ph_next = ((ph_i + 1) / DIM_PH) * 2.0 * math.pi
                        p1 = project_3d(*torus_to_3d(th_val, ph_next), yaw=torus_yaw)
                        p2 = project_3d(*torus_to_3d(th_val + (2.0*math.pi/DIM_TH), ph_next), yaw=torus_yaw)
                        pygame.draw.line(screen, (30, 48, 75), p1[:2], p2[:2], 2)

            # 2. След аватара на поверхности 3D Тора
            if len(avatar.trail) > 1:
                trail_3d = []
                for t_th, t_ph in avatar.trail:
                    rad_th = (t_th / DIM_TH) * 2.0 * math.pi
                    rad_ph = (t_ph / DIM_PH) * 2.0 * math.pi
                    trail_3d.append(project_3d(*torus_to_3d(rad_th, rad_ph), yaw=torus_yaw)[:2])
                pygame.draw.lines(screen, (0, 180, 255), False, trail_3d, 2)

            # 3. Кристалл-Цель на 3D Торе
            tgt_rad_th = (maze.target_th / DIM_TH) * 2.0 * math.pi
            tgt_rad_ph = (maze.target_ph / DIM_PH) * 2.0 * math.pi
            tx, ty, _ = project_3d(*torus_to_3d(tgt_rad_th, tgt_rad_ph), yaw=torus_yaw)
            pulse_r = int(9 + math.sin(time.time() * 8) * 3)
            pygame.draw.circle(screen, (255, 220, 0), (tx, ty), pulse_r, 2)
            pygame.draw.circle(screen, (255, 200, 50), (tx, ty), 5)

            # 4. Сам Аватар на 3D Торе
            av_rad_th = (avatar.th / DIM_TH) * 2.0 * math.pi
            av_rad_ph = (avatar.ph / DIM_PH) * 2.0 * math.pi
            ax_p, ay_p, _ = project_3d(*torus_to_3d(av_rad_th, av_rad_ph), yaw=torus_yaw)
            pygame.draw.circle(screen, (255, 50, 200), (ax_p, ay_p), 9)
            pygame.draw.circle(screen, (255, 255, 255), (ax_p, ay_p), 3)

            # Стрелка направления из аватара
            intent_len = math.hypot(force_x, force_y) * 40.0
            if intent_len > 5.0:
                end_ax = ax_p + int(force_x * intent_len)
                end_ay = ay_p - int(force_y * intent_len)
                pygame.draw.line(screen, (0, 255, 220), (ax_p, ay_p), (end_ax, end_ay), 3)
                pygame.draw.circle(screen, (255, 255, 255), (end_ax, end_ay), 4)

            # --- B. 2D РАЗВЕРТКА ТОРОИДАЛЬНОЙ КАРТЫ (СПРАВА ВВЕРХУ) ---
            PANEL_X = 760
            pygame.draw.rect(screen, (12, 16, 24), (PANEL_X, 15, 500, 380), border_radius=6)
            pygame.draw.rect(screen, (30, 42, 60), (PANEL_X, 15, 500, 380), 1, border_radius=6)
            screen.blit(font_lg.render("2D ТОРОИДАЛЬНАЯ КАРТА (PAC-MAN WRAP)", True, (0, 255, 200)), (PANEL_X + 20, 22))

            # Развертка лабиринта (Flat Torus Map)
            map_w, map_h = 460, 280
            map_ox, map_oy = PANEL_X + 20, 65
            cell_w = map_w / DIM_TH
            cell_h = map_h / DIM_PH

            pygame.draw.rect(screen, (16, 22, 32), (map_ox, map_oy, map_w, map_h))
            
            # Стены на 2D карте
            for th_i in range(DIM_TH):
                for ph_i in range(DIM_PH):
                    cx_cell = map_ox + th_i * cell_w
                    cy_cell = map_oy + ph_i * cell_h
                    if maze.walls_v[th_i][ph_i] == 1:
                        pygame.draw.line(screen, (45, 65, 95), (cx_cell + cell_w, cy_cell), (cx_cell + cell_w, cy_cell + cell_h), 2)
                    if maze.walls_h[th_i][ph_i] == 1:
                        pygame.draw.line(screen, (45, 65, 95), (cx_cell, cy_cell + cell_h), (cx_cell + cell_w, cy_cell + cell_h), 2)

            # Таргет на 2D карте
            t2_x = map_ox + int(maze.target_th * cell_w)
            t2_y = map_oy + int(maze.target_ph * cell_h)
            pygame.draw.circle(screen, (255, 220, 0), (t2_x, t2_y), 6)

            # Аватар на 2D карте
            a2_x = map_ox + int(avatar.th * cell_w)
            a2_y = map_oy + int(avatar.ph * cell_h)
            pygame.draw.circle(screen, (255, 50, 200), (a2_x, a2_y), 7)
            pygame.draw.circle(screen, (255, 255, 255), (a2_x, a2_y), 3)

            screen.blit(font_xs.render("• Выход за правую границу -> Появление слева | Выход вверх -> Появление снизу", True, (140, 160, 180)), (PANEL_X + 20, 355))

            # --- C. ТЕЛЕМЕТРИЯ НАВИГАЦИИ И КРИСТАЛЛОВ (СПРАВА ВНИЗУ) ---
            BOT_Y = 410
            pygame.draw.rect(screen, (12, 16, 24), (PANEL_X, BOT_Y, 500, 315), border_radius=6)
            pygame.draw.rect(screen, (30, 42, 60), (PANEL_X, BOT_Y, 500, 315), 1, border_radius=6)

            screen.blit(font_med.render(f"СОБРАНО ЦЕЛЕЙ НА ТОРЕ: {crystals_collected}", True, (255, 220, 100)), (PANEL_X + 20, BOT_Y + 18))
            screen.blit(font_huge.render(f"★ {crystals_collected}", True, (0, 255, 180)), (PANEL_X + 20, BOT_Y + 45))

            # Телеметрия осей
            heading_deg = math.degrees(math.atan2(force_y, force_x)) % 360
            screen.blit(font_sm.render(f"• Фазовый компас (Θ):  {heading_deg:.0f}°", True, (200, 220, 255)), (PANEL_X + 200, BOT_Y + 45))
            screen.blit(font_sm.render(f"• Тета-Часы мозга:     {frame.theta_freq:.2f} Hz", True, (255, 150, 200)), (PANEL_X + 200, BOT_Y + 70))
            screen.blit(font_sm.render(f"• Разгон намерения:    x{(1.0 + avatar.persistence * 3.8):.1f}", True, (0, 255, 200)), (PANEL_X + 200, BOT_Y + 95))

            # 120 ребер спектра
            mean_g120 = np.mean(np.abs(node.iplv_32), axis=0)
            max_v = np.max(mean_g120) + 1e-6
            for p in range(NUM_PAIRS):
                bh = int((mean_g120[p] / max_v) * 140)
                col_b = (255, 80, 80) if p < 6 else ((80, 255, 120) if p < 72 else (80, 150, 255))
                pygame.draw.rect(screen, col_b, (PANEL_X + 20 + p * 3.8, BOT_Y + 295 - bh, 2.5, bh))

            screen.blit(font_xs.render("СПЕКТР 120 iPLV | 6 Core / 66 Outer / 48 Cross", True, (150, 160, 170)), (PANEL_X + 20, BOT_Y + 135))

            # Статус бар внизу
            s_txt = f"● ТОРОИДАЛЬНЫЙ ПОЛЕТ: Направляйте аватар через коридоры Тора! [R] Новый лабиринт"
            screen.blit(font_med.render(s_txt, True, (0, 255, 200)), (25, HEIGHT - 35))

            pygame.display.flip()

    finally:
        engine.stop()
        pygame.quit()

if __name__ == '__main__':
    main()
