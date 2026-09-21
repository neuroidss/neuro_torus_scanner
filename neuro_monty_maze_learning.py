#!/usr/bin/env python3
"""
🧠 NEUROCANVAS × MONTY: SEPARATE GOAL & SENSOR SYSTEMS WITH AUTONOMOUS TAKEOVER
- Разделение систем по канонам tbp.monty:
    * MazeSensorModule (SM): знает ТОЛЬКО локальные стены (никакого знания о выходе).
    * HippocampalGoalGenerator (GSG): ведет когнитивную карту и знает, где выход.
    * MontyFrontalLM (LM): связывает локальные стены, цель и моторное намерение человека.
- Переключение режимов по нажатию [ПРОБЕЛ]:
    * [HUMAN BCI]: Человек ведет аватар через FCz, Monty учится повторять ваши решения.
    * [MONTY AUTONOMOUS]: Monty сам управляет аватаром на основе выученного у вас!
- Честная оценка автономии (0..100%): растет по мере того, как Monty точно угадывает ваш выбор.
- 100% эталонная стрелка человека (желтая/бирюзовая) + стрелка автопилота Monty (пурпурная).
"""

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import sys
import math
import time
from collections import deque
import numpy as np
import torch
import pygame

from neuro_heterarchy_core import (
    HeterarchicalBrainEngine, NUM_FREQS, NUM_PAIRS, DEVICE, NodeState, SCALE_28_120
)
from tbp.monty.cmp import Message, Goal

DIM = 11
CELL_SIZE = 56
MAZE_W = DIM * CELL_SIZE
MAZE_H = DIM * CELL_SIZE

# ==============================================================================
# 1. ЛАБИРИНТ
# ==============================================================================
class TopoMaze:
    def __init__(self, dim=DIM):
        self.dim = dim
        self.grid = [[1 for _ in range(dim)] for _ in range(dim)]
        self._gen(1, 1)
        self.exit_pos = (dim - 2, dim - 2)
        self.grid[self.exit_pos[1]][self.exit_pos[0]] = 2
        self.dist_map = self._compute_bfs()
        self.bake_surface()

    def _gen(self, x, y):
        self.grid[y][x] = 0
        dirs = [(0, 2), (0, -2), (2, 0), (-2, 0)]
        np.random.shuffle(dirs)
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 0 < nx < self.dim - 1 and 0 < ny < self.dim - 1 and self.grid[ny][nx] == 1:
                self.grid[y + dy // 2][x + dx // 2] = 0
                self._gen(nx, ny)

    def _compute_bfs(self):
        dist = np.full((self.dim, self.dim), 999.0, dtype=np.float32)
        ex, ey = self.exit_pos
        dist[ey, ex] = 0.0
        q = deque([(ex, ey)])
        while q:
            cx, cy = q.popleft()
            cd = dist[cy, cx]
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.dim and 0 <= ny < self.dim:
                    if self.grid[ny][nx] != 1 and dist[ny, nx] > cd + 1.0:
                        dist[ny, nx] = cd + 1.0
                        q.append((nx, ny))
        return dist

    def get_distance_at(self, x: float, y: float) -> float:
        gx = int(np.clip(math.floor(x), 0, self.dim - 1))
        gy = int(np.clip(math.floor(y), 0, self.dim - 1))
        base_d = self.dist_map[gy, gx]
        ex, ey = self.exit_pos[0] + 0.5, self.exit_pos[1] + 0.5
        return float(base_d + math.hypot(x - ex, y - ey) * 0.05)

    def is_wall(self, gx, gy):
        if gx < 0 or gx >= self.dim or gy < 0 or gy >= self.dim: return True
        return self.grid[int(gy)][int(gx)] == 1

    def bake_surface(self):
        self.baked_surface = pygame.Surface((MAZE_W, MAZE_H), pygame.SRCALPHA)
        self.baked_surface.fill((0, 0, 0, 0))
        for gy in range(self.dim):
            for gx in range(self.dim):
                if self.grid[gy][gx] == 1:
                    rect = (gx * CELL_SIZE, gy * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    pygame.draw.rect(self.baked_surface, (12, 18, 28), rect)
                    pygame.draw.rect(self.baked_surface, (25, 42, 60), rect, 1)
                elif self.grid[gy][gx] == 2:
                    cx = int((gx + 0.5) * CELL_SIZE)
                    cy = int((gy + 0.5) * CELL_SIZE)
                    pygame.draw.circle(self.baked_surface, (0, 255, 120, 50), (cx, cy), int(CELL_SIZE * 0.45))
                    pygame.draw.circle(self.baked_surface, (0, 255, 120), (cx, cy), int(CELL_SIZE * 0.28))
                    pygame.draw.circle(self.baked_surface, (255, 255, 255), (cx, cy), int(CELL_SIZE * 0.12))

# ==============================================================================
# 2. ФИЗИЧЕСКИЙ АВАТАР СО СКОЛЬЖЕНИЕМ
# ==============================================================================
class WallSlidingAvatar:
    def __init__(self):
        self.x, self.y = 1.5, 1.5
        self.vx, self.vy = 0.0, 0.0
        self.angle = -math.pi / 2.0
        self.persistence = 0.0
        self.last_ix, self.last_iy = 0.0, 0.0
        self.wm_turn_curve = 0.0
        self.temporal_bias = 0.0
        self.actual_dx = 0.0
        self.actual_dy = 0.0
        self.progress_delta = 0.0
        self.trail = []

    def update_physics(self, dt, force_x, force_y, wm_curvature, temp_bias, maze: TopoMaze):
        self.wm_turn_curve = wm_curvature
        self.temporal_bias = temp_bias

        mag = math.hypot(force_x, force_y)
        if mag > 0.05:
            last_mag = math.hypot(self.last_ix, self.last_iy) + 1e-6
            dot = (force_x * self.last_ix + force_y * self.last_iy) / (mag * last_mag)
            alignment = max(0.0, dot)
        else:
            alignment = 0.0

        self.persistence = self.persistence * 0.95 + 0.05 * alignment * math.tanh(mag * 2.0)
        self.last_ix, self.last_iy = force_x, force_y

        accel_multiplier = 1.0 + temp_bias
        active_boost = 1.0 + self.persistence * 4.0
        base_speed = 3.4 * active_boost * accel_multiplier

        target_vx = force_x * base_speed
        target_vy = force_y * base_speed

        MAX_SPEED = 9.0
        t_mag = math.hypot(target_vx, target_vy)
        if t_mag > MAX_SPEED:
            target_vx = (target_vx / t_mag) * MAX_SPEED
            target_vy = (target_vy / t_mag) * MAX_SPEED

        self.vx = self.vx * 0.86 + target_vx * 0.14
        self.vy = self.vy * 0.86 + target_vy * 0.14

        dist_before = maze.get_distance_at(self.x, self.y)

        move_dist = math.hypot(self.vx, self.vy) * dt
        steps = max(1, int(math.ceil(move_dist / 0.04)))
        sdx = (self.vx * dt) / steps
        sdy = (self.vy * dt) / steps
        r = 0.22

        old_x, old_y = self.x, self.y
        for _ in range(steps):
            if not maze.is_wall(self.x + sdx + math.copysign(r, sdx), self.y):
                self.x += sdx
            if not maze.is_wall(self.x, self.y + sdy + math.copysign(r, sdy)):
                self.y += sdy

        self.actual_dx = self.x - old_x
        self.actual_dy = self.y - old_y

        dist_after = maze.get_distance_at(self.x, self.y)
        self.progress_delta = dist_before - dist_after

        reached_exit = False
        if int(self.x) == maze.exit_pos[0] and int(self.y) == maze.exit_pos[1]:
            reached_exit = True
            maze.__init__(DIM)
            self.x, self.y = 1.5, 1.5

        self.trail.append((self.x, self.y))
        if len(self.trail) > 35: self.trail.pop(0)

        return reached_exit

# ==============================================================================
# 3. СЕНСОРНЫЙ МОДУЛЬ (SM) — ЗНАЕТ ТОЛЬКО ЛОКАЛЬНЫЕ СТЕНЫ
# ==============================================================================
class MazeSensorModule:
    """
    Сенсорный орган Monty: знает только то, что чувствует вокруг себя в радиусе 0.35 клеток.
    Никакого знания о выходе или глобальной карте у него нет!
    """
    def process(self, avatar: WallSlidingAvatar, maze: TopoMaze) -> Message:
        x, y = avatar.x, avatar.y
        r = 0.38
        
        # Локальное ощущение препятствий по 4 направлениям
        wall_n = float(maze.is_wall(x, y - r))
        wall_s = float(maze.is_wall(x, y + r))
        wall_e = float(maze.is_wall(x + r, y))
        wall_w = float(maze.is_wall(x - r, y))

        return Message(
            location=np.array([x, y, 0.0], dtype=np.float64),
            morphological_features={
                "pose_vectors": np.eye(3, dtype=np.float64),
                "pose_fully_defined": True,
                "on_object": 1.0,
            },
            non_morphological_features={
                "wall_north": wall_n,
                "wall_south": wall_s,
                "wall_east":  wall_e,
                "wall_west":  wall_w,
            },
            confidence=1.0,
            pass_message=True,
            sender_id="Local_SM",
            sender_type="SM",
            process_features_in_lm=True
        )

# ==============================================================================
# 4. КОГНИТИВНАЯ КАРТА И ГЕНЕРАТОР ЦЕЛИ (GSG) — ЗНАЕТ, ГДЕ ВЫХОД
# ==============================================================================
class HippocampalGoalGenerator:
    """
    Энторинально-гиппокампальная система: ведет глобальную аллоцентрическую карту.
    Именно она знает, где находится выход, и вычисляет идеальный вектор градиента к цели.
    """
    def generate_goal(self, avatar: WallSlidingAvatar, maze: TopoMaze) -> Goal:
        cur_d = maze.get_distance_at(avatar.x, avatar.y)
        
        # Пробные смещения в 4 стороны для нахождения направления скорейшего спуска
        eps = 0.4
        d_n = maze.get_distance_at(avatar.x, avatar.y - eps) if not maze.is_wall(avatar.x, avatar.y - eps) else 999.0
        d_s = maze.get_distance_at(avatar.x, avatar.y + eps) if not maze.is_wall(avatar.x, avatar.y + eps) else 999.0
        d_e = maze.get_distance_at(avatar.x + eps, avatar.y) if not maze.is_wall(avatar.x + eps, avatar.y) else 999.0
        d_w = maze.get_distance_at(avatar.x - eps, avatar.y) if not maze.is_wall(avatar.x - eps, avatar.y) else 999.0

        min_d = min(d_n, d_s, d_e, d_w)
        
        gx, gy = 0.0, 0.0
        if min_d < 900.0:
            if min_d == d_n: gy = -1.0
            elif min_d == d_s: gy = 1.0
            elif min_d == d_e: gx = 1.0
            elif min_d == d_w: gx = -1.0

        return Goal(
            location=np.array([maze.exit_pos[0] + 0.5, maze.exit_pos[1] + 0.5, 0.0], dtype=np.float64),
            morphological_features=None,
            non_morphological_features={
                "ideal_dir_x": gx,
                "ideal_dir_y": gy,
                "dist_to_exit": cur_d
            },
            confidence=float(np.clip(1.0 / (cur_d + 1.0), 0.05, 1.0)),
            pass_message=True,
            sender_id="Hippocampal_GSG",
            sender_type="GSG",
            process_features_in_lm=True,
            goal_tolerances=None
        )

# ==============================================================================
# 5. ИСПОЛНИТЕЛЬНАЯ КОЛОНКА MONTY (УЧИТСЯ У МОЗГА И УМЕЕТ РУЛИТЬ САМА)
# ==============================================================================
class MontyAutonomousExecutive:
    """
    Лобная колонка Monty:
    - Принимает сенсорику от SM (локальные стены) и цель от GSG (вектор к выходу).
    - В режиме УЧИТЕЛЯ (HUMAN): смотрит, как человек рулит через FCz, и учит политику.
    - В режиме АВТОПИЛОТА (MONTY): генерирует (force_x, force_y) самостоятельно!
    """
    def __init__(self):
        # Ассоциативная матрица политики: связывает [Стены 4 + Цель 2] -> [Действие 4]
        # 6 входных признаков -> 4 направления
        self.weights = torch.zeros((4, 6), device=DEVICE, dtype=torch.float32)
        
        # Статистика совпадений предсказаний Monty с намерениями человека
        self.match_history = deque(maxlen=150)
        self.autonomy_score = 0.0      # 0..100% готовность автопилота
        self.total_learned_steps = 0

        self.last_monty_fx = 0.0
        self.last_monty_fy = 0.0

    def get_state_vector(self, sm_msg: Message, gsg_goal: Goal) -> torch.Tensor:
        walls = sm_msg.non_morphological_features
        g_feat = gsg_goal.non_morphological_features
        # Вектор состояния: [Wall_N, Wall_S, Wall_E, Wall_W, Goal_Gx, Goal_Gy]
        s = [
            walls["wall_north"],
            walls["wall_south"],
            walls["wall_east"],
            walls["wall_west"],
            g_feat["ideal_dir_x"],
            g_feat["ideal_dir_y"]
        ]
        return torch.tensor(s, device=DEVICE, dtype=torch.float32)

    def predict_action(self, state_vec: torch.Tensor) -> tuple[float, float, int]:
        """Monty вычисляет собственное действие по выученным весам."""
        with torch.no_grad():
            scores = torch.mv(self.weights, state_vec)
            # Отсекаем направления, где стена
            if state_vec[0] > 0.5: scores[0] -= 5.0  # North
            if state_vec[1] > 0.5: scores[1] -= 5.0  # South
            if state_vec[2] > 0.5: scores[2] -= 5.0  # East
            if state_vec[3] > 0.5: scores[3] -= 5.0  # West

            best_dir = int(torch.argmax(scores).item())
            
            # Перевод дискретного направления в плавный вектор силы
            # 0: Вперед (-Y), 1: Назад (+Y), 2: Вправо (+X), 3: Влево (-X)
            dir_vectors = [
                (0.0, -1.0),
                (0.0,  1.0),
                (1.0,  0.0),
                (-1.0, 0.0)
            ]
            fx, fy = dir_vectors[best_dir]
            self.last_monty_fx = fx
            self.last_monty_fy = fy
            return fx, fy, best_dir

    def learn_from_human(self, state_vec: torch.Tensor, human_fx: float, human_fy: float, is_moving: bool):
        """Обучение на демонстрации человека с FCz."""
        if not is_moving:
            return 0.0, False

        # Определение направления человека
        if abs(human_fy) >= abs(human_fx):
            human_dir = 0 if human_fy < 0 else 1
        else:
            human_dir = 2 if human_fx > 0 else 3

        # Что предсказал Monty до обучения?
        _, _, monty_dir = self.predict_action(state_vec)
        is_match = (monty_dir == human_dir)
        self.match_history.append(1.0 if is_match else 0.0)

        # Обучение на аномалиях: если Monty не угадал — обновляем синапсы
        if not is_match:
            lr = 0.05
            # Укрепляем связь текущего состояния со сделанным человеком выбором
            self.weights[human_dir] += lr * state_vec
            # Ослабляем ошибочно предсказанный вариант
            self.weights[monty_dir] -= lr * 0.5 * state_vec
            self.weights = torch.clamp(self.weights, -1.0, 3.0)
            self.total_learned_steps += 1

        # Расчет честного индекса автономии (0% на старте)
        if len(self.match_history) >= 15:
            acc = float(np.mean(self.match_history))
            maturity = min(1.0, self.total_learned_steps / 80.0)
            self.autonomy_score = acc * maturity * 100.0
        else:
            self.autonomy_score = 0.0

        return self.autonomy_score, is_match

# ==============================================================================
# 6. ОСНОВНОЙ ЦИКЛ ПРИЛОЖЕНИЯ
# ==============================================================================
def main():
    import multiprocessing as mp
    mp.freeze_support()

    pygame.init()
    flags = pygame.HWSURFACE | pygame.DOUBLEBUF
    screen = pygame.display.set_mode((1380, 760), flags, vsync=0)
    pygame.display.set_caption("NeuroCanvas × Monty: Human BCI vs Autonomous Pilot [SPACE to Toggle]")
    clock = pygame.time.Clock()

    engine = HeterarchicalBrainEngine()
    engine.start()

    maze = TopoMaze(DIM)
    avatar = WallSlidingAvatar()

    # Раздельные системы по Monty:
    sm = MazeSensorModule()
    gsg = HippocampalGoalGenerator()
    monty = MontyAutonomousExecutive()

    N_elements = 2
    lead_idx = 0
    sample_indices = np.linspace(0, NUM_FREQS - 1, N_elements, dtype=int)

    # РЕЖИМ УПРАВЛЕНИЯ: False = Человек с FCz, True = Автопилот Monty!
    AUTONOMOUS_MONTY_MODE = False

    try:
        while True:
            dt = clock.tick(60) / 1000.0
            dt = min(0.05, dt)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        AUTONOMOUS_MONTY_MODE = not AUTONOMOUS_MONTY_MODE
                        mode_tag = "АВТОПИЛОТ MONTY" if AUTONOMOUS_MONTY_MODE else "УПРАВЛЕНИЕ МОЗГОМ (FCz)"
                        print(f"🔄 [РЕЖИМ ПЕРЕКЛЮЧЕН]: {mode_tag}")
                    elif pygame.K_1 <= event.key <= pygame.K_4:
                        lead_idx = event.key - pygame.K_1
                    elif event.key == pygame.K_r:
                        maze = TopoMaze(DIM)
                        avatar.x, avatar.y = 1.5, 1.5

            frame = engine.get_frame()
            nodes = [frame.fcz_macro, frame.pz_spatial, frame.oz_sensory, frame.cz_motor]
            lead_node = nodes[lead_idx]

            # Вектор человека с FCz
            axes = lead_node.gamepad_axes
            human_fx = axes.lx
            human_fy = -axes.ly
            wm_curvature = axes.rx
            temporal_bias = axes.ry

            # 1. Сенсорный модуль считывает стены
            sm_msg = sm.process(avatar, maze)

            # 2. Гиппокамп выдает цель и направление к выходу
            gsg_goal = gsg.generate_goal(avatar, maze)

            # 3. Monty формирует вектор состояния и предсказывает действие
            s_vec = monty.get_state_vector(sm_msg, gsg_goal)
            monty_fx, monty_fy, monty_pred_dir = monty.predict_action(s_vec)

            # 4. ВЫБОР ДРАЙВЕРА: Человек или Автопилот Monty?
            if AUTONOMOUS_MONTY_MODE:
                # Рулит Monty!
                drive_fx = monty_fx
                drive_fy = monty_fy
            else:
                # Рулит Человек с FCz!
                drive_fx = human_fx
                drive_fy = human_fy

            # Шаг физики в лабиринте
            solved = avatar.update_physics(dt, drive_fx, drive_fy, wm_curvature, temporal_bias, maze)

            # Обучение Monty на действиях человека (если рулит человек)
            step_len = math.hypot(avatar.actual_dx, avatar.actual_dy)
            is_moving = (step_len >= 0.001)
            
            if not AUTONOMOUS_MONTY_MODE:
                autonomy_ready, matched = monty.learn_from_human(s_vec, human_fx, human_fy, is_moving)
            else:
                autonomy_ready = monty.autonomy_score
                matched = True

            # ==================================================================
            # ОТРИСОВКА
            # ==================================================================
            screen.fill((6, 8, 12))
            ox, oy = 30, 45

            maze_surface = pygame.Surface((MAZE_W, MAZE_H), pygame.SRCALPHA)
            maze_surface.fill((0, 0, 0, 0))
            maze_surface.blit(maze.baked_surface, (0, 0))

            px = int(avatar.x * CELL_SIZE)
            py = int(avatar.y * CELL_SIZE)

            if len(avatar.trail) > 1:
                trail_pts = [(int(tx * CELL_SIZE), int(ty * CELL_SIZE)) for tx, ty in avatar.trail]
                t_col = (255, 80, 200) if AUTONOMOUS_MONTY_MODE else (0, 180, 255)
                pygame.draw.lines(maze_surface, t_col, False, trail_pts, 2)

            # -------------------------------------------------------------
            # 1. СТРЕЛКА ЧЕЛОВЕКА (ЖЕЛТАЯ / БИРЮЗОВАЯ ИЗ FCz)
            # -------------------------------------------------------------
            VEC_SCALE = 120.0
            spline_pts = [(px, py)]
            base_gx = lead_node.traj_32[0, 0]
            base_gy = lead_node.traj_32[0, 1]
            for idx_k in sample_indices[1:]:
                sk_x = lead_node.traj_32[idx_k, 0] - base_gx
                sk_y = lead_node.traj_32[idx_k, 1] - base_gy
                sk_len = math.hypot(sk_x, sk_y)
                if sk_len > 1.0: sk_x /= sk_len; sk_y /= sk_len
                sp_x = px + sk_x * VEC_SCALE
                sp_y = py + sk_y * VEC_SCALE
                spline_pts.append((int(sp_x), int(sp_y)))

            # Вектор человека
            pygame.draw.line(maze_surface, (0, 255, 255), spline_pts[0], spline_pts[1], 4)
            pygame.draw.circle(maze_surface, (255, 255, 255), spline_pts[1], 5)

            # -------------------------------------------------------------
            # 2. СТРЕЛКА MONTY (ПУРПУРНАЯ / САЛАТОВАЯ — АВТОПИЛОТ)
            # -------------------------------------------------------------
            m_end_x = px + int(monty_fx * 70.0)
            m_end_y = py + int(monty_fy * 70.0)
            pygame.draw.line(maze_surface, (255, 60, 180), (px, py), (m_end_x, m_end_y), 3)
            pygame.draw.circle(maze_surface, (255, 120, 220), (m_end_x, m_end_y), 4)

            # Аватар
            av_col = (255, 50, 200) if AUTONOMOUS_MONTY_MODE else (0, 255, 200)
            pygame.draw.circle(maze_surface, av_col, (px, py), 13)
            pygame.draw.circle(maze_surface, (255, 255, 255), (px, py), 4)

            screen.blit(maze_surface, (ox, oy))
            border_col = (255, 50, 200) if AUTONOMOUS_MONTY_MODE else (0, 200, 255)
            pygame.draw.rect(screen, border_col, (ox, oy, MAZE_W, MAZE_H), 2)

            # ==================================================================
            # ПАНЕЛЬ УПРАВЛЕНИЯ И АВТОНОМИИ MONTY (ПРАВАЯ СТОРОНА)
            # ==================================================================
            rx = 690
            f_b = pygame.font.SysFont("consolas", 13, bold=True)
            f_s = pygame.font.SysFont("consolas", 11)
            f_huge = pygame.font.SysFont("consolas", 26, bold=True)

            pygame.draw.rect(screen, (12, 16, 24), (rx, 45, 650, 225), border_radius=6)
            mode_col = (255, 60, 180) if AUTONOMOUS_MONTY_MODE else (0, 255, 200)
            pygame.draw.rect(screen, mode_col, (rx, 45, 650, 225), 2, border_radius=6)

            mode_title = "► АВТОПИЛОТ MONTY [АКТИВЕН] (РУЛИТ MONTY)" if AUTONOMOUS_MONTY_MODE else "► УЧИТЕЛЬ: МОЗГ ЧЕЛОВЕКА [FCz BCI] (MONTY УЧИТСЯ)"
            screen.blit(f_b.render(mode_title, True, mode_col), (rx + 15, 55))
            screen.blit(f_huge.render(f"ГОТОВНОСТЬ АВТОНОМИИ: {monty.autonomy_score:.1f}%", True, mode_col), (rx + 15, 80))

            # Шкала готовности к передаче руля
            bar_w = 400
            pygame.draw.rect(screen, (25, 35, 45), (rx + 15, 120, bar_w, 16), border_radius=4)
            fill_w = int(np.clip(monty.autonomy_score / 100.0, 0.0, 1.0) * bar_w)
            pygame.draw.rect(screen, mode_col, (rx + 15, 120, fill_w, 16), border_radius=4)

            screen.blit(f_s.render(f"• [ПРОБЕЛ]: ПЕРЕКЛЮЧИТЬ УПРАВЛЕНИЕ (Человек <-> Monty)", True, (255, 255, 100)), (rx + 15, 145))
            screen.blit(f_s.render(f"• Бирюзовая стрелка: Намерение мозга человека (FCz)", True, (0, 255, 255)), (rx + 15, 165))
            screen.blit(f_s.render(f"• Пурпурная стрелка: Решение автопилота Monty (Frontal LM)", True, (255, 60, 180)), (rx + 15, 185))
            screen.blit(f_s.render(f"• Совпадение предсказаний: {float(np.mean(monty.match_history))*100.0 if monty.match_history else 0.0:.1f}% | Шагов обучения: {monty.total_learned_steps}", True, (180, 200, 220)), (rx + 15, 205))
            screen.blit(f_s.render(f"• Сенсорный модуль SM: чувствует только стены! Цель ведет GSG.", True, (140, 150, 160)), (rx + 15, 225))

            # Блок телеметрии систем
            pygame.draw.rect(screen, (12, 16, 24), (rx, 285, 650, 85), border_radius=6)
            pygame.draw.rect(screen, (35, 45, 60), (rx, 285, 650, 85), 1, border_radius=6)

            walls = sm_msg.non_morphological_features
            w_str = f"N:{int(walls['wall_north'])} S:{int(walls['wall_south'])} E:{int(walls['wall_east'])} W:{int(walls['wall_west'])}"
            screen.blit(f_b.render(f"СЕНСОРНЫЙ МОДУЛЬ (SM): Локальные стены вокруг: [{w_str}]", True, (0, 255, 200)), (rx + 15, 295))
            
            g_feat = gsg_goal.non_morphological_features
            screen.blit(f_s.render(f"• Гиппокамп (GSG): Дистанция до выхода: {g_feat['dist_to_exit']:.1f} шагов | Градиент: ({g_feat['ideal_dir_x']:+.1f}, {g_feat['ideal_dir_y']:+.1f})", True, (255, 200, 100)), (rx + 15, 320))
            screen.blit(f_s.render(f"• Исполнительная колонка (LM): Предсказывает: [{monty.class_names[monty_pred_dir]}]", True, (255, 100, 200)), (rx + 15, 340))

            # Радары узлов
            colors = [(255, 50, 200), (0, 200, 255), (100, 255, 100), (255, 180, 0)]
            for i, n in enumerate(nodes):
                ry_ui = 385 + i * 90
                col = colors[i]
                is_lead = (i == lead_idx)

                pygame.draw.rect(screen, (12, 16, 22), (rx, ry_ui, 650, 82), border_radius=6)
                pygame.draw.rect(screen, col if is_lead else (45, 55, 65), (rx, ry_ui, 650, 82), 2 if is_lead else 1, border_radius=6)

                role_label = f"[{i+1}] {n.name}" + (" [ВЕДУЩИЙ ◄]" if is_lead else "")
                screen.blit(f_b.render(role_label, True, col if is_lead else (130, 140, 150)), (rx + 15, ry_ui + 10))

                ax_node = n.gamepad_axes
                screen.blit(f_s.render(f"Оси: Lx={ax_node.lx:+.2f}, Ly={-ax_node.ly:+.2f} | rx={ax_node.rx:+.2f}", True, (160, 170, 180)), (rx + 15, ry_ui + 32))
                screen.blit(f_s.render(f"Вихрь Tq: {n.tq:+.2f} | Тяга: {n.thrust:.2f} | Тета-часы: {frame.theta_freq:.2f} Гц", True, (160, 170, 180)), (rx + 15, ry_ui + 52))

                cx_radar, cy_radar, rc = rx + 570, ry_ui + 41, 30
                pygame.draw.circle(screen, (20, 28, 38), (cx_radar, cy_radar), rc, 1)
                pygame.draw.line(screen, (30, 40, 50), (cx_radar - rc, cy_radar), (cx_radar + rc, cy_radar), 1)
                pygame.draw.line(screen, (30, 40, 50), (cx_radar, cy_radar - rc), (cx_radar, cy_radar + rc), 1)

                nbx, nby = ax_node.lx, -ax_node.ly
                n_len = math.hypot(nbx, nby)
                if n_len > 1.0: nbx /= n_len; nby /= n_len
                pygame.draw.line(screen, (255, 220, 0), (cx_radar, cy_radar), (cx_radar + int(nbx * (rc - 4)), cy_radar + int(nby * (rc - 4))), 2)
                pygame.draw.circle(screen, (255, 255, 255), (cx_radar + int(nbx * (rc - 4)), cy_radar + int(nby * (rc - 4))), 3)

            fps_val = clock.get_fps()
            screen.blit(f_b.render(f"NEUROCANVAS × MONTY | SENSOR/GOAL SEPARATION & AUTONOMY | {fps_val:.0f} FPS", True, (255, 255, 255)), (30, 15))

            pygame.display.flip()

    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
        pygame.quit()

if __name__ == '__main__':
    main()
