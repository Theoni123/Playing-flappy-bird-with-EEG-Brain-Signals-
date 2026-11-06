"""Flappy Bird controlled by blinks (or the spacebar)."""

import queue
import random

import numpy as np
import pygame

WIDTH, PLAY_HEIGHT, PANEL_HEIGHT = 480, 600, 140
HEIGHT = PLAY_HEIGHT + PANEL_HEIGHT
FPS = 60
GROUND_HEIGHT = 60

# Slower and more forgiving than the original to absorb blink latency.
GRAVITY = 0.30
FLAP_VELOCITY = -6.5
MAX_FALL = 8.0
PIPE_SPEED = 2.0
PIPE_GAP = 210
PIPE_WIDTH = 70
PIPE_SPACING = 280
BIRD_X = 120
BIRD_RADIUS = 16
RESTART_DELAY_MS = 1000

SKY = (112, 197, 206)
GROUND = (222, 216, 149)
GRASS = (92, 184, 72)
PIPE = (96, 184, 64)
PIPE_DARK = (60, 130, 40)
BIRD = (250, 210, 50)
PANEL_BG = (24, 26, 33)
TRACE = (90, 220, 255)
THRESH = (255, 90, 90)
TEXT = (255, 255, 255)


class Pipe:
    def __init__(self, x, gap=PIPE_GAP):
        self.x = x
        self.gap = gap
        margin = 50
        self.gap_y = random.randint(margin + gap // 2,
                                    PLAY_HEIGHT - GROUND_HEIGHT - margin - gap // 2)
        self.scored = False

    def rects(self):
        top = pygame.Rect(self.x, 0, PIPE_WIDTH, self.gap_y - self.gap // 2)
        bottom_y = self.gap_y + self.gap // 2
        bottom = pygame.Rect(self.x, bottom_y, PIPE_WIDTH, PLAY_HEIGHT - GROUND_HEIGHT - bottom_y)
        return top, bottom


class FlappyGame:
    def __init__(self, reader=None, mock=None, speed=1.0):
        """`reader` is a BlinkReader (or None for keyboard only); `mock` is a
        MockStream whose blinks can be injected with the B key. `speed` scales
        time, so flaps keep the same height but everything happens slower."""
        self.reader = reader
        self.mock = mock
        nod = reader is not None and getattr(reader, "control", None) == "nod"
        # Nod mode: one nod carries farther, pipes are slower and more open,
        # so you need fewer nods to stay alive.
        hop = 0.85 if nod else 1.0
        fall = 0.62 if nod else 1.0
        self.gravity = GRAVITY * speed ** 2 * fall
        self.flap_velocity = FLAP_VELOCITY * speed * hop
        self.max_fall = MAX_FALL * speed * (0.7 if nod else 1.0)
        self.pipe_speed = PIPE_SPEED * speed * (0.62 if nod else 1.0)
        self.pipe_gap = 300 if nod else PIPE_GAP
        self.pipe_spacing = 400 if nod else PIPE_SPACING
        pygame.init()
        pygame.display.set_caption("Blink Bird")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Helvetica", 22, bold=True)
        self.big_font = pygame.font.SysFont("Helvetica", 56, bold=True)
        self.small_font = pygame.font.SysFont("Menlo", 13)
        self.best = 0
        self.blink_flash = 0
        self.blink_count = 0
        self.reset()

    def reset(self):
        self.state = "ready"
        self.bird_y = PLAY_HEIGHT / 2 - 40
        self.velocity = 0.0
        self.pipes = [Pipe(WIDTH + 100 + i * self.pipe_spacing, self.pipe_gap) for i in range(3)]
        self.score = 0
        self.dead_at = 0
        self.ground_offset = 0

    # --- input ---------------------------------------------------------------

    def flap(self):
        if self.state == "ready":
            self.state = "playing"
            self.velocity = self.flap_velocity
        elif self.state == "playing":
            self.velocity = self.flap_velocity
        elif self.state == "dead" and pygame.time.get_ticks() - self.dead_at > RESTART_DELAY_MS:
            self.reset()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False
                if event.key in (pygame.K_SPACE, pygame.K_UP):
                    self.flap()
                if event.key == pygame.K_b and self.mock:
                    self.mock.blink()
        if self.reader:
            while True:
                try:
                    self.reader.flaps.get_nowait()
                except queue.Empty:
                    break
                self.blink_flash = 12
                self.blink_count += 1
                self.flap()
        return True

    # --- simulation ----------------------------------------------------------

    def update(self):
        if self.blink_flash:
            self.blink_flash -= 1
        if self.state == "ready":
            self.bird_y = PLAY_HEIGHT / 2 - 40 + 8 * np.sin(pygame.time.get_ticks() / 250)
            self.ground_offset = (self.ground_offset + self.pipe_speed) % 24
            return
        if self.state == "dead":
            if self.bird_y < PLAY_HEIGHT - GROUND_HEIGHT - BIRD_RADIUS:
                self.velocity = min(self.velocity + self.gravity, self.max_fall)
                self.bird_y += self.velocity
            return

        self.velocity = min(self.velocity + self.gravity, self.max_fall)
        self.bird_y += self.velocity
        if self.bird_y < BIRD_RADIUS:
            self.bird_y = BIRD_RADIUS
            # Start falling instead of sticking to the ceiling, so the next nod
            # can be seen and counted.
            if self.velocity < 0:
                self.velocity = self.gravity * 10
        self.ground_offset = (self.ground_offset + self.pipe_speed) % 24

        for pipe in self.pipes:
            pipe.x -= self.pipe_speed
        if self.pipes[0].x < -PIPE_WIDTH:
            self.pipes.pop(0)
            self.pipes.append(Pipe(self.pipes[-1].x + self.pipe_spacing, self.pipe_gap))

        bird = pygame.Rect(0, 0, BIRD_RADIUS * 2 - 6, BIRD_RADIUS * 2 - 6)
        bird.center = (BIRD_X, int(self.bird_y))
        for pipe in self.pipes:
            if any(bird.colliderect(r) for r in pipe.rects()):
                self.die()
            if not pipe.scored and pipe.x + PIPE_WIDTH < BIRD_X:
                pipe.scored = True
                self.score += 1
        if self.bird_y + BIRD_RADIUS >= PLAY_HEIGHT - GROUND_HEIGHT:
            self.die()

    def die(self):
        if self.state != "dead":
            self.state = "dead"
            self.dead_at = pygame.time.get_ticks()
            self.best = max(self.best, self.score)

    # --- drawing -------------------------------------------------------------

    def draw_text(self, text, font, center, color=TEXT):
        shadow = font.render(text, True, (0, 0, 0))
        surf = font.render(text, True, color)
        rect = surf.get_rect(center=center)
        self.screen.blit(shadow, rect.move(2, 2))
        self.screen.blit(surf, rect)

    def draw_bird(self):
        x, y = BIRD_X, int(self.bird_y)
        angle = max(-30, min(70, self.velocity * 6))
        bird = pygame.Surface((BIRD_RADIUS * 3, BIRD_RADIUS * 3), pygame.SRCALPHA)
        c = BIRD_RADIUS * 3 // 2
        pygame.draw.circle(bird, BIRD, (c, c), BIRD_RADIUS)
        pygame.draw.circle(bird, (0, 0, 0), (c, c), BIRD_RADIUS, 2)
        pygame.draw.ellipse(bird, (255, 240, 160), (c - 14, c, 16, 10))
        pygame.draw.circle(bird, TEXT, (c + 6, c - 5), 6)
        pygame.draw.circle(bird, (0, 0, 0), (c + 8, c - 5), 3)
        pygame.draw.polygon(bird, (245, 120, 40), [(c + 12, c), (c + 24, c + 3), (c + 12, c + 7)])
        rotated = pygame.transform.rotate(bird, -angle)
        self.screen.blit(rotated, rotated.get_rect(center=(x, y)))

    def draw_world(self):
        self.screen.fill(SKY, (0, 0, WIDTH, PLAY_HEIGHT))
        for pipe in self.pipes:
            for rect in pipe.rects():
                pygame.draw.rect(self.screen, PIPE, rect)
                pygame.draw.rect(self.screen, PIPE_DARK, rect, 3)
            top, bottom = pipe.rects()
            for lip_y in (top.bottom - 24, bottom.top):
                lip = pygame.Rect(pipe.x - 5, lip_y, PIPE_WIDTH + 10, 24)
                pygame.draw.rect(self.screen, PIPE, lip)
                pygame.draw.rect(self.screen, PIPE_DARK, lip, 3)

        ground_y = PLAY_HEIGHT - GROUND_HEIGHT
        self.screen.fill(GROUND, (0, ground_y, WIDTH, GROUND_HEIGHT))
        self.screen.fill(GRASS, (0, ground_y, WIDTH, 12))
        for x in range(-24, WIDTH + 24, 24):
            gx = x - int(self.ground_offset)
            pygame.draw.line(self.screen, PIPE_DARK, (gx, ground_y + 12), (gx + 12, ground_y), 3)

        self.draw_bird()

        if self.state == "playing" or self.state == "dead":
            self.draw_text(str(self.score), self.big_font, (WIDTH // 2, 70))
        if self.state == "ready":
            self.draw_text("BLINK BIRD", self.big_font, (WIDTH // 2, 140))
            action = "Nod" if self.reader and self.reader.control == "nod" else "Blink"
            self.draw_text(f"{action} (or press space) to fly", self.font, (WIDTH // 2, 380))
            if self.mock:
                self.draw_text("Mock mode: press B to fake a blink", self.font, (WIDTH // 2, 410))
        if self.state == "dead":
            self.draw_text("GAME OVER", self.big_font, (WIDTH // 2, 200))
            self.draw_text(f"Score {self.score}   Best {self.best}", self.font, (WIDTH // 2, 260))
            if pygame.time.get_ticks() - self.dead_at > RESTART_DELAY_MS:
                action = "Nod" if self.reader and self.reader.control == "nod" else "Blink"
                self.draw_text(f"{action} to play again", self.font, (WIDTH // 2, 300))

    def draw_panel(self):
        top = PLAY_HEIGHT
        panel = pygame.Rect(0, top, WIDTH, PANEL_HEIGHT)
        self.screen.fill(PANEL_BG, panel)
        if self.blink_flash:
            pygame.draw.rect(self.screen, (255, 220, 60), panel, 4)

        if not self.reader:
            self.screen.blit(self.small_font.render("keyboard only (no EEG stream)", True, TEXT),
                             (10, top + 8))
            return

        self.screen.blit(self.small_font.render(self.reader.status, True, TEXT), (10, top + 6))
        kind = "nods" if self.reader.control == "nod" else "blinks"
        self.screen.blit(
            self.small_font.render(f"{kind} detected: {self.blink_count}", True, TEXT),
            (10, top + PANEL_HEIGHT - 20),
        )

        plot = pygame.Rect(10, top + 26, WIDTH - 20, PANEL_HEIGHT - 50)
        pygame.draw.rect(self.screen, (40, 44, 55), plot)
        trace = self.reader.trace()
        threshold = self.reader.threshold
        scale = max(threshold * 1.6, float(trace.max()) if trace.size else 0.0, 1.0)

        def to_y(value):
            return int(plot.bottom - value / scale * (plot.height - 4))

        ty = to_y(threshold)
        pygame.draw.line(self.screen, THRESH, (plot.left, ty), (plot.right, ty), 1)
        self.screen.blit(self.small_font.render(f"threshold {threshold:.0f}", True, THRESH),
                         (WIDTH - 150, top + PANEL_HEIGHT - 20))

        if trace.size > 2:
            step = max(1, trace.size // plot.width)
            values = trace[::step]
            n = values.size
            points = [(plot.left + int(i * plot.width / n), to_y(v)) for i, v in enumerate(values)]
            pygame.draw.lines(self.screen, TRACE, False, points, 2)

    def run(self):
        running = True
        while running:
            running = self.handle_events()
            self.update()
            self.draw_world()
            self.draw_panel()
            pygame.display.flip()
            self.clock.tick(FPS)
        pygame.quit()
