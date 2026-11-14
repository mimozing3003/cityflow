"""
TrajPilot CityFlow - Stage 07: Live Pygame HUD & Visual Demonstration Dashboard
Renders real-time RGB camera feed, 30-frame predicted trajectory overlay,
telemetry stats, 16-channel feature audit panel, and mandated scientific disclaimers.
"""

import sys
import pygame
import numpy as np

DISCLAIMER_HEADER = "CARLA DEMONSTRATION — FEATURE EQUIVALENCE NOT PROVEN"
DISCLAIMER_TEXT = (
    "Frozen I-24 GRU inference is demonstrated in CARLA using available/derived features. "
    "This demonstration does not establish cross-domain generalization or exact I-24 feature equivalence."
)


class CarlaVisualizer:
    def __init__(self, width=1280, height=720, title="TRAJPILOT CITYFLOW | CARLA 0.9.16"):
        pygame.init()
        pygame.font.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption(title)
        
        # Fonts
        self.font_large = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_main = pygame.font.SysFont("Consolas", 14, bold=False)
        self.font_bold = pygame.font.SysFont("Consolas", 14, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 11, bold=False)

        # Palette
        self.c_bg = (15, 18, 22)
        self.c_panel = (24, 28, 36)
        self.c_border = (45, 55, 72)
        self.c_accent = (0, 210, 255)
        self.c_pred = (255, 60, 100)
        self.c_past = (255, 200, 40)
        self.c_text = (240, 240, 245)
        self.c_dim = (140, 150, 165)
        self.c_good = (40, 220, 120)
        self.c_warn = (255, 180, 50)
        self.c_bad = (255, 80, 80)

    def render_frame(
        self,
        rgb_image: np.ndarray,
        pred_screen_points: list,
        past_screen_points: list,
        telemetry: dict,
        features_metadata: dict
    ):
        """
        Renders full visual frame with HUD overlay.
        """
        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False
        
        # 1. Base Camera Surface
        if rgb_image is not None:
            # Transpose (H, W, 3) to Pygame surface (W, H)
            surface = pygame.surfarray.make_surface(np.transpose(rgb_image, (1, 0, 2)))
            if surface.get_width() != self.width or surface.get_height() != self.height:
                surface = pygame.transform.scale(surface, (self.width, self.height))
            self.screen.blit(surface, (0, 0))
        else:
            self.screen.fill(self.c_bg)

        # 2. Draw Past Trajectory Breadcrumbs
        if past_screen_points and len(past_screen_points) > 1:
            for pt in past_screen_points:
                pygame.draw.circle(self.screen, self.c_past, pt, 3)
            pygame.draw.lines(self.screen, self.c_past, False, past_screen_points, 2)

        # 3. Draw Predicted 30-Step Trajectory
        if pred_screen_points and len(pred_screen_points) > 1:
            # Trajectory glow and ribbon
            pygame.draw.lines(self.screen, (255, 100, 140), False, pred_screen_points, 4)
            for i, pt in enumerate(pred_screen_points):
                r = 4 if i % 5 == 0 else 2
                color = (255, 255, 255) if i == len(pred_screen_points) - 1 else self.c_pred
                pygame.draw.circle(self.screen, color, pt, r)

        # 4. Top Header & Mandated Scientific Disclaimer
        self._draw_header()

        # 5. Left Telemetry HUD Panel
        self._draw_telemetry_panel(telemetry)

        # 6. Right Feature Status & Missingness Panel
        self._draw_features_panel(features_metadata)

        pygame.display.flip()
        return True

    def _draw_header(self):
        # Translucent top bar
        bar_surf = pygame.Surface((self.width, 68), pygame.SRCALPHA)
        bar_surf.fill((12, 16, 24, 230))
        self.screen.blit(bar_surf, (0, 0))
        pygame.draw.line(self.screen, self.c_border, (0, 68), (self.width, 68), 1)

        # Title
        t_title = self.font_large.render("TRAJPILOT CITYFLOW  |  CARLA 0.9.16 LIVE PREDICTION", True, self.c_accent)
        self.screen.blit(t_title, (16, 8))

        # Scientific Disclaimer Header
        t_warn_head = self.font_bold.render(f"[!] {DISCLAIMER_HEADER}", True, self.c_warn)
        self.screen.blit(t_warn_head, (16, 30))

        # Full disclaimer text
        t_warn_sub = self.font_small.render(DISCLAIMER_TEXT, True, self.c_dim)
        self.screen.blit(t_warn_sub, (16, 48))

    def _draw_telemetry_panel(self, telem):
        p_w, p_h = 290, 240
        x, y = 16, 80
        panel = pygame.Surface((p_w, p_h), pygame.SRCALPHA)
        panel.fill((20, 24, 32, 220))
        self.screen.blit(panel, (x, y))
        pygame.draw.rect(self.screen, self.c_border, (x, y, p_w, p_h), 1)

        # Header
        t_head = self.font_bold.render("SYSTEM & TELEMETRY", True, self.c_accent)
        self.screen.blit(t_head, (x + 12, y + 10))
        pygame.draw.line(self.screen, self.c_border, (x + 12, y + 28), (x + p_w - 12, y + 28), 1)

        items = [
            ("Simulator FPS", f"{telem.get('fps', 0.0):.1f} Hz (10 FPS target)"),
            ("Inference Latency", f"{telem.get('latency_ms', 0.0):.2f} ms"),
            ("Compute Device", f"{telem.get('device', 'CUDA')}"),
            ("Model Architecture", "GRU (2 layers, 128 hid)"),
            ("Model Parameters", "227,132 (Frozen I-24)"),
            ("History Horizon", "30 frames (3.0s)"),
            ("Future Horizon", "30 frames (3.0s)"),
            ("Ego Speed", f"{telem.get('speed', 0.0) * 3.6:.1f} km/h ({telem.get('speed', 0.0):.1f} m/s)"),
            ("Scene Vehicles", f"{telem.get('num_vehicles', 0)} active")
        ]

        curr_y = y + 36
        for label, val in items:
            lbl_surf = self.font_small.render(f"{label}:", True, self.c_dim)
            val_surf = self.font_small.render(val, True, self.c_text)
            self.screen.blit(lbl_surf, (x + 12, curr_y))
            self.screen.blit(val_surf, (x + 145, curr_y))
            curr_y += 21

    def _draw_features_panel(self, meta_dict):
        p_w, p_h = 360, 440
        x, y = self.width - p_w - 16, 80
        panel = pygame.Surface((p_w, p_h), pygame.SRCALPHA)
        panel.fill((20, 24, 32, 220))
        self.screen.blit(panel, (x, y))
        pygame.draw.rect(self.screen, self.c_border, (x, y, p_w, p_h), 1)

        # Calculate availability
        total = 16
        avail = sum(1 for m in meta_dict.values() if m.available) if meta_dict else 0

        t_head = self.font_bold.render("16 CANONICAL FEATURES AUDIT", True, self.c_accent)
        t_sub = self.font_small.render(f"Availability: {avail}/{total} Active Channels", True, self.c_good if avail == total else self.c_warn)
        self.screen.blit(t_head, (x + 12, y + 10))
        self.screen.blit(t_sub, (x + p_w - 180, y + 12))
        pygame.draw.line(self.screen, self.c_border, (x + 12, y + 28), (x + p_w - 12, y + 28), 1)

        curr_y = y + 34
        for name, meta in (meta_dict.items() if meta_dict else []):
            if meta.available:
                status_color = self.c_good if meta.exact_or_approximate == "NATIVE" else self.c_warn
                status_text = meta.exact_or_approximate
            else:
                status_color = self.c_bad
                status_text = "MISSING"

            t_name = self.font_small.render(name, True, self.c_text)
            t_status = self.font_small.render(f"[{status_text}]", True, status_color)
            t_src = self.font_small.render(meta.source[:20], True, self.c_dim)

            self.screen.blit(t_name, (x + 12, curr_y))
            self.screen.blit(t_status, (x + 180, curr_y))
            self.screen.blit(t_src, (x + 270, curr_y))
            curr_y += 24

        # Footer explanation
        curr_y = max(curr_y, y + p_h - 40)
        note = self.font_small.render("* Missing channels normalized to 0, mask=1.0", True, self.c_dim)
        self.screen.blit(note, (x + 12, curr_y))

    def close(self):
        pygame.quit()
