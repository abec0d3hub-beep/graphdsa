"""Pygame graph pathfinding visualizer for the bundled Amsterdam map."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import json
import math
from pathlib import Path
import sys
from typing import Iterator

import pygame


BASE_DIR = Path(__file__).resolve().parent
GRAPH_PATH = BASE_DIR / "graph.json"
MAP_PATH = BASE_DIR / "map.png"

HEADER_HEIGHT = 142
FPS = 60

ALGORITHMS = (
    "Uniform Cost Search (UCS)",
    "Greedy Best-First Search",
    "A* Search",
)

COLORS = {
    "background": (20, 26, 37),
    "panel": (30, 39, 54),
    "panel_border": (58, 72, 94),
    "text": (238, 243, 250),
    "muted": (163, 177, 197),
    "road_outline": (32, 42, 56),
    "road": (250, 252, 255),
    "edge_label": (24, 32, 44),
    "node": (70, 88, 220),
    "open": (35, 181, 240),
    "explored": (255, 160, 55),
    "start": (39, 201, 112),
    "destination": (239, 74, 91),
    "search_edge": (38, 190, 255),
    "search_runner": (0, 105, 190),
    "path": (255, 68, 94),
    "runner": (255, 225, 60),
    "button": (55, 75, 104),
    "button_hover": (74, 101, 139),
    "button_active": (39, 121, 178),
}

EdgeKey = tuple[str, str]
Point = tuple[float, float]


def edge_key(a: str, b: str) -> EdgeKey:
    return tuple(sorted((a, b)))


@dataclass(frozen=True)
class SearchStep:
    """One completed graph expansion for the exploration animation."""

    current: str
    open_nodes: frozenset[str]
    explored_nodes: frozenset[str]
    evaluated_edges: frozenset[EdgeKey]
    scanned_edges: tuple[tuple[str, str], ...]
    path: tuple[str, ...] = ()
    cost: float = math.inf


class Graph:
    """Graph data and Euclidean edge weights, independent of rendering."""

    def __init__(self, vertices: dict[str, Point], edges: list[EdgeKey]) -> None:
        self.vertices = vertices
        self.edges = edges
        self.adjacency: dict[str, list[tuple[str, float]]] = {
            node: [] for node in vertices
        }
        self.edge_weights: dict[EdgeKey, float] = {}

        for a, b in edges:
            if a not in vertices or b not in vertices:
                raise ValueError(f"Edge references an unknown vertex: {a}, {b}")
            x1, y1 = vertices[a]
            x2, y2 = vertices[b]
            weight = math.hypot(x2 - x1, y2 - y1)
            self.adjacency[a].append((b, weight))
            self.adjacency[b].append((a, weight))
            self.edge_weights[edge_key(a, b)] = weight

    @classmethod
    def load(cls, path: Path) -> Graph:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        raw_vertices = payload.get("vertices")
        raw_edges = payload.get("edges")
        if not isinstance(raw_vertices, dict) or not isinstance(raw_edges, list):
            raise ValueError("graph.json must contain 'vertices' and 'edges'.")

        vertices: dict[str, Point] = {}
        for node, coordinates in raw_vertices.items():
            if (
                not isinstance(coordinates, list)
                or len(coordinates) != 2
                or any(
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    for value in coordinates
                )
            ):
                raise ValueError(
                    f"Vertex {node!r} must have two numeric coordinates."
                )
            vertices[str(node)] = (float(coordinates[0]), float(coordinates[1]))

        edges: list[EdgeKey] = []
        for edge in raw_edges:
            if not isinstance(edge, list) or len(edge) != 2:
                raise ValueError("Each edge must contain two vertex IDs.")
            a, b = str(edge[0]), str(edge[1])
            if a not in vertices or b not in vertices:
                raise ValueError(f"Edge references an unknown vertex: {a}, {b}.")
            edges.append((a, b))

        return cls(vertices, edges)


class Pathfinder:
    """Uniform Cost, Greedy Best-First, and A* search implementations."""

    def __init__(self, graph: Graph) -> None:
        self.graph = graph

    def heuristic(self, node: str, goal: str) -> float:
        x1, y1 = self.graph.vertices[node]
        x2, y2 = self.graph.vertices[goal]
        return math.hypot(x2 - x1, y2 - y1)

    def steps(self, start: str, goal: str, algorithm: str) -> Iterator[SearchStep]:
        if start not in self.graph.vertices or goal not in self.graph.vertices:
            raise ValueError("Start and destination must be graph vertices.")
        if algorithm not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        costs = {node: math.inf for node in self.graph.vertices}
        previous: dict[str, str | None] = {
            node: None for node in self.graph.vertices
        }
        costs[start] = 0.0
        explored: set[str] = set()
        evaluated_edges: set[EdgeKey] = set()
        queue: list[tuple[float, int, float, str]] = []
        sequence = 0

        def priority(node: str, cost: float) -> float:
            if algorithm == ALGORITHMS[1]:
                return self.heuristic(node, goal)
            if algorithm == ALGORITHMS[2]:
                return cost + self.heuristic(node, goal)
            return cost

        heapq.heappush(queue, (priority(start, 0.0), sequence, 0.0, start))

        while queue:
            _, _, current_cost, current = heapq.heappop(queue)
            if current_cost != costs[current] or current in explored:
                continue
            explored.add(current)
            scanned_edges: list[tuple[str, str]] = []

            if current != goal:
                for neighbor, weight in self.graph.adjacency[current]:
                    key = edge_key(current, neighbor)
                    evaluated_edges.add(key)
                    scanned_edges.append((current, neighbor))
                    if neighbor in explored:
                        continue
                    candidate = current_cost + weight
                    if candidate < costs[neighbor]:
                        costs[neighbor] = candidate
                        previous[neighbor] = current
                        sequence += 1
                        heapq.heappush(
                            queue,
                            (
                                priority(neighbor, candidate),
                                sequence,
                                candidate,
                                neighbor,
                            ),
                        )

            open_nodes = frozenset(
                item[3]
                for item in queue
                if item[3] not in explored and item[2] == costs[item[3]]
            )

            path: list[str] = []
            if current == goal:
                node: str | None = goal
                while node is not None:
                    path.append(node)
                    node = previous[node]
                path.reverse()

            yield SearchStep(
                current=current,
                open_nodes=open_nodes,
                explored_nodes=frozenset(explored),
                evaluated_edges=frozenset(evaluated_edges),
                scanned_edges=tuple(scanned_edges),
                path=tuple(path),
                cost=current_cost if path else math.inf,
            )
            if path:
                return


class GraphVisualizer:
    """Pygame UI, input handling, and animation state."""

    def __init__(self, graph: Graph, map_path: Path) -> None:
        pygame.init()
        pygame.display.set_caption("Graph Pathfinding Visualizer")
        desktop_width, desktop_height = pygame.display.get_desktop_sizes()[0]
        self.desktop_size = (desktop_width, desktop_height)
        self.fullscreen = True
        self.screen = pygame.display.set_mode(
            (desktop_width, desktop_height), pygame.FULLSCREEN
        )
        self.clock = pygame.time.Clock()
        self.graph = graph
        self.pathfinder = Pathfinder(graph)
        self.map_image = self.load_map(map_path)

        self.font = pygame.font.SysFont("segoeui", 17)
        self.small_font = pygame.font.SysFont("segoeui", 13)
        self.node_font = pygame.font.SysFont("segoeui", 13, bold=True)
        self.title_font = pygame.font.SysFont("segoeui", 22, bold=True)

        self.algorithm_index = 0
        self.start: str | None = None
        self.goal: str | None = None
        self.search_iterator: Iterator[SearchStep] | None = None
        self.search_accumulator = 0.0
        self.search_interval = 0.08
        self.open_nodes: set[str] = set()
        self.explored_nodes: set[str] = set()
        self.evaluated_edges: set[EdgeKey] = set()
        self.path: list[str] = []
        self.phase = "selection"
        self.result = "Click a vertex to choose a start, then choose a destination."

        self.speed = 300.0
        self.dragging_speed = False
        self.runner_position: Point | None = None
        self.runner_segment = 0
        self.runner_distance = 0.0
        self.runner_segments: list[tuple[Point, Point, float]] = []
        self.pending_search_edges: list[tuple[str, str]] = []
        self.active_search_edge: tuple[str, str] | None = None
        self.search_edge_distance = 0.0
        self.search_runner_position: Point | None = None
        self.pending_solution: tuple[tuple[str, ...], float] | None = None
        self.search_finished = False

    @staticmethod
    def load_map(path: Path) -> pygame.Surface | None:
        if not path.exists():
            return None
        try:
            return pygame.image.load(str(path)).convert()
        except pygame.error as error:
            raise ValueError(f"Could not load map image {path}: {error}") from error

    @property
    def algorithm(self) -> str:
        return ALGORITHMS[self.algorithm_index]

    def map_layout(self) -> tuple[pygame.Rect, float]:
        width, height = self.screen.get_size()
        viewport = pygame.Rect(0, HEADER_HEIGHT, width, height - HEADER_HEIGHT)
        source_width, source_height = (
            self.map_image.get_size() if self.map_image else (1721, 838)
        )
        scale = min(
            viewport.width / source_width,
            viewport.height / source_height,
        )
        drawn_width = source_width * scale
        drawn_height = source_height * scale
        rect = pygame.Rect(
            viewport.centerx - drawn_width / 2,
            viewport.centery - drawn_height / 2,
            drawn_width,
            drawn_height,
        )
        return rect, scale

    def to_screen(self, point: Point) -> tuple[int, int]:
        rect, scale = self.map_layout()
        return round(rect.x + point[0] * scale), round(rect.y + point[1] * scale)

    def draw_text(
        self,
        text: str,
        position: tuple[int, int],
        color: tuple[int, int, int] | None = None,
        font: pygame.font.Font | None = None,
        centered: bool = False,
    ) -> pygame.Rect:
        surface = (font or self.font).render(
            text, True, color or COLORS["text"]
        )
        rect = surface.get_rect()
        if centered:
            rect.center = position
        else:
            rect.topleft = position
        self.screen.blit(surface, rect)
        return rect

    def control_rects(self) -> dict[str, pygame.Rect]:
        width = self.screen.get_width()
        return {
            "algorithm": pygame.Rect(204, 76, 284, 42),
            "run": pygame.Rect(504, 76, 140, 42),
            "reset": pygame.Rect(656, 76, 110, 42),
            "slider": pygame.Rect(max(width - 290, 890), 91, 190, 8),
        }

    def draw_button(
        self, rect: pygame.Rect, label: str, active: bool = False
    ) -> None:
        color = COLORS["button_active"] if active else COLORS["button"]
        if rect.collidepoint(pygame.mouse.get_pos()):
            color = COLORS["button_hover"]
        pygame.draw.rect(self.screen, color, rect, border_radius=9)
        pygame.draw.rect(
            self.screen, COLORS["panel_border"], rect, width=1, border_radius=9
        )
        self.draw_text(label, rect.center, centered=True)

    def draw_header(self) -> None:
        width = self.screen.get_width()
        pygame.draw.rect(
            self.screen, COLORS["panel"], (0, 0, width, HEADER_HEIGHT)
        )
        pygame.draw.line(
            self.screen,
            COLORS["panel_border"],
            (0, HEADER_HEIGHT - 1),
            (width, HEADER_HEIGHT - 1),
        )
        self.draw_text("Graph Pathfinding Visualizer", (20, 12), font=self.title_font)
        self.draw_text(
            "F11: Toggle fullscreen  |  R: Reset",
            (max(340, width - 300), 18),
            COLORS["muted"],
            self.small_font,
        )

        start_text = f"Start: {self.start if self.start is not None else '—'}"
        goal_text = f"Destination: {self.goal if self.goal is not None else '—'}"
        self.draw_text(start_text, (22, 53), COLORS["start"])
        self.draw_text(goal_text, (22, 82), COLORS["destination"])

        controls = self.control_rects()
        self.draw_button(controls["algorithm"], self.algorithm)
        self.draw_button(controls["run"], "Run Search", active=True)
        self.draw_button(controls["reset"], "Reset")

        slider = controls["slider"]
        self.draw_text("Speed", (slider.x, 54), COLORS["muted"], self.small_font)
        pygame.draw.line(
            self.screen, COLORS["panel_border"], slider.midleft, slider.midright, 5
        )
        ratio = (self.speed - 60.0) / (700.0 - 60.0)
        knob_x = round(slider.left + ratio * slider.width)
        pygame.draw.circle(self.screen, COLORS["open"], (knob_x, slider.centery), 10)
        self.draw_text(
            f"{int(self.speed)} px/s",
            (slider.right + 14, slider.centery),
            COLORS["muted"],
            self.small_font,
            centered=False,
        )

        status = self.result
        self.draw_text(
            status[:180],
            (20, 121),
            COLORS["text"] if self.phase != "selection" else COLORS["muted"],
            self.small_font,
        )

    def draw_map(self) -> None:
        viewport = pygame.Rect(
            0, HEADER_HEIGHT, self.screen.get_width(),
            self.screen.get_height() - HEADER_HEIGHT
        )
        self.screen.fill(COLORS["background"], viewport)
        rect, _ = self.map_layout()
        if self.map_image is not None:
            size = (max(1, round(rect.width)), max(1, round(rect.height)))
            scaled = pygame.transform.smoothscale(self.map_image, size)
            self.screen.blit(scaled, rect)
        else:
            pygame.draw.rect(self.screen, (38, 48, 62), rect)

        self.draw_edges()
        self.draw_nodes()
        self.draw_legend()

    def draw_edges(self) -> None:
        _, scale = self.map_layout()
        road_width = max(8, round(11 * scale))
        overlay_width = max(3, road_width - 5)
        path_edges = {
            edge_key(left, right) for left, right in zip(self.path, self.path[1:])
        }

        for a, b in self.graph.edges:
            start = self.to_screen(self.graph.vertices[a])
            end = self.to_screen(self.graph.vertices[b])
            pygame.draw.line(
                self.screen, COLORS["road_outline"], start, end, road_width + 3
            )
            pygame.draw.line(self.screen, COLORS["road"], start, end, road_width)

            key = edge_key(a, b)
            overlay = None
            if key in path_edges:
                overlay = COLORS["path"]
            elif key in self.evaluated_edges:
                overlay = COLORS["search_edge"]
            if overlay:
                pygame.draw.line(self.screen, overlay, start, end, overlay_width)

            midpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
            dx, dy = end[0] - start[0], end[1] - start[1]
            length = math.hypot(dx, dy) or 1.0
            label_position = (
                round(midpoint[0] - dy / length * 9),
                round(midpoint[1] + dx / length * 9),
            )
            weight_text = f"{self.graph.edge_weights[key]:.1f}"
            label = self.small_font.render(
                weight_text, True, COLORS["edge_label"]
            )
            label_rect = label.get_rect(center=label_position)
            outline = self.small_font.render(weight_text, True, (255, 255, 255))
            for offset_x, offset_y in (
                (-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (1, -1), (-1, 1), (1, 1),
            ):
                self.screen.blit(
                    outline, label_rect.move(offset_x, offset_y)
                )
            self.screen.blit(label, label_rect)

    def draw_nodes(self) -> None:
        _, scale = self.map_layout()
        radius = max(12, round(15 * scale))
        for node, point in self.graph.vertices.items():
            center = self.to_screen(point)
            if node == self.start:
                color = COLORS["start"]
            elif node == self.goal:
                color = COLORS["destination"]
            elif node in self.open_nodes:
                color = COLORS["open"]
            elif node in self.explored_nodes:
                color = COLORS["explored"]
            else:
                color = COLORS["node"]
            pygame.draw.circle(self.screen, COLORS["road_outline"], center, radius + 2)
            pygame.draw.circle(self.screen, color, center, radius)
            pygame.draw.circle(
                self.screen, (255, 255, 255), center, radius, width=2
            )
            self.draw_text(
                node, center, (255, 255, 255), self.node_font, centered=True
            )

        if self.runner_position is not None:
            center = self.to_screen(self.runner_position)
            pygame.draw.circle(self.screen, (32, 38, 48), center, radius - 1)
            pygame.draw.circle(self.screen, COLORS["runner"], center, radius - 4)
            pygame.draw.circle(
                self.screen, (255, 255, 255), center, radius - 4, width=2
            )
        if self.search_runner_position is not None:
            center = self.to_screen(self.search_runner_position)
            search_radius = max(7, round(9 * scale))
            pygame.draw.circle(
                self.screen, (255, 255, 255), center, search_radius + 2
            )
            pygame.draw.circle(
                self.screen, COLORS["search_runner"], center, search_radius
            )
            pygame.draw.circle(
                self.screen, (120, 225, 255), center, search_radius - 3
            )

    def draw_legend(self) -> None:
        x, y = 16, HEADER_HEIGHT + 14
        entries = (
            (COLORS["start"], "Start"),
            (COLORS["destination"], "Destination"),
            (COLORS["open"], "Open list"),
            (COLORS["explored"], "Explored"),
            (COLORS["search_runner"], "Scanning edge"),
            (COLORS["path"], "Final path"),
        )
        width = 142
        height = 24 + len(entries) * 22
        panel = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, COLORS["panel"], panel, border_radius=8)
        pygame.draw.rect(
            self.screen, COLORS["panel_border"], panel, width=1, border_radius=8
        )
        for index, (color, label) in enumerate(entries):
            center = (x + 15, y + 23 + index * 22)
            pygame.draw.circle(self.screen, color, center, 6)
            self.draw_text(
                label,
                (x + 28, center[1]),
                COLORS["text"],
                self.small_font,
                centered=False,
            )

    def render(self) -> None:
        self.screen.fill(COLORS["background"])
        self.draw_map()
        self.draw_header()
        pygame.display.flip()

    def node_at(self, position: tuple[int, int]) -> str | None:
        _, scale = self.map_layout()
        threshold = max(18, 21 * scale)
        nearest: tuple[str, float] | None = None
        for node, point in self.graph.vertices.items():
            center = self.to_screen(point)
            distance = math.hypot(position[0] - center[0], position[1] - center[1])
            if distance <= threshold and (nearest is None or distance < nearest[1]):
                nearest = node, distance
        return nearest[0] if nearest else None

    def start_search(self) -> None:
        if self.start is None or self.goal is None:
            self.result = "Choose both a start and a destination before searching."
            return
        if self.start == self.goal:
            self.result = "Start and destination must be different vertices."
            return

        self.open_nodes.clear()
        self.explored_nodes.clear()
        self.evaluated_edges.clear()
        self.path.clear()
        self.runner_position = None
        self.search_runner_position = self.graph.vertices[self.start]
        self.pending_search_edges.clear()
        self.active_search_edge = None
        self.search_edge_distance = 0.0
        self.pending_solution = None
        self.search_finished = False
        self.search_iterator = self.pathfinder.steps(
            self.start, self.goal, self.algorithm
        )
        self.search_accumulator = self.search_interval
        self.phase = "exploring"
        self.result = f"Exploring with {self.algorithm}..."

    def reset(self) -> None:
        self.start = None
        self.goal = None
        self.search_iterator = None
        self.open_nodes.clear()
        self.explored_nodes.clear()
        self.evaluated_edges.clear()
        self.path.clear()
        self.runner_position = None
        self.runner_segments.clear()
        self.search_runner_position = None
        self.pending_search_edges.clear()
        self.active_search_edge = None
        self.search_edge_distance = 0.0
        self.pending_solution = None
        self.search_finished = False
        self.phase = "selection"
        self.result = "Click a vertex to choose a start, then choose a destination."

    def select_node(self, node: str) -> None:
        if self.phase != "selection":
            self.reset()
        if self.start is None or self.goal is not None:
            self.start = node
            self.goal = None
            self.phase = "selection"
            self.result = f"Start set to {node}. Choose a destination vertex."
        elif node != self.start:
            self.goal = node
            self.phase = "selection"
            self.result = (
                f"Destination set to {node}. Press Run Search or Space to begin."
            )

    def begin_path_running(self, path: tuple[str, ...], cost: float) -> None:
        self.path = list(path)
        self.open_nodes.clear()
        self.search_runner_position = None
        self.active_search_edge = None
        self.pending_search_edges.clear()
        self.phase = "running"
        self.runner_position = self.graph.vertices[path[0]]
        self.runner_segments = []
        for a, b in zip(path, path[1:]):
            first, second = self.graph.vertices[a], self.graph.vertices[b]
            length = math.dist(first, second)
            self.runner_segments.append((first, second, length))
        self.runner_segment = 0
        self.runner_distance = 0.0
        self.result = (
            f"{self.algorithm}: route distance {cost:.1f}. "
            "Final path highlighted; marker en route to destination."
        )
        if not self.runner_segments:
            self.finish_path()

    def finish_path(self) -> None:
        self.phase = "complete"
        if self.goal is not None:
            self.runner_position = self.graph.vertices[self.goal]
        self.result = f"Arrived at destination {self.goal}."

    def update_search(self, elapsed: float) -> None:
        if self.phase != "exploring" or self.search_iterator is None:
            return

        if self.active_search_edge is not None:
            start_node, end_node = self.active_search_edge
            start = self.graph.vertices[start_node]
            end = self.graph.vertices[end_node]
            length = math.dist(start, end)
            self.search_edge_distance = min(
                self.search_edge_distance + self.speed * 10 * elapsed,
                length,
            )
            ratio = self.search_edge_distance / length if length else 1.0
            self.search_runner_position = (
                start[0] + (end[0] - start[0]) * ratio,
                start[1] + (end[1] - start[1]) * ratio,
            )
            if self.search_edge_distance >= length:
                self.search_runner_position = end
                self.active_search_edge = None
                self.search_edge_distance = 0.0
            return

        if self.pending_search_edges:
            self.active_search_edge = self.pending_search_edges.pop(0)
            start_node, _ = self.active_search_edge
            self.search_runner_position = self.graph.vertices[start_node]
            return

        if self.pending_solution is not None:
            path, cost = self.pending_solution
            self.pending_solution = None
            self.search_iterator = None
            self.begin_path_running(path, cost)
            return

        if self.search_finished:
            self.search_iterator = None
            self.phase = "no_path"
            self.result = "No route exists between the selected vertices."
            return

        self.search_accumulator += elapsed
        if self.search_accumulator < self.search_interval:
            return
        self.search_accumulator = 0.0

        try:
            step = next(self.search_iterator)
        except StopIteration:
            self.search_finished = True
            return

        self.open_nodes = set(step.open_nodes)
        self.explored_nodes = set(step.explored_nodes)
        self.evaluated_edges = set(step.evaluated_edges)
        self.pending_search_edges = list(step.scanned_edges)
        self.result = (
            f"{self.algorithm}: scanning from vertex {step.current} | "
            f"{len(self.explored_nodes)} explored, {len(self.open_nodes)} open"
        )
        if step.path:
            self.pending_solution = (step.path, step.cost)

    def update_runner(self, elapsed: float) -> None:
        if self.phase != "running" or self.runner_position is None:
            return
        remaining = self.speed * elapsed
        while remaining > 0 and self.runner_segment < len(self.runner_segments):
            start, end, length = self.runner_segments[self.runner_segment]
            step = min(remaining, length - self.runner_distance)
            self.runner_distance += step
            remaining -= step
            ratio = self.runner_distance / length if length else 1.0
            self.runner_position = (
                start[0] + (end[0] - start[0]) * ratio,
                start[1] + (end[1] - start[1]) * ratio,
            )
            if self.runner_distance >= length:
                self.runner_segment += 1
                self.runner_distance = 0.0
        if self.runner_segment >= len(self.runner_segments):
            self.finish_path()

    def set_speed_from_mouse(self, mouse_x: int) -> None:
        slider = self.control_rects()["slider"]
        ratio = min(1.0, max(0.0, (mouse_x - slider.left) / slider.width))
        self.speed = 60.0 + ratio * (700.0 - 60.0)

    def toggle_fullscreen(self) -> None:
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            self.screen = pygame.display.set_mode(
                self.desktop_size, pygame.FULLSCREEN
            )
        else:
            width = max(900, self.desktop_size[0] - 100)
            height = max(600, self.desktop_size[1] - 100)
            self.screen = pygame.display.set_mode(
                (width, height), pygame.RESIZABLE
            )

    def handle_click(self, position: tuple[int, int]) -> None:
        controls = self.control_rects()
        if position[1] < HEADER_HEIGHT:
            if controls["algorithm"].collidepoint(position):
                if self.phase in ("selection", "complete", "no_path"):
                    self.algorithm_index = (
                        self.algorithm_index + 1
                    ) % len(ALGORITHMS)
            elif controls["run"].collidepoint(position):
                self.start_search()
            elif controls["reset"].collidepoint(position):
                self.reset()
            elif controls["slider"].inflate(0, 26).collidepoint(position):
                self.dragging_speed = True
                self.set_speed_from_mouse(position[0])
            return

        node = self.node_at(position)
        if node is not None:
            self.select_node(node)

    def run(self) -> None:
        running = True
        while running:
            elapsed = min(self.clock.tick(FPS) / 10000.0, 0.08)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    if not self.fullscreen:
                        self.screen = pygame.display.set_mode(
                            event.size, pygame.RESIZABLE
                        )
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.handle_click(event.pos)
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    self.dragging_speed = False
                elif event.type == pygame.MOUSEMOTION and self.dragging_speed:
                    self.set_speed_from_mouse(event.pos[0])
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.start_search()
                    elif event.key == pygame.K_r:
                        self.reset()
                    elif event.key == pygame.K_F11:
                        self.toggle_fullscreen()

            self.update_search(elapsed)
            self.update_runner(elapsed)
            self.render()

        pygame.quit()


def main() -> int:
    try:
        graph = Graph.load(GRAPH_PATH)
        app = GraphVisualizer(graph, MAP_PATH)
        app.run()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Could not start the pathfinding visualizer: {error}", file=sys.stderr)
        pygame.quit()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
