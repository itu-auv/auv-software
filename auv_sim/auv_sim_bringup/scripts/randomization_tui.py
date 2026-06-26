#!/usr/bin/env python3
"""Terminal UI to configure and launch a randomized-simulation run.

A thin front-end over episode_runner.py: pick the mission tasks (and their
order), the episode count, the base seed, and which randomization categories to
apply, then launch. The runner's own stdout is the live progress view.

  rosrun auv_sim_bringup randomization_tui.py     (or)     python3 randomization_tui.py
"""

import curses
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import randomization as randmod

RUNNER = os.path.join(HERE, "episode_runner.py")

# smach short names (auv_smach/launch/start.launch state_map) + descriptions
TASKS = [
    ("init", "INITIALIZE (state reset -- always runs first)"),
    ("gate", "Navigate through gate"),
    ("slalom", "Navigate through slalom"),
    ("torpedo", "Torpedo task"),
    ("bin", "Bin task"),
    ("octagon", "Octagon task"),
    ("gps", "Navigate to GPS target"),
    ("acoustic_tx", "Acoustic transmitter"),
    ("acoustic_rx", "Acoustic receiver"),
    ("return", "Return through gate"),
    ("pipeline", "Navigate through pipeline"),
]


class Config:
    def __init__(self):
        self.order = ["init"]          # ordered selected task short-names
        self.episodes = 5
        self.seed = None               # None => auto
        self.timeout = 120
        self.world = "pool"
        self.enabled = set(randmod.CATEGORIES)  # randomized categories

    def ordered_tasks(self):
        # always run init first (clean state between episodes on the shared roscore)
        tasks = [t for t in self.order if t != "init"]
        return ["init"] + tasks


def build_command(cfg, python=None):
    """Assemble the episode_runner command (pure -> unit-testable)."""
    python = python or sys.executable
    cmd = [python, RUNNER,
           "--episodes", str(cfg.episodes),
           "--world", cfg.world,
           "--randomize", ",".join(sorted(cfg.enabled)) if cfg.enabled else "none",
           "--episode-timeout", str(cfg.timeout)]
    if cfg.seed is not None:
        cmd += ["--seed", str(cfg.seed)]
    states = ",".join(cfg.ordered_tasks())
    cmd += ["--mission",
            "auv_smach start.launch test_mode:=true test_states:={}".format(states)]
    return cmd


# --------------------------------------------------------------------------
# curses UI
# --------------------------------------------------------------------------

def _toggle_task(cfg, name):
    if name == "init":
        return  # locked: always first
    if name in cfg.order:
        cfg.order.remove(name)
    else:
        cfg.order.append(name)


def draw(stdscr, cfg, focus, rows):
    stdscr.erase()
    h, w = stdscr.getmaxyx()
    title = " Randomized Simulation Launcher "
    stdscr.attron(curses.A_REVERSE)
    stdscr.addstr(0, 0, title.ljust(w - 1)[:w - 1])
    stdscr.attroff(curses.A_REVERSE)
    stdscr.addstr(1, 0, " up/down (j/k): move   space: toggle/select   "
                        "left/right (h/l) or digits: number   r: RUN   q: quit")

    y = 3
    for i, row in enumerate(rows):
        sel = (i == focus)
        prefix = "> " if sel else "  "
        if sel:
            stdscr.attron(curses.A_BOLD)
        if row[0] == "task":
            name, desc = row[1], row[2]
            if name in cfg.order:
                tag = "[{}]".format(cfg.order.index(name) + 1)
            else:
                tag = "[ ]"
            lock = " (locked)" if name == "init" else ""
            stdscr.addstr(y, 0, "{}{} {:<12} {}{}".format(prefix, tag, name, desc, lock)[:w - 1])
        elif row[0] == "episodes":
            stdscr.addstr(y, 0, "{}Episodes      : {}".format(prefix, cfg.episodes)[:w - 1])
        elif row[0] == "seed":
            s = "auto" if cfg.seed is None else cfg.seed
            stdscr.addstr(y, 0, "{}Base seed     : {}  (space: auto/manual, "
                                "type digits)".format(prefix, s)[:w - 1])
        elif row[0] == "timeout":
            stdscr.addstr(y, 0, "{}Episode sec   : {}".format(prefix, cfg.timeout)[:w - 1])
        elif row[0] == "cat":
            cat = row[1]
            mark = "x" if cat in cfg.enabled else " "
            stdscr.addstr(y, 0, "{}Randomize [{}] {}".format(prefix, mark, cat)[:w - 1])
        elif row[0] == "run":
            stdscr.addstr(y, 0, "{}*** RUN ***".format(prefix)[:w - 1])
        if sel:
            stdscr.attroff(curses.A_BOLD)
        y += 1

    states = ",".join(cfg.ordered_tasks())
    stdscr.addstr(min(y + 1, h - 2), 0,
                  ("mission: test_states={}".format(states))[:w - 1])
    stdscr.refresh()


def run_ui(stdscr):
    curses.curs_set(0)
    stdscr.keypad(True)  # so KEY_LEFT/RIGHT/UP/DOWN are decoded
    cfg = Config()
    rows = [("task", n, d) for n, d in TASKS] + [
        ("episodes",), ("seed",), ("timeout",),
    ] + [("cat", c) for c in randmod.CATEGORIES] + [("run",)]
    focus = 0
    while True:
        draw(stdscr, cfg, focus, rows)
        key = stdscr.getch()
        row = rows[focus]
        if key in (curses.KEY_UP, ord("k")):
            focus = (focus - 1) % len(rows)
        elif key in (curses.KEY_DOWN, ord("j")):
            focus = (focus + 1) % len(rows)
        elif key == ord("q"):
            return None
        elif key == ord("r"):
            return cfg
        elif key == ord(" "):
            if row[0] == "task":
                _toggle_task(cfg, row[1])
            elif row[0] == "cat":
                cfg.enabled ^= {row[1]}
            elif row[0] == "seed":
                cfg.seed = None if cfg.seed is not None else 0
        elif key in (curses.KEY_LEFT, curses.KEY_RIGHT, ord("h"), ord("l")):
            delta = 1 if key in (curses.KEY_RIGHT, ord("l")) else -1
            if row[0] == "episodes":
                cfg.episodes = max(1, cfg.episodes + delta)
            elif row[0] == "timeout":
                cfg.timeout = max(10, cfg.timeout + delta * 10)
            elif row[0] == "seed" and cfg.seed is not None:
                cfg.seed = max(0, cfg.seed + delta)
        elif ord("0") <= key <= ord("9"):
            d = key - ord("0")
            if row[0] == "seed":
                cfg.seed = (cfg.seed or 0) * 10 + d
            elif row[0] == "episodes":
                cfg.episodes = cfg.episodes * 10 + d if cfg.episodes < 1000 else d
        elif key in (curses.KEY_BACKSPACE, 127, 8):
            if row[0] == "seed" and cfg.seed is not None:
                cfg.seed //= 10


def main():
    cfg = curses.wrapper(run_ui)
    if cfg is None:
        print("cancelled.")
        return 0
    cmd = build_command(cfg)
    print("\nlaunching:\n  " + " ".join(cmd) + "\n")
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
