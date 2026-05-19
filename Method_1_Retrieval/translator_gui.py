"""
translator_gui.py
=================
Main GUI for the Syllable-Constrained Hindi Song Translator.

Run:   python translator_gui.py
       (or double-click run_translator.bat which activates nlp_venv first)
"""

import threading
import random
import tkinter as tk
from tkinter import ttk, font as tkfont

from sentence_transformers import SentenceTransformer

from semantic_translator import get_unique_hindi_lines, get_or_create_embeddings, model_name
from syllable_splitter  import split_english_syllables, split_hindi_syllables
from syllable_counter   import count_english_syllables, count_hindi_syllables
from ranking_engine     import RankingEngine
from synonym_swapper    import SynonymSwapper

# ── TTS helper (gTTS — genuine Hindi pronunciation via Google TTS) ─────────────

def speak_hindi(text: str):
    """
    Speak Hindi text using Google TTS (gTTS).
    Saves a temp MP3 and plays it with pygame.mixer.
    Requires internet connection. Runs in a daemon thread.
    """
    def _run():
        import tempfile, os, time
        tmp_path = None
        try:
            from gtts import gTTS
            import pygame

            tts = gTTS(text=text, lang='hi', slow=False)
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
                tmp_path = f.name
            tts.save(tmp_path)

            pygame.mixer.init()
            pygame.mixer.music.load(tmp_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        except Exception as e:
            print(f'[TTS] Error: {e}')
        finally:
            if tmp_path:
                try: os.unlink(tmp_path)
                except: pass

    threading.Thread(target=_run, daemon=True).start()





# ── Colour palette ────────────────────────────────────────────────────────────
BG        = '#0f0f1a'
CARD_BG   = '#1a1a2e'
PANEL_BG  = '#16213e'
ACCENT    = '#7c3aed'
ACCENT2   = '#a78bfa'
TEXT      = '#e2e8f0'
SUBTEXT   = '#94a3b8'
SUCCESS   = '#10b981'
WARNING   = '#f59e0b'
DANGER    = '#ef4444'
BTN_BG    = '#7c3aed'
BTN_FG    = '#ffffff'
BORDER    = '#2d2d4e'

SYLLABLE_COLORS = [
    '#FF6B6B', '#51CF66', '#339AF0',
    '#FFD43B', '#CC5DE8', '#FF922B',
    '#20C997', '#F06595',
]

METRIC_COLORS = {
    'purport':   '#a78bfa',
    'meaning':   '#60a5fa',
    'syllable':  '#34d399',
    'bollywood': '#fbbf24',
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def syllable_match_color(en_syl: int, hi_syl: int) -> str:
    diff = abs(en_syl - hi_syl)
    if diff == 0:  return SUCCESS
    if diff <= 1:  return WARNING
    return DANGER


def make_button(parent, text, command, bg=BTN_BG, width=None):
    cfg = dict(text=text, command=command, bg=bg, fg=BTN_FG,
               activebackground=ACCENT2, activeforeground='white',
               relief='flat', bd=0, cursor='hand2',
               font=('Segoe UI', 10, 'bold'), pady=6)
    if width:
        cfg['width'] = width
    return tk.Button(parent, **cfg)


def separator(parent, color=BORDER, pady=6):
    f = tk.Frame(parent, bg=color, height=1)
    f.pack(fill='x', pady=pady)
    return f


def metric_bar(parent, label: str, value: float, color: str):
    """Render a labelled progress bar inside parent."""
    row = tk.Frame(parent, bg=CARD_BG)
    row.pack(fill='x', pady=2)

    tk.Label(row, text=label, bg=CARD_BG, fg=SUBTEXT,
             font=('Segoe UI', 9), width=12, anchor='w').pack(side='left')

    bar_bg = tk.Canvas(row, width=160, height=10, bg='#2d2d4e',
                       highlightthickness=0, bd=0)
    bar_bg.pack(side='left', padx=(4, 6))
    filled = int(value * 160)
    if filled > 0:
        bar_bg.create_rectangle(0, 0, filled, 10, fill=color, outline='')

    tk.Label(row, text=f'{value:.2f}', bg=CARD_BG, fg=color,
             font=('Segoe UI', 9, 'bold')).pack(side='left')


def syllable_chips(parent, syllables: list, bg_colors: list):
    """Render coloured syllable pill labels inside a frame."""
    frame = tk.Frame(parent, bg=CARD_BG)
    frame.pack(anchor='w', pady=(2, 0))
    for i, syl in enumerate(syllables):
        color = bg_colors[i % len(bg_colors)]
        tk.Label(frame, text=f' {syl} ', bg=color, fg='white',
                 font=('Segoe UI', 11, 'bold'),
                 padx=4, pady=3, relief='flat').pack(side='left', padx=2)
    return frame


# ── Result Card ───────────────────────────────────────────────────────────────

class ResultCard:
    """Displays one translated line inside the scrollable results pane."""

    def __init__(self, parent, line_num: int, english_line: str,
                 pool: list, ranking: RankingEngine,
                 weight_vars: dict, on_random):
        self.parent       = parent
        self.english_line = english_line
        self.pool         = pool
        self.ranking      = ranking
        self.weight_vars  = weight_vars
        self.on_random    = on_random
        self.current_idx  = self._best_idx()  # pick by weights, not pool[0]

        # Outer card frame
        self.card = tk.Frame(parent, bg=CARD_BG, bd=0, relief='flat',
                             highlightthickness=1, highlightbackground=BORDER)
        self.card.pack(fill='x', padx=10, pady=6)

        self._header(line_num, english_line)
        self._body_frame = tk.Frame(self.card, bg=CARD_BG)
        self._body_frame.pack(fill='x', padx=14, pady=(0, 10))

        self.render(self.current_idx)

    def _header(self, n, text):
        hdr = tk.Frame(self.card, bg=ACCENT, height=2)
        hdr.pack(fill='x')
        row = tk.Frame(self.card, bg=CARD_BG)
        row.pack(fill='x', padx=14, pady=(8, 4))
        tk.Label(row, text=f'#{n}', bg=CARD_BG, fg=ACCENT2,
                 font=('Segoe UI', 9, 'bold')).pack(side='left')
        tk.Label(row, text=f'  {text}', bg=CARD_BG, fg=TEXT,
                 font=('Segoe UI', 11)).pack(side='left')

    def _weights(self) -> dict:
        return {k: v.get() for k, v in self.weight_vars.items()}

    def _best_idx(self) -> int:
        """Pick the pool candidate with the highest quick weighted score."""
        if not self.pool:
            return 0
        w = self._weights()
        en_syl = count_english_syllables(self.english_line)
        best_i, best_s = 0, -1.0
        for i, c in enumerate(self.pool):
            # Fast approximation: purport + meaning share semantic_score; syllable uses diff
            syl_score = max(0.0, 1.0 - abs(c['hi_syllables'] - en_syl) / max(en_syl, 1))
            combined = ((w.get('purport', 0.35) + w.get('meaning', 0.25)) * c['semantic_score']
                        + w.get('syllable', 0.30) * syl_score)
            if combined > best_s:
                best_s, best_i = combined, i
        return best_i

    def render(self, idx: int):
        # Clear body
        for w in self._body_frame.winfo_children():
            w.destroy()

        if not self.pool:
            tk.Label(self._body_frame, text='No candidates found.',
                     bg=CARD_BG, fg=DANGER,
                     font=('Segoe UI', 10)).pack()
            return

        cand    = self.pool[idx]
        hindi   = cand['line']
        en_emb  = cand['en_emb']
        scores  = self.ranking.score_all(
            self.english_line, hindi, self._weights(), en_emb)

        en_syl  = scores['en_syllables']
        hi_syl  = scores['hi_syllables']
        match_c = syllable_match_color(en_syl, hi_syl)

        # ── Syllable display ────────────────────────────────────────────────
        en_sylls = split_english_syllables(self.english_line)
        hi_sylls = split_hindi_syllables(hindi)

        # EN row
        en_row = tk.Frame(self._body_frame, bg=CARD_BG)
        en_row.pack(fill='x', pady=(4, 0))
        tk.Label(en_row, text='EN ', bg=CARD_BG, fg=SUBTEXT,
                 font=('Segoe UI', 9, 'bold'), width=3).pack(side='left')
        en_chip_frame = tk.Frame(en_row, bg=CARD_BG)
        en_chip_frame.pack(side='left')
        for i, syl in enumerate(en_sylls):
            color = SYLLABLE_COLORS[i % len(SYLLABLE_COLORS)]
            tk.Label(en_chip_frame, text=f' {syl} ', bg=color, fg='white',
                     font=('Segoe UI', 11, 'bold'),
                     padx=4, pady=3, relief='flat').pack(side='left', padx=2)
        tk.Label(en_row, text=f' ({en_syl})', bg=CARD_BG, fg=SUBTEXT,
                 font=('Segoe UI', 9)).pack(side='left', padx=4)

        # HI row
        hi_row = tk.Frame(self._body_frame, bg=CARD_BG)
        hi_row.pack(fill='x', pady=(4, 6))
        tk.Label(hi_row, text='HI ', bg=CARD_BG, fg=SUBTEXT,
                 font=('Segoe UI', 9, 'bold'), width=3).pack(side='left')
        hi_chip_frame = tk.Frame(hi_row, bg=CARD_BG)
        hi_chip_frame.pack(side='left')
        for i, syl in enumerate(hi_sylls):
            color = SYLLABLE_COLORS[i % len(SYLLABLE_COLORS)]
            tk.Label(hi_chip_frame, text=f' {syl} ', bg=color, fg='white',
                     font=('Segoe UI', 11, 'bold'),
                     padx=4, pady=3, relief='flat').pack(side='left', padx=2)

        # Syllable count badge
        badge_txt = f'✓ {hi_syl}' if en_syl == hi_syl else f'△ {hi_syl}'
        tk.Label(hi_row, text=f' ({badge_txt})', bg=CARD_BG, fg=match_c,
                 font=('Segoe UI', 9, 'bold')).pack(side='left', padx=4)

        separator(self._body_frame, pady=4)

        # ── 4 Metric bars ────────────────────────────────────────────────────
        metric_bar(self._body_frame, 'Purport',   scores['purport'],   METRIC_COLORS['purport'])
        metric_bar(self._body_frame, 'Meaning',   scores['meaning'],   METRIC_COLORS['meaning'])
        metric_bar(self._body_frame, 'Syllable',  scores['syllable'],  METRIC_COLORS['syllable'])
        metric_bar(self._body_frame, 'Bollywood', scores['bollywood'], METRIC_COLORS['bollywood'])

        separator(self._body_frame, pady=4)

        # ── Footer: score + random ────────────────────────────────────────────
        foot = tk.Frame(self._body_frame, bg=CARD_BG)
        foot.pack(fill='x')

        score_lbl = tk.Label(foot,
            text=f'Score  {scores["final"]:.3f}',
            bg=CARD_BG, fg=ACCENT2, font=('Segoe UI', 12, 'bold'))
        score_lbl.pack(side='left')

        def on_rand():
            next_idx = random.choice(
                [i for i in range(len(self.pool)) if i != self.current_idx]
                or [0])
            self.current_idx = next_idx
            self.render(next_idx)
            if self.on_random:
                self.on_random()

        rand_btn = make_button(foot, '🎲 Random', on_rand, bg='#374151', width=10)
        rand_btn.pack(side='right')

        # ── Re-score button ───────────────────────────────────────────────────
        def on_rescore():
            self.current_idx = self._best_idx()   # re-rank by current weights
            self.render(self.current_idx)

        rescore_btn = make_button(foot, '↻ Re-score', on_rescore,
                                  bg='#1f2937', width=10)
        rescore_btn.pack(side='right', padx=4)

        # ── 🔊 Speak Hindi button ─────────────────────────────────────────────
        speak_btn = make_button(foot, '🔊', lambda: speak_hindi(hindi),
                                bg='#065f46', width=4)
        speak_btn.pack(side='right', padx=4)


# ── Main Application ──────────────────────────────────────────────────────────

class TranslatorApp:
    def __init__(self, root: tk.Tk):
        self.root   = root
        self.model  = None
        self.db_lines      = []
        self.db_embeddings = None
        self.ranking = None
        self.swapper = None
        self.cards   = []

        root.title('🎵 Hindi Lyric Translator')
        root.configure(bg=BG)
        root.geometry('1200x800')
        root.minsize(900, 600)

        self._build_ui()
        self._start_loading()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # Top header
        hdr = tk.Frame(self.root, bg=ACCENT, height=48)
        hdr.pack(fill='x')
        hdr.pack_propagate(False)
        tk.Label(hdr, text='🎵  Hindi Lyric Translator',
                 bg=ACCENT, fg='white',
                 font=('Segoe UI', 15, 'bold')).pack(side='left', padx=16, pady=10)

        self.status_lbl = tk.Label(hdr, text='Loading model…', bg=ACCENT,
                                   fg='#ddd6fe', font=('Segoe UI', 9))
        self.status_lbl.pack(side='right', padx=16)

        # Main area
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill='both', expand=True)

        self._build_left_panel(main)
        self._build_right_panel(main)

    def _build_left_panel(self, parent):
        left = tk.Frame(parent, bg=PANEL_BG, width=300)
        left.pack(side='left', fill='y')
        left.pack_propagate(False)

        # ── Input ────────────────────────────────────────────────────────────
        tk.Label(left, text='English Lyrics', bg=PANEL_BG, fg=ACCENT2,
                 font=('Segoe UI', 11, 'bold')).pack(anchor='w', padx=14, pady=(14, 2))

        self.lyrics_input = tk.Text(
            left, height=12, wrap='word',
            bg='#1e1e38', fg=TEXT, insertbackground=TEXT,
            font=('Segoe UI', 10), relief='flat', bd=0,
            highlightthickness=1, highlightbackground=BORDER,
            padx=8, pady=8)
        self.lyrics_input.pack(fill='x', padx=14, pady=(0, 6))
        self.lyrics_input.insert('end',
            "I can't stop the feeling\n"
            "Got this feeling in my body\n"
            "I ain't gonna stop now")

        self.translate_btn = make_button(
            left, '▶  Translate All', self._on_translate)
        self.translate_btn.config(state='disabled')   # enabled once model is ready
        self.translate_btn.pack(fill='x', padx=14, pady=4)

        separator(left, pady=8)

        # ── Metric Weights ────────────────────────────────────────────────────
        tk.Label(left, text='Metric Weights', bg=PANEL_BG, fg=ACCENT2,
                 font=('Segoe UI', 11, 'bold')).pack(anchor='w', padx=14, pady=(0, 6))

        self.weight_vars = {}
        metric_defaults  = {
            'purport':   0.35,
            'meaning':   0.25,
            'syllable':  0.30,
            'bollywood': 0.10,
        }
        metric_colors_list = list(METRIC_COLORS.items())

        for name, default in metric_defaults.items():
            color = METRIC_COLORS[name]
            self._slider_row(left, name.capitalize(), name, default, color)

        separator(left, pady=8)

        tk.Label(left, text='Tip: click ↻ Re-score to apply weight changes.',
                 bg=PANEL_BG, fg=SUBTEXT,
                 font=('Segoe UI', 8), wraplength=260, justify='left'
                 ).pack(anchor='w', padx=14)

    def _slider_row(self, parent, label, key, default, color):
        var = tk.DoubleVar(value=default)
        self.weight_vars[key] = var

        row = tk.Frame(parent, bg=PANEL_BG)
        row.pack(fill='x', padx=14, pady=3)

        tk.Label(row, text=f'●', bg=PANEL_BG, fg=color,
                 font=('Segoe UI', 10)).pack(side='left')
        tk.Label(row, text=f' {label}', bg=PANEL_BG, fg=TEXT,
                 font=('Segoe UI', 10), width=10, anchor='w').pack(side='left')

        val_lbl = tk.Label(row, textvariable=tk.StringVar(),
                           bg=PANEL_BG, fg=color,
                           font=('Segoe UI', 9, 'bold'), width=4)
        val_lbl.pack(side='right')

        def update_label(*_):
            val_lbl.config(text=f'{var.get():.2f}')

        var.trace_add('write', update_label)
        update_label()

        scale = ttk.Scale(row, from_=0.0, to=1.0, variable=var, orient='horizontal')
        scale.pack(side='left', fill='x', expand=True, padx=4)

    def _build_right_panel(self, parent):
        right = tk.Frame(parent, bg=BG)
        right.pack(side='left', fill='both', expand=True)

        tk.Label(right, text='Results', bg=BG, fg=ACCENT2,
                 font=('Segoe UI', 12, 'bold')).pack(anchor='w', padx=14, pady=(10, 4))

        # Scrollable canvas
        canvas_frame = tk.Frame(right, bg=BG)
        canvas_frame.pack(fill='both', expand=True, padx=4)

        self.canvas = tk.Canvas(canvas_frame, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(canvas_frame, orient='vertical',
                                  command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side='right', fill='y')
        self.canvas.pack(side='left', fill='both', expand=True)

        self.scroll_frame = tk.Frame(self.canvas, bg=BG)
        self.canvas_window = self.canvas.create_window(
            (0, 0), window=self.scroll_frame, anchor='nw')

        self.scroll_frame.bind('<Configure>', self._on_scroll_frame_resize)
        self.canvas.bind('<Configure>', self._on_canvas_resize)
        self.canvas.bind_all('<MouseWheel>', self._on_mousewheel)

        # Placeholder
        self.placeholder = tk.Label(
            self.scroll_frame,
            text='Paste English lyrics on the left and click ▶ Translate All.',
            bg=BG, fg=SUBTEXT, font=('Segoe UI', 12))
        self.placeholder.pack(pady=80)

    def _on_scroll_frame_resize(self, _e):
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))

    def _on_canvas_resize(self, e):
        self.canvas.itemconfig(self.canvas_window, width=e.width - 10)

    def _on_mousewheel(self, e):
        self.canvas.yview_scroll(int(-1 * (e.delta / 120)), 'units')

    # ── Model loading ─────────────────────────────────────────────────────────

    def _start_loading(self):
        t = threading.Thread(target=self._load_model, daemon=True)
        t.start()

    def _load_model(self):
        self._set_status('Loading sentence-transformer model…')
        self.model = SentenceTransformer(model_name)

        self._set_status('Reading Hindi lyrics database…')
        self.db_lines = get_unique_hindi_lines()

        self._set_status('Building / loading embeddings cache…')
        _, self.db_embeddings = get_or_create_embeddings(self.model, self.db_lines)

        self._set_status('Initialising ranking engine…')
        self.ranking = RankingEngine(self.model, self.db_lines)
        self.swapper = SynonymSwapper(self.model, self.db_lines, self.db_embeddings)

        self.root.after(0, self._on_model_ready)

    def _set_status(self, msg: str):
        self.root.after(0, lambda: self.status_lbl.config(text=msg))

    def _on_model_ready(self):
        self.status_lbl.config(text='Ready  ✓', fg='#6ee7b7')
        self.translate_btn.config(state='normal')

    # ── Translation ───────────────────────────────────────────────────────────

    def _on_translate(self):
        if self.model is None or self.swapper is None:
            return
        self.translate_btn.config(state='disabled', text='Translating…')
        lyrics = self.lyrics_input.get('1.0', 'end').strip()
        lines  = [l.strip() for l in lyrics.splitlines() if l.strip()]
        if not lines:
            self.translate_btn.config(state='normal', text='▶  Translate All')
            return
        t = threading.Thread(target=self._translate_lines, args=(lines,), daemon=True)
        t.start()

    def _translate_lines(self, lines: list):
        # Clear old results
        self.root.after(0, self._clear_results)
        pools = []
        for i, line in enumerate(lines):
            self._set_status(f'Translating line {i + 1}/{len(lines)}…')
            pool = self.swapper.get_candidate_pool(line, pool_size=50)
            pools.append(pool)
        self.root.after(0, lambda: self._render_all(lines, pools))

    def _clear_results(self):
        for w in self.scroll_frame.winfo_children():
            w.destroy()
        self.cards = []

    def _render_all(self, lines: list, pools: list):
        if self.placeholder.winfo_exists():
            try: self.placeholder.destroy()
            except: pass

        for i, (line, pool) in enumerate(zip(lines, pools), start=1):
            card = ResultCard(
                parent      = self.scroll_frame,
                line_num    = i,
                english_line= line,
                pool        = pool,
                ranking     = self.ranking,
                weight_vars = self.weight_vars,
                on_random   = None,
            )
            self.cards.append(card)

        self._set_status(f'Done — {len(lines)} line(s) translated.  ✓')
        self.translate_btn.config(state='normal', text='▶  Translate All')
        self.canvas.yview_moveto(0)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    root = tk.Tk()

    # Apply ttk dark-ish style
    style = ttk.Style(root)
    style.theme_use('clam')
    style.configure('Vertical.TScrollbar', background=BORDER,
                    troughcolor=BG, arrowcolor=SUBTEXT)
    style.configure('TScale', background=PANEL_BG,
                    troughcolor=BORDER, sliderthickness=14)

    app = TranslatorApp(root)
    root.mainloop()
