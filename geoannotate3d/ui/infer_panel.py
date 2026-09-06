"""
ui/infer_panel.py — Panel de inferencia integrado (Paso 6)

Permite cargar un modelo .pth entrenado, ejecutar inferencia sobre
la nube cargada en el canvas y ver el resultado directamente.

Flujo:
  1. Usuario selecciona el .pth
  2. Se muestra info del modelo (arq, clases, mIoU)
  3. Ajusta parámetros (pts/parche, overlap, batch)
  4. Ejecuta — la inferencia corre en QThread
  5. Al terminar, el resultado se aplica al label_store
     → la nube aparece coloreada por clase en el canvas
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QDoubleSpinBox, QTextEdit, QProgressBar,
    QFrame, QFileDialog, QSizePolicy, QGroupBox,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from ui.icons import icon as qicon, pixmap as qpixmap
from ui.theme import (
    SURFACE, SURFACE_2, SURFACE_3, BORDER, BORDER_SOFT,
    TEXT, TEXT_DIM, TEXT_MUTE,
    ACCENT, ACCENT_STRONG, ACCENT_SOFT,
    OK, OK_SOFT, WARN, WARN_SOFT,
)


# ── Worker de inferencia ──────────────────────────────────────────────────────

class InferWorker(QThread):
    """Corre la inferencia en background usando infer.py como librería."""
    progress    = pyqtSignal(float)       # 0-100
    log_line    = pyqtSignal(str)
    finished_ok = pyqtSignal(object, object)  # (predictions (N,) uint8, confidence (N,) float32|None)
    error       = pyqtSignal(str)

    def __init__(self, config: dict):
        super().__init__()
        self._cfg  = config
        self._stop = False

    def stop(self): self._stop = True

    def run(self):
        try:
            self._infer()
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()[:600]}")

    def _infer(self):
        import torch
        cfg = self._cfg

        # ── Importar funciones de infer.py ───────────────────────────────────
        infer_path = Path(__file__).resolve().parent.parent / "infer.py"
        import importlib.util
        spec = importlib.util.spec_from_file_location("infer_module", str(infer_path))
        infer_mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(infer_mod)

        # ── Cargar modelo ─────────────────────────────────────────────────────
        self.log_line.emit("[Inferencia] Cargando modelo…")
        device = cfg["device"]
        model, num_classes, class_names = infer_mod.load_model(cfg["model_path"], device)
        self.log_line.emit(f"[Inferencia] Clases: {num_classes} — {class_names}")

        # ── Preparar cloud dict desde pc ─────────────────────────────────────
        pc = cfg["pc"]
        xyz       = pc.xyz.astype(np.float32)
        intensity = pc.intensity if hasattr(pc, 'intensity') and pc.intensity is not None \
                    else np.zeros(len(xyz), np.float32)
        rgb = pc.rgb if hasattr(pc, 'rgb') and pc.rgb is not None else None

        cloud = {"xyz": xyz, "intensity": intensity, "rgb": rgb}

        # ── Inferencia por parches ────────────────────────────────────────────
        self.log_line.emit(f"[Inferencia] {len(xyz):,} puntos · parches {cfg['pts']} · "
                           f"overlap {cfg['overlap']:.0%} · batch {cfg['batch']}")

        # Patch progress callback
        _last_pct = [0.0]
        def _progress_hook(pct):
            if pct - _last_pct[0] >= 2.0:
                _last_pct[0] = pct
                self.progress.emit(pct)

        # Monkey-patch print to emit log lines
        _orig_print = __builtins__['print'] if isinstance(__builtins__, dict) else print
        def _my_print(*args, **kwargs):
            msg = " ".join(str(a) for a in args)
            self.log_line.emit(msg)
            if "%" in msg:
                try:
                    pct = float(msg.split("[")[1].split("%]")[0].strip())
                    self.progress.emit(pct)
                except Exception:
                    pass
        import builtins; builtins.print = _my_print

        try:
            predictions, confidence = infer_mod.infer_cloud(
                model, cloud,
                num_classes = num_classes,
                pts         = cfg["pts"],
                overlap     = cfg["overlap"],
                batch_size  = cfg["batch"],
                device      = device,
                return_confidence = True,
            )
        finally:
            builtins.print = _orig_print

        self.progress.emit(100.0)
        self.log_line.emit(f"[Inferencia] Completada — {int((predictions>0).sum()):,} "
                           f"puntos clasificados · confianza media "
                           f"{float(confidence.mean()):.0%}")
        self.finished_ok.emit(predictions, confidence)


# ── Worker de inferencia por lote (carpeta completa) ───────────────────────────

class BatchInferWorker(QThread):
    """
    Corre inferencia sobre TODOS los archivos de nube soportados de una
    carpeta, sin necesidad de cargarlos uno por uno en el canvas — antes
    solo se podía inferir la nube activa; para procesar un dataset
    completo de producción había que repetir el flujo manualmente
    archivo por archivo.

    El modelo se carga UNA vez y se reutiliza para todos los archivos.
    Cada archivo se guarda como '<stem>_classified.las' (mismo formato
    que produce infer.py / la inferencia individual) en el directorio
    de salida.
    """
    file_progress  = pyqtSignal(int, int, str)   # (índice, total, nombre archivo)
    log_line       = pyqtSignal(str)
    finished_batch = pyqtSignal(int, int)        # (n_ok, n_error)
    error          = pyqtSignal(str)

    SUPPORTED_EXTS = (".las", ".laz", ".ply", ".npy", ".txt", ".csv", ".xyz", ".pts", ".asc")

    def __init__(self, config: dict):
        super().__init__()
        self._cfg  = config
        self._stop = False

    def stop(self): self._stop = True

    def run(self):
        try:
            self._run_batch()
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()[:600]}")

    def _run_batch(self):
        cfg = self._cfg
        in_dir  = Path(cfg["input_dir"])
        out_dir = Path(cfg["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(p for p in in_dir.iterdir()
                       if p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTS)
        if not files:
            self.error.emit(f"No se encontraron nubes soportadas en:\n{in_dir}")
            return

        infer_path = Path(__file__).resolve().parent.parent / "infer.py"
        import importlib.util
        spec = importlib.util.spec_from_file_location("infer_module", str(infer_path))
        infer_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(infer_mod)

        device = cfg["device"]
        self.log_line.emit(f"[Lote] Cargando modelo…")
        model, num_classes, class_names = infer_mod.load_model(cfg["model_path"], device)
        self.log_line.emit(f"[Lote] {len(files)} archivo(s) encontrados en {in_dir}")

        n_ok = n_err = 0
        for i, path in enumerate(files, start=1):
            if self._stop:
                self.log_line.emit("[Lote] Detenido por el usuario."); break
            self.file_progress.emit(i, len(files), path.name)
            self.log_line.emit(f"\n[{i}/{len(files)}] {path.name}")
            try:
                cloud = infer_mod.load_cloud(str(path))
                self.log_line.emit(f"  {len(cloud['xyz']):,} puntos")
                predictions = infer_mod.infer_cloud(
                    model, cloud, num_classes=num_classes,
                    pts=cfg["pts"], overlap=cfg["overlap"],
                    batch_size=cfg["batch"], device=device)
                out_path = out_dir / f"{path.stem}_classified.las"
                infer_mod.save_result(cloud, predictions, class_names, str(out_path))
                self.log_line.emit(f"  → {out_path}")
                n_ok += 1
            except Exception as e:
                self.log_line.emit(f"  [ERROR] {path.name}: {e}")
                n_err += 1

        self.log_line.emit(f"\n[Lote] Completado: {n_ok} OK, {n_err} con error.")
        self.finished_batch.emit(n_ok, n_err)


# ── Panel de inferencia ───────────────────────────────────────────────────────

class InferPanel(QWidget):
    """
    Panel a pantalla completa (igual que TrainingPanel).
    Emite inference_done(predictions, confidence) cuando termina para que
    main_window aplique las predicciones al label_store y guarde la
    confianza en pc.confidence (habilita el modo de color "Confianza").
    """
    inference_done = pyqtSignal(object, object)   # predictions (N,) uint8, confidence (N,) float32
    inference_stop = pyqtSignal()

    # Estilos reutilizables
    _S_LABEL = f"color:{TEXT_MUTE};font-size:10.5px;"
    _S_SPIN  = (f"QSpinBox,QDoubleSpinBox{{background:{SURFACE_2};border:1px solid {BORDER};"
                f"border-radius:3px;color:{TEXT_DIM};padding:3px 5px;font-size:10.5px;}}"
                f"QSpinBox:hover,QDoubleSpinBox:hover{{border-color:{ACCENT};}}")
    _S_BTN   = (f"QPushButton{{background:{SURFACE};border:1px solid {BORDER};border-radius:4px;"
                f"color:{TEXT_DIM};padding:6px 12px;font-size:10.5px;}}"
                f"QPushButton:hover{{border-color:{ACCENT};color:{ACCENT_STRONG};}}")

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker:   Optional[InferWorker] = None
        self._batch_worker: Optional[BatchInferWorker] = None
        self._model_path: Optional[str] = None
        self._batch_input_dir: Optional[str] = None
        self._batch_output_dir: Optional[str] = None
        self._pc = None
        self._project = None
        self._start_t = 0.0
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)
        self.setStyleSheet(f"background:{SURFACE_2};")

        # Header
        hdr = QWidget()
        hdr.setStyleSheet(f"background:{SURFACE};border-bottom:1px solid {BORDER};")
        hdr.setFixedHeight(38)
        hl = QHBoxLayout(hdr); hl.setContentsMargins(14,0,14,0); hl.setSpacing(8)
        ic = QLabel(); ic.setPixmap(qpixmap("magic", ACCENT_STRONG, 15))
        hl.addWidget(ic)
        t = QLabel("Inferencia")
        t.setStyleSheet(f"color:{ACCENT_STRONG};font-size:11.5px;font-weight:700;")
        hl.addWidget(t)
        sub = QLabel("clasificar nube con modelo entrenado")
        sub.setStyleSheet(f"color:{TEXT_MUTE};font-size:10.5px;")
        hl.addWidget(sub)
        hl.addStretch()
        self._elapsed = QLabel("00:00:00")
        self._elapsed.setStyleSheet(f"color:{TEXT_DIM};font-size:10.5px;font-family:'Consolas';")
        hl.addWidget(self._elapsed)
        root.addWidget(hdr)

        # Body: left config | right log
        from PyQt5.QtWidgets import QSplitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(4)
        splitter.setStyleSheet(f"QSplitter::handle{{background:{BORDER_SOFT};}}")

        # ── Left: config ─────────────────────────────────────────────────────
        left = QWidget(); left.setStyleSheet(f"background:{SURFACE_2};"); left.setMinimumWidth(280)
        ll = QVBoxLayout(left); ll.setContentsMargins(12,12,12,12); ll.setSpacing(10)

        # Modelo
        m_gb, ml = self._card("Modelo entrenado", "cpu")

        model_row = QHBoxLayout(); model_row.setSpacing(6)
        self._model_lbl = QLabel("(no seleccionado)")
        self._model_lbl.setStyleSheet(f"color:{TEXT_DIM};font-size:10.5px;")
        self._model_lbl.setWordWrap(True)
        model_row.addWidget(self._model_lbl, 1)
        browse_btn = self._icon_btn("folder2")
        browse_btn.clicked.connect(self._browse_model)
        model_row.addWidget(browse_btn)
        ml.addLayout(model_row)

        # Info del modelo (min-height fijo: evita que el recorte de texto
        # dependa del timing de heightForWidth mientras el QSplitter aún
        # está asentando anchos — con 3 líneas típicas de contenido no
        # queremos que el cuadro quede más bajo de lo que necesita)
        self._model_info = QLabel("")
        self._model_info.setWordWrap(True)
        self._model_info.setMinimumHeight(56)
        self._model_info.setStyleSheet(
            f"color:{OK};font-size:10.5px;background:{OK_SOFT};"
            f"border:1px solid {OK_SOFT};border-radius:3px;padding:6px;")
        self._model_info.setVisible(False)
        ml.addWidget(self._model_info)
        ll.addWidget(m_gb)

        # Parámetros
        p_gb, pl = self._card("Parámetros de inferencia", "sliders2")
        pl.addLayout(self._row("Pts/parche", self._spin(8192, 512, 65536, step=512)))
        self._pts_spin = pl.itemAt(pl.count()-1).layout().itemAt(1).widget()
        pl.addLayout(self._row("Overlap",     self._dspin(0.5, 0.0, 0.9)))
        self._ov_spin = pl.itemAt(pl.count()-1).layout().itemAt(1).widget()
        pl.addLayout(self._row("Batch",       self._spin(4, 1, 32)))
        self._batch_spin = pl.itemAt(pl.count()-1).layout().itemAt(1).widget()

        # Device
        from PyQt5.QtWidgets import QComboBox
        self._dev_combo = QComboBox()
        self._dev_combo.setStyleSheet(
            f"QComboBox{{background:{SURFACE_2};border:1px solid {BORDER};border-radius:3px;"
            f"color:{TEXT_DIM};padding:3px 6px;font-size:10.5px;}}"
            f"QComboBox:hover{{border-color:{ACCENT};}}"
            f"QComboBox QAbstractItemView{{background:{SURFACE};color:{TEXT_DIM};"
            f"selection-background-color:{ACCENT_SOFT};selection-color:{ACCENT_STRONG};}}")
        try:
            import torch
            if torch.cuda.is_available():
                self._dev_combo.addItem(f"CUDA ({torch.cuda.get_device_name(0)})", "cuda")
        except ImportError:
            pass
        self._dev_combo.addItem("CPU", "cpu")
        pl.addLayout(self._row("Dispositivo", self._dev_combo))
        ll.addWidget(p_gb)

        # Inferencia por lote (carpeta completa) — antes solo se podía
        # inferir la nube activa del canvas; para clasificar una carpeta
        # completa de nubes (uso de producción típico) había que repetir
        # el flujo manualmente para cada archivo. Reutiliza los mismos
        # parámetros (pts/overlap/batch/dispositivo) y el mismo modelo
        # de arriba.
        b_gb, bl = self._card("Inferencia por lote (carpeta)", "folder2")
        bl.addWidget(self._desc(
            "Clasifica todos los archivos de nube de una carpeta con el "
            "modelo de arriba, sin abrirlos uno por uno."))

        in_row = QHBoxLayout(); in_row.setSpacing(6)
        self._batch_in_lbl = QLabel("(sin elegir)")
        self._batch_in_lbl.setStyleSheet(f"color:{TEXT_DIM};font-size:10px;")
        self._batch_in_lbl.setWordWrap(True)
        in_row.addWidget(self._batch_in_lbl, 1)
        in_btn = self._icon_btn("folder2")
        in_btn.setToolTip("Carpeta con las nubes a clasificar")
        in_btn.clicked.connect(self._browse_batch_input)
        in_row.addWidget(in_btn)
        bl.addLayout(in_row)

        self._batch_count_lbl = QLabel("")
        self._batch_count_lbl.setStyleSheet(f"color:{TEXT_MUTE};font-size:9.5px;")
        bl.addWidget(self._batch_count_lbl)

        self._batch_btn = QPushButton("  Ejecutar por lote")
        self._batch_btn.setIcon(qicon("play-fill", ACCENT_STRONG))
        self._batch_btn.setEnabled(False)
        self._batch_btn.setStyleSheet(
            f"QPushButton{{background:{ACCENT_SOFT};border:1px solid {ACCENT_SOFT};"
            f"border-radius:4px;color:{ACCENT_STRONG};padding:8px;font-size:10.5px;"
            f"font-weight:600;text-align:left;}}"
            f"QPushButton:hover{{border-color:{ACCENT};}}"
            f"QPushButton:disabled{{background:{SURFACE_2};color:{TEXT_MUTE};border-color:{SURFACE_2};}}")
        self._batch_btn.clicked.connect(self._on_run_batch)
        bl.addWidget(self._batch_btn)
        ll.addWidget(b_gb)

        # Nota explicativa
        note = QLabel(
            "Al terminar la inferencia, las predicciones se aplican directamente "
            "a la nube en el canvas como anotaciones.\n\n"
            "Puedes revisarlas y corregirlas con las herramientas de etiquetado "
            "antes de guardar el proyecto.")
        note.setStyleSheet(
            f"color:{TEXT_DIM};font-size:10.5px;background:{ACCENT_SOFT};"
            f"border-radius:4px;padding:9px;border-left:3px solid {ACCENT};")
        note.setWordWrap(True)
        ll.addWidget(note)

        # Botones
        btn_row = QHBoxLayout(); btn_row.setSpacing(6)
        self._run_btn = QPushButton("  Ejecutar inferencia")
        self._run_btn.setIcon(qicon("play-fill", "#ffffff"))
        self._run_btn.setEnabled(False)
        self._run_btn.setStyleSheet(
            f"QPushButton{{background:{ACCENT};color:#ffffff;border:none;"
            f"border-radius:4px;padding:9px;font-size:10.5px;font-weight:600;text-align:left;}}"
            f"QPushButton:hover{{background:{ACCENT_STRONG};}}"
            f"QPushButton:disabled{{background:{ACCENT_SOFT};color:{TEXT_MUTE};}}")
        self._run_btn.clicked.connect(self._on_run)

        self._stop_btn = QPushButton("  Detener")
        self._stop_btn.setIcon(qicon("stop-fill", WARN))
        self._stop_btn.setEnabled(False)
        self._stop_btn.setStyleSheet(
            f"QPushButton{{background:{WARN_SOFT};border:1px solid {WARN_SOFT};"
            f"border-radius:4px;padding:9px;font-size:10.5px;color:{WARN};font-weight:600;}}"
            f"QPushButton:hover{{border-color:{WARN};}}"
            f"QPushButton:disabled{{color:{TEXT_MUTE};background:{SURFACE_2};border-color:{SURFACE_2};}}")
        self._stop_btn.clicked.connect(self._on_stop)

        btn_row.addWidget(self._run_btn, 1); btn_row.addWidget(self._stop_btn)
        ll.addLayout(btn_row)

        self._progress = QProgressBar()
        self._progress.setRange(0,100); self._progress.setValue(0)
        self._progress.setFixedHeight(6); self._progress.setTextVisible(False)
        self._progress.setStyleSheet(
            f"QProgressBar{{background:{SURFACE_3};border:none;border-radius:3px;}}"
            f"QProgressBar::chunk{{background:{ACCENT};border-radius:3px;}}")
        ll.addWidget(self._progress)
        ll.addStretch()

        splitter.addWidget(left)

        # ── Right: log ────────────────────────────────────────────────────────
        right = QWidget(); right.setStyleSheet(f"background:{SURFACE_2};")
        rl = QVBoxLayout(right); rl.setContentsMargins(12,12,12,12); rl.setSpacing(8)

        log_gb, logl = self._card("Log de inferencia", "info-circle")
        logl.setContentsMargins(0,0,0,0)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setStyleSheet(
            f"QTextEdit{{background:{SURFACE_2};border:1px solid {BORDER_SOFT};border-radius:3px;"
            f"color:{TEXT_DIM};font-family:'Consolas';font-size:10.5px;padding:5px;}}")
        logl.addWidget(self._log)
        rl.addWidget(log_gb, 1)

        splitter.addWidget(right)
        splitter.setSizes([300, 700])
        root.addWidget(splitter, 1)

        # Timer
        from PyQt5.QtCore import QTimer
        self._timer = QTimer()
        self._timer.timeout.connect(self._update_elapsed)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _card(self, title, icon_name=None):
        """Tarjeta con encabezado que envuelve — mismo patrón que ToolPanel/GeoPanel."""
        frame = QFrame(); frame.setObjectName("inferCard")
        frame.setStyleSheet(
            f"QFrame#inferCard{{background:{SURFACE};border:1px solid {BORDER};border-radius:4px;}}")
        outer = QVBoxLayout(frame)
        outer.setContentsMargins(11, 10, 11, 11); outer.setSpacing(7)
        hdr_row = QHBoxLayout(); hdr_row.setSpacing(6)
        if icon_name:
            ic = QLabel(); ic.setStyleSheet("background:transparent;border:none;")
            ic.setPixmap(qpixmap(icon_name, ACCENT_STRONG, 13))
            hdr_row.addWidget(ic)
        title_lbl = QLabel(title); title_lbl.setWordWrap(True)
        title_lbl.setStyleSheet(f"color:{ACCENT_STRONG};font-size:10.5px;font-weight:700;background:transparent;")
        hdr_row.addWidget(title_lbl, 1)
        outer.addLayout(hdr_row)
        content = QVBoxLayout(); content.setSpacing(6)
        outer.addLayout(content)
        return frame, content

    def _desc(self, text: str) -> QLabel:
        d = QLabel(text)
        d.setWordWrap(True)
        d.setStyleSheet(f"color:{TEXT_MUTE};font-size:10px;background:transparent;")
        return d

    def _icon_btn(self, icon_name):
        b = QPushButton(); b.setIcon(qicon(icon_name, TEXT_DIM)); b.setFixedSize(26, 24)
        b.setStyleSheet(
            f"QPushButton{{background:{SURFACE_2};border:1px solid {BORDER};"
            f"border-radius:3px;}}"
            f"QPushButton:hover{{border-color:{ACCENT};background:{ACCENT_SOFT};}}")
        return b

    def _row(self, label, widget):
        hl = QHBoxLayout(); hl.setSpacing(6)
        lbl = QLabel(label); lbl.setStyleSheet(self._S_LABEL); lbl.setMinimumWidth(70)
        lbl.setWordWrap(True)
        hl.addWidget(lbl); hl.addWidget(widget, 1)
        return hl

    def _spin(self, default, mn, mx, step=1):
        s = QSpinBox(); s.setRange(mn,mx); s.setValue(default); s.setSingleStep(step)
        s.setStyleSheet(self._S_SPIN); return s

    def _dspin(self, default, mn, mx):
        s = QDoubleSpinBox(); s.setRange(mn,mx); s.setValue(default)
        s.setDecimals(2); s.setSingleStep(0.1); s.setStyleSheet(self._S_SPIN); return s

    # ── API pública ───────────────────────────────────────────────────────────

    def set_context(self, pc, project):
        """Recibe la nube y el proyecto para la inferencia."""
        self._pc      = pc
        self._project = project
        self._update_run_btn()

    def _update_run_btn(self):
        ready = (self._model_path is not None and
                 self._pc is not None and
                 self._pc.n_points > 0)
        self._run_btn.setEnabled(ready)

    # ── Slots ────────────────────────────────────────────────────────────────

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar modelo entrenado", "",
            "Modelos PyTorch (*.pth *.pt);;Todos (*)")
        if not path:
            return
        self._model_path = path
        self._model_lbl.setText(f"…/{Path(path).parent.name}/{Path(path).name}")
        self._model_lbl.setToolTip(path)

        # Mostrar info del modelo
        try:
            import torch
            ckpt = torch.load(path, map_location="cpu")
            arch  = ckpt.get("arch", "RandLA-Net")
            nc    = ckpt.get("num_classes", "?")
            miou  = ckpt.get("miou", 0.0)
            ep    = ckpt.get("epoch", "?")
            names = ckpt.get("class_names", [])
            self._model_info.setText(
                f"Arquitectura: {arch}\n"
                f"Clases: {nc}  |  Mejor mIoU: {miou:.4f}  |  Epoch: {ep}\n"
                f"Clases: {', '.join(names[:6])}{'…' if len(names)>6 else ''}")
            self._model_info.setVisible(True)
        except Exception as e:
            self._model_info.setText(f"Error leyendo modelo: {e}")
            self._model_info.setVisible(True)

        self._update_run_btn()
        self._update_batch_btn()

    def _on_run(self):
        if self._pc is None or self._model_path is None:
            return

        device = self._dev_combo.currentData() or "cpu"
        cfg = {
            "model_path": self._model_path,
            "pc":         self._pc,
            "pts":        self._pts_spin.value(),
            "overlap":    self._ov_spin.value(),
            "batch":      self._batch_spin.value(),
            "device":     device,
        }

        self._log.clear()
        self._progress.setValue(0)
        self._run_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._start_t = time.time()
        self._timer.start(1000)

        self._worker = InferWorker(cfg)
        self._worker.log_line.connect(self._on_log)
        self._worker.progress.connect(lambda p: self._progress.setValue(int(p)))
        self._worker.finished_ok.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_stop(self):
        if self._worker: self._worker.stop()
        if self._batch_worker: self._batch_worker.stop()
        self._stop_btn.setEnabled(False)
        self._log.append("[Inferencia] Deteniendo…")

    def _browse_batch_input(self):
        d = QFileDialog.getExistingDirectory(self, "Carpeta con nubes a clasificar")
        if not d:
            return
        self._batch_input_dir = d
        self._batch_output_dir = str(Path(d) / "batch_output")
        self._batch_in_lbl.setText(f"…/{Path(d).name}")
        self._batch_in_lbl.setToolTip(d)
        n = sum(1 for p in Path(d).iterdir()
                if p.is_file() and p.suffix.lower() in BatchInferWorker.SUPPORTED_EXTS)
        self._batch_count_lbl.setText(
            f"{n} archivo(s) soportado(s) — salida en …/batch_output")
        self._update_batch_btn()

    def _update_batch_btn(self):
        ready = (self._model_path is not None and self._batch_input_dir is not None)
        self._batch_btn.setEnabled(ready)

    def _on_run_batch(self):
        if self._model_path is None or self._batch_input_dir is None:
            return
        device = self._dev_combo.currentData() or "cpu"
        cfg = {
            "model_path": self._model_path,
            "input_dir":  self._batch_input_dir,
            "output_dir": self._batch_output_dir,
            "pts":        self._pts_spin.value(),
            "overlap":    self._ov_spin.value(),
            "batch":      self._batch_spin.value(),
            "device":     device,
        }
        self._log.clear()
        self._progress.setRange(0, 0)   # indeterminado: el % por archivo no es representativo del lote
        self._run_btn.setEnabled(False)
        self._batch_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._start_t = time.time()
        self._timer.start(1000)

        self._batch_worker = BatchInferWorker(cfg)
        self._batch_worker.log_line.connect(self._on_log)
        self._batch_worker.file_progress.connect(self._on_batch_file_progress)
        self._batch_worker.finished_batch.connect(self._on_batch_done)
        self._batch_worker.error.connect(self._on_error)
        self._batch_worker.start()

    def _on_batch_file_progress(self, i: int, total: int, filename: str):
        self._elapsed.setToolTip(f"Archivo {i}/{total}: {filename}")

    def _on_batch_done(self, n_ok: int, n_err: int):
        self._timer.stop()
        self._progress.setRange(0, 100); self._progress.setValue(100)
        self._run_btn.setEnabled(True)
        self._batch_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        elapsed = time.time() - self._start_t
        self._log.append(f"\n✓ Lote completado en {elapsed:.0f}s — {n_ok} OK, {n_err} con error.")
        self._log.append(f"Resultados en: {self._batch_output_dir}")

    def _on_log(self, line: str):
        self._log.append(line)
        self._log.moveCursor(self._log.textCursor().End)

    def _on_done(self, predictions: np.ndarray, confidence: np.ndarray):
        self._timer.stop()
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._progress.setValue(100)
        elapsed = time.time() - self._start_t
        self._log.append(f"\n✓ Inferencia completada en {elapsed:.0f}s")
        self._log.append("Aplicando predicciones al canvas…")
        self.inference_done.emit(predictions, confidence)

    def _on_error(self, msg: str):
        self._timer.stop()
        self._progress.setRange(0, 100)
        self._run_btn.setEnabled(True)
        self._batch_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._log.append(f"\n[ERROR]\n{msg}")

    def _update_elapsed(self):
        e = int(time.time() - self._start_t)
        h, r = divmod(e, 3600); m, s = divmod(r, 60)
        self._elapsed.setText(f"{h:02d}:{m:02d}:{s:02d}")
