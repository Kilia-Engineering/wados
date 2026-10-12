#!/usr/bin/env python3
"""
Optimization Tab for Waverider GUI
Provides interface for multi-objective optimization with NSGA-II
"""

import sys
import os
import json
from datetime import datetime
from pathlib import Path
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QGroupBox, QGridLayout, QSlider,
                             QDoubleSpinBox, QSpinBox, QCheckBox, QComboBox,
                             QProgressBar, QTextEdit, QTabWidget, QFileDialog,
                             QMessageBox, QSplitter, QFrame, QTableWidget,
                             QTableWidgetItem, QHeaderView, QAbstractScrollArea,
                             QSizePolicy)
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QFont
import numpy as np

# Add project paths (flexible for different OS)
script_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.exists('/mnt/project'):
    sys.path.insert(0, '/mnt/project')
sys.path.insert(0, script_dir)

# Lazy imports - only import when needed to prevent Windows import errors
# These will be imported in the worker thread
# from optimization_engine import WaveriderProblem
# from optimization_utils import OptimizationResults


# Rough seconds per design for the time estimate (mesh + PySAGAS, one core)
METHOD_SECONDS = {"ltoc": 10, "gvwd": 1, "pswr": 2, "planar": 4, "shadow": 2,
                  "hybrid": 3, "vmn": 3, "vmof": 4, "vmplo": 4, "mfof": 5}


def _corrections_text(corrections) -> str:
    """Log line for the base-pressure and friction settings (aero_corrections)."""
    from aero_corrections import INVISCID, AeroCorrections
    c = AeroCorrections.from_dict(corrections) if corrections else INVISCID
    return c.describe()


class OptimizationWorker(QThread):
    """Worker thread for running optimization in background"""
    
    # Signals
    progress_update = pyqtSignal(int, int)  # current_gen, total_gen
    design_evaluated = pyqtSignal(int, dict)  # design_id, metrics
    generation_complete = pyqtSignal(int, dict)  # generation, stats
    optimization_complete = pyqtSignal(str)  # results_folder
    error_occurred = pyqtSignal(str)  # error_message
    log_message = pyqtSignal(str)  # console message
    
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.should_stop = False
        self.algorithm = None
        
    def run(self):
        """Run the optimization"""
        try:
            # Lazy import to prevent Windows import errors at module level
            from optimization_engine import WaveriderProblem
            from optimization_utils import OptimizationResults
            
            self.log_message.emit("=" * 60)
            self.log_message.emit("OPTIMIZATION STARTED")
            self.log_message.emit("=" * 60)
            self.log_message.emit(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.log_message.emit("")
            
            # Extract configuration
            objectives = self.config['objectives']
            constraints = self.config['constraints']
            design_vars = self.config['design_variables']
            algorithm_params = self.config['algorithm']
            fixed_params = self.config['fixed_parameters']
            sim_params = self.config.get('simulation_params', {})
            
            # Log configuration
            obj_strs = [f"{'Volumetric Efficiency' if obj['name'] == 'VolEff' else obj['name']} "
                        f"({obj['mode']})" for obj in objectives]
            self.log_message.emit(f"📊 Objectives: {', '.join(obj_strs)}")
            self.log_message.emit(f"🔒 Constraints: {len(constraints)} active")
            self.log_message.emit(f"🧬 Population: {algorithm_params['pop_size']} designs/generation")
            self.log_message.emit(f"🔄 Generations: {algorithm_params['n_gen']}")
            self.log_message.emit(f"⚙️  CPU Cores: {algorithm_params['n_cores']}")
            self.log_message.emit("")
            
            # Initialize results manager
            results = OptimizationResults(base_dir=self.config.get('results_dir', 'results'))
            results.save_config(self.config)
            self.log_message.emit(f"📁 Results folder: {results.results_folder}")
            
            # Initialize CSV with proper headers
            design_var_names = [v['name'] for v in design_vars] or ['X1', 'X2', 'X3', 'X4']
            objective_names = [obj['name'] for obj in objectives]
            constraint_names = [c['name'] for c in constraints if c.get('active', True) and c['name'] != 'design_space']
            results.initialize_designs_csv(design_var_names, objective_names, constraint_names)
            
            self.log_message.emit("")
            
            # Create problem
            self.log_message.emit("🔧 Creating optimization problem...")
            method_case = None
            if self.config.get('method_case'):
                # A design tab's case (method_optimization): its own builder,
                # PySAGAS on the closed mesh, A_ref = each design's planform area
                from method_optimization import MethodProblem, OptimizationCase, evaluate_design
                method_case = OptimizationCase.from_dict(self.config['method_case'])
                flow = {'mach': fixed_params['M_inf'],
                        'pressure': sim_params.get('pressure', 2549.0),
                        'temperature': sim_params.get('temperature', 221.55),
                        'aoa': sim_params.get('aoa', 0.0),
                        'corrections': sim_params.get('corrections')}
                self.log_message.emit(f"   Method: {method_case.title}")
                self.log_message.emit(f"   Flight: M∞={flow['mach']:g}, α={flow['aoa']}°, "
                                      f"P∞={flow['pressure']:.1f} Pa, T∞={flow['temperature']:.2f} K")
                self.log_message.emit(f"   {_corrections_text(flow['corrections'])}")
                for v in method_case.active():
                    self.log_message.emit(f"   {v.label}: {v.lower:g} to {v.upper:g} (now {v.value:g})")
                problem = MethodProblem(method_case, objectives, constraints, flow=flow,
                                        n_cores=algorithm_params['n_cores'])
                self.log_message.emit("✓ Problem created successfully")
                self.log_message.emit("   Reference area: each design's planform area")
                base = evaluate_design(method_case.method, method_case.params(), flow)
                if base['success']:
                    self.log_message.emit(
                        f"   Starting design: CL/CD={base['CL'] / base['CD']:.3f}, "
                        f"CL={base['CL']:.4f}, CD={base['CD']:.5f}, Volume={base['Volume']:.4g} m³, "
                        f"Volumetric Efficiency={base['VolEff']:.4f} ({base['time']:.1f} s)")
                    self.log_message.emit(
                        f"   CD = {base['CD_inviscid']:.5f} inviscid {base['CD_base']:+.5f} base "
                        f"+ {base['CD_friction']:.5f} friction")
                else:
                    self.log_message.emit(f"   ⚠ Starting design failed: {base['error']}")
                self.log_message.emit("")
            else:
                self.log_message.emit(f"   M_inf={fixed_params['M_inf']}, β={fixed_params['beta']}°")
                self.log_message.emit(f"   height={fixed_params['height']}m, width={fixed_params['width']}m")
                self.log_message.emit(f"   α={sim_params.get('aoa', 0.0)}°, P∞={sim_params.get('pressure', 2549.0):.1f} Pa, T∞={sim_params.get('temperature', 221.55):.2f} K")
                self.log_message.emit(f"   {_corrections_text(sim_params.get('corrections'))}")
                self.log_message.emit(f"   mesh_size={algorithm_params.get('mesh_size', 0.1)}")
                problem = WaveriderProblem(
                    objectives=objectives,
                    constraints=constraints,
                    M_inf=fixed_params['M_inf'],
                    beta=fixed_params['beta'],
                    height=fixed_params['height'],
                    width=fixed_params['width'],
                    aoa=sim_params.get('aoa', 0.0),
                    pressure=sim_params.get('pressure'),
                    temperature=sim_params.get('temperature'),
                    A_ref=sim_params.get('A_ref'),
                    mesh_size=algorithm_params.get('mesh_size', 0.1),
                    n_cores=algorithm_params['n_cores'],
                    match_shockwave=sim_params.get('match_shockwave', False),
                    corrections=sim_params.get('corrections'),
                    bounds=[(v['min'], v['max']) for v in design_vars] if len(design_vars) == 4 else None,
                    verbose=True  # Enable verbose mode to see errors
                )
                self.log_message.emit("✓ Problem created successfully")
                self.log_message.emit(f"   Reference area: {problem.A_ref:.4f} m²")
                self.log_message.emit("")
            
            # Setup algorithm with genetic operators
            from pymoo.algorithms.moo.nsga2 import NSGA2
            from pymoo.operators.crossover.sbx import SBX
            from pymoo.operators.mutation.pm import PM
            from pymoo.optimize import minimize
            
            # Get GA parameters
            crossover_prob = algorithm_params.get('crossover_prob', 0.9)
            mutation_prob = algorithm_params.get('mutation_prob', 0.1)
            
            self.log_message.emit(f"🧬 GA Parameters: crossover={crossover_prob:.2f}, mutation={mutation_prob:.2f}")
            
            self.algorithm = NSGA2(
                pop_size=algorithm_params['pop_size'],
                crossover=SBX(prob=crossover_prob, eta=15),
                mutation=PM(prob=mutation_prob, eta=20),
                eliminate_duplicates=True
            )
            
            # Callback to track progress
            design_counter = [0]  # Use list to allow modification in nested function
            from method_optimization import objective_value
            
            def callback_func(algorithm):
                if self.should_stop:
                    algorithm.termination.force_termination = True
                    return
                    
                gen = algorithm.n_gen
                n_gen = algorithm_params['n_gen']
                
                # Update progress
                self.progress_update.emit(gen, n_gen)
                
                # Log current generation
                if gen == 1:
                    self.log_message.emit(f"🚀 Starting Generation {gen}/{n_gen}...")
                else:
                    self.log_message.emit(f"📈 Generation {gen}/{n_gen} complete")
                
                # The designs evaluated in this generation, with their results
                # (the population after survival is a different set)
                batch_X = getattr(problem, 'last_eval_X', None)
                batch_results = getattr(problem, 'last_eval_results', None) or []
                if batch_X is None or len(batch_X) != len(batch_results):
                    batch_X = [ind.X for ind in algorithm.pop]
                    batch_results = [None] * len(batch_X)
                
                # Count successes and failures
                n_success = 0
                n_failed = 0
                
                for i, (X_vals, actual_result) in enumerate(zip(batch_X, batch_results)):
                    success = bool(actual_result and actual_result.get('success', False))
                    if success:
                        n_success += 1
                    else:
                        n_failed += 1
                    
                    design_counter[0] += 1
                    design_id = design_counter[0]
                    x_str = ", ".join(f"{float(v):.4g}" for v in X_vals)
                    
                    # Log first few designs in detail
                    if i < 3 or not success:
                        self.log_message.emit(f"   Design {design_id}: X=[{x_str}]")
                        if success:
                            self.log_message.emit(
                                f"      CL={actual_result['CL']:.4f}, CD={actual_result['CD']:.5f}, "
                                f"Volume={actual_result['Volume']:.4g}")
                        elif actual_result:
                            self.log_message.emit(f"      Error: {actual_result.get('error', '')}")
                    
                    # Build design variables dict
                    design_variables_dict = {name: float(X_vals[j])
                                             for j, name in enumerate(design_var_names)}
                    
                    # Objectives with their actual values (not pymoo's F,
                    # which is negated for maximization and penalized on failure)
                    objectives_dict = {}
                    for obj in objectives:
                        objectives_dict[obj['name']] = (
                            float(objective_value(actual_result, obj['name'])) if success
                            else float('nan'))
                    
                    # Constraints: the constrained quantity
                    constraints_dict = {}
                    for const in constraints:
                        const_name = const['name']
                        if not const.get('active', True) or const_name == 'design_space':
                            continue
                        key = {'CL_min': 'CL', 'CD_max': 'CD', 'Cm_max': 'Cm',
                               'Volume_min': 'Volume'}.get(const_name)
                        constraints_dict[const_name] = (
                            float(actual_result[key]) if success and key else float('nan'))
                    
                    # Log to results manager
                    results.log_design(
                        design_vars=design_variables_dict,
                        objectives=objectives_dict,
                        constraints=constraints_dict,
                        generation=gen,
                        success=success,
                        eval_time=float(actual_result.get('time', 0.0)) if actual_result else 0.0
                    )
                    
                    # Emit signal for real-time update
                    metrics = {
                        'design_id': design_id,
                        'generation': gen,
                        'success': success,
                        **design_variables_dict,
                        **objectives_dict,
                        **constraints_dict
                    }
                    if success:
                        metrics['CL'] = actual_result['CL']
                        metrics['CD'] = actual_result['CD']
                        metrics['Volume'] = actual_result['Volume']
                        if actual_result['CD'] > 0:
                            metrics['CL/CD'] = actual_result['CL'] / actual_result['CD']
                    
                    self.design_evaluated.emit(design_id, metrics)
                
                # Generation statistics
                gen_stats = {
                    'generation': gen,
                    'designs_evaluated': design_counter[0],
                    'designs_in_gen': len(batch_X)
                }
                self.generation_complete.emit(gen, gen_stats)
                self.log_message.emit(f"   Designs evaluated: {design_counter[0]} (✓{n_success} / ✗{n_failed})")
                self.log_message.emit("")
            
            # Run optimization
            self.log_message.emit("🏁 Starting optimization run...")
            self.log_message.emit("")
            
            from pymoo.termination import get_termination
            
            res = minimize(
                problem,
                self.algorithm,
                termination=get_termination("n_gen", algorithm_params['n_gen']),
                callback=callback_func,
                verbose=False,
                seed=1
            )
            
            if self.should_stop:
                self.log_message.emit("")
                self.log_message.emit("⏹ OPTIMIZATION STOPPED BY USER")
                self.log_message.emit("")
            else:
                self.log_message.emit("")
                self.log_message.emit("=" * 60)
                self.log_message.emit("✅ OPTIMIZATION COMPLETE")
                self.log_message.emit("=" * 60)
                self.log_message.emit("")
            
            # Extract Pareto front from pymoo result
            self.log_message.emit("📊 Extracting Pareto front...")
            objective_names = [obj['name'] for obj in objectives]
            minimize_flags = [obj['mode'].lower() == 'minimize' for obj in objectives]
            
            # Get Pareto-optimal designs from pymoo result
            pareto_designs = []
            if res.X is not None and len(res.X) > 0:
                # res.X contains the design variables of Pareto-optimal solutions
                # res.F contains the objective values
                X_pareto = res.X if res.X.ndim > 1 else res.X.reshape(1, -1)
                F_pareto = res.F if res.F.ndim > 1 else res.F.reshape(1, -1)
                
                for i in range(len(X_pareto)):
                    design = {
                        'X': X_pareto[i],
                        'objectives': {}
                    }
                    for j, obj_name in enumerate(objective_names):
                        # Note: F values may be negated for maximize objectives
                        design['objectives'][obj_name] = float(F_pareto[i, j])
                    pareto_designs.append(design)
                
                self.log_message.emit(f"✓ Found {len(pareto_designs)} Pareto-optimal designs")
            else:
                self.log_message.emit("⚠ No Pareto-optimal designs found")
            self.log_message.emit("")
            
            # Try to extract from CSV (the proper source of Pareto designs)
            try:
                pareto_df = results.extract_pareto_front(objective_names, minimize_flags)
                if pareto_df is not None and len(pareto_df) > 0:
                    self.log_message.emit(f"✓ Pareto front: {len(pareto_df)} designs")
                else:
                    pareto_df = None
                    self.log_message.emit("⚠ No successful designs for Pareto front")
            except Exception as e:
                self.log_message.emit(f"⚠ Could not extract Pareto front from CSV: {e}")
                pareto_df = None
            
            # Generate plots
            self.log_message.emit("📈 Generating plots...")
            try:
                results.plot_convergence(objective_names, minimize_flags)
                self.log_message.emit("✓ Convergence plot saved")
            except Exception as e:
                self.log_message.emit(f"⚠ Convergence plot failed: {e}")
            
            if len(objective_names) == 2:
                try:
                    results.plot_pareto_front(objective_names, minimize_flags)
                    self.log_message.emit("✓ Pareto front plot saved")
                except Exception as e:
                    self.log_message.emit(f"⚠ Pareto front plot failed: {e}")
            
            # Generate Pareto design files using pareto_df from CSV
            if method_case is not None and pareto_df is not None and len(pareto_df) > 0:
                # STL, the tab's parameter dict (JSON) and a summary per design
                from method_optimization import write_design_files
                self.log_message.emit("")
                self.log_message.emit("🔧 Writing Pareto design files...")
                n_written = 0
                for _, row in pareto_df.iterrows():
                    design_id = int(row['Design_ID'])
                    try:
                        params = method_case.params([row[name] for name in design_var_names])
                        info = {name: row[name] for name in objective_names if name in row}
                        write_design_files(method_case, params, results.pareto_designs_dir,
                                           f"design_{design_id:04d}", info)
                        n_written += 1
                    except Exception as e:
                        self.log_message.emit(f"   ⚠ design {design_id}: {e}")
                self.log_message.emit(f"✓ {n_written} Pareto designs written to "
                                      f"{results.pareto_designs_dir}")
            elif pareto_df is not None and len(pareto_df) > 0:
                self.log_message.emit("")
                self.log_message.emit("🔧 Generating CAD files for Pareto designs...")
                
                # Import waverider generator and CAD exporter
                from waverider_generator.generator import waverider as WaveriderGenerator
                from waverider_generator.cad_export import to_CAD
                
                results.generate_pareto_design_files(
                    pareto_df=pareto_df,
                    waverider_generator=WaveriderGenerator,
                    cad_exporter=to_CAD,
                    fixed_params={
                        'M_inf': fixed_params['M_inf'],
                        'beta': fixed_params['beta'],
                        'height': fixed_params['height'],
                        'width': fixed_params['width'],
                        'n_planes': fixed_params.get('n_planes', 40),
                        'n_streamwise': fixed_params.get('n_streamwise', 30),
                        'delta_streamwise': fixed_params.get('delta_streamwise', 0.1),
                        'mesh_size': algorithm_params.get('mesh_size', 0.1)
                    }
                )
                self.log_message.emit("✓ Pareto design files generated")
            
            # Print summary
            self.log_message.emit("")
            try:
                results.print_summary()
            except Exception as e:
                self.log_message.emit(f"⚠ Summary failed: {e}")
            
            # Emit completion signal
            self.optimization_complete.emit(results.results_folder)
            
        except Exception as e:
            import traceback
            error_msg = f"ERROR: {str(e)}\n{traceback.format_exc()}"
            self.error_occurred.emit(error_msg)
            self.log_message.emit("")
            self.log_message.emit("❌ OPTIMIZATION FAILED")
            self.log_message.emit(error_msg)
    
    def stop(self):
        """Request optimization to stop"""
        self.should_stop = True
        self.log_message.emit("⏸ Stopping optimization (will complete current generation)...")



class OptimizationTab(QWidget):
    """Tab for setting up and running waverider optimization"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_gui = parent
        
        # Initialize state
        self.optimization_running = False
        self.optimization_worker = None
        # Cases sent by the design tabs: title -> (OptimizationCase, tab or None)
        self.method_cases = {}
        self.last_results_folder = None
        self.last_config = None
        self.constraint_window = None
        
        self.init_ui()
        
        # Connect to parent GUI geometry changes if available
        self.connect_geometry_signals()
    
    def connect_geometry_signals(self):
        """Connect to parent GUI geometry spin boxes to update constraints"""
        if hasattr(self, 'parent_gui') and self.parent_gui:
            # Connect width spin
            if hasattr(self.parent_gui, 'width_spin'):
                self.parent_gui.width_spin.valueChanged.connect(self.on_geometry_changed)
            # Connect height spin
            if hasattr(self.parent_gui, 'height_spin'):
                self.parent_gui.height_spin.valueChanged.connect(self.on_geometry_changed)
    
    def on_geometry_changed(self):
        """Handle width/height change from parent GUI - update X2 constraint"""
        # Recalculate X2 max based on new geometry
        self.on_x1_changed()
        self.update_design_var_hints()
        
    def init_ui(self):
        """Initialize the user interface"""
        main_layout = QHBoxLayout(self)
        
        # Left panel: Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(self.create_method_group())
        self.design_vars_group = self.create_design_variables_group()
        left_layout.addWidget(self.design_vars_group)
        left_layout.addWidget(self.create_objectives_group())
        left_layout.addWidget(self.create_constraints_group())
        left_layout.addWidget(self.create_simulation_params_group())
        from aero_corrections_panel import AeroCorrectionsGroup
        self.opt_corrections_group = AeroCorrectionsGroup()
        left_layout.addWidget(self.opt_corrections_group)
        left_layout.addWidget(self.create_algorithm_settings_group())
        left_layout.addWidget(self.create_run_controls_group())
        left_layout.addStretch()
        
        # Right panel: Progress and results
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.addWidget(self.create_progress_group())
        right_layout.addWidget(self.create_console_group())
        
        # Add panels to splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)  # Left panel
        splitter.setStretchFactor(1, 2)  # Right panel gets more space
        
        main_layout.addWidget(splitter)
        
        # Initial updates (using QTimer to ensure all widgets are ready)
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(100, self.update_design_var_hints)
        QTimer.singleShot(100, self.update_time_estimate)
        
    # ------------------------------------------------------------------
    #  Design method: the OC Waverider (X1-X4) or a case sent by a tab
    # ------------------------------------------------------------------

    OC_METHOD_TEXT = "OC Waverider (X1–X4, settings from the OC Waverider tab)"

    def create_method_group(self):
        """Which design method to optimize: the OC waverider, or a design a
        tab sent with "Send to Optimization" (see method_optimization)."""
        group = QGroupBox("Design Method")
        layout = QVBoxLayout()

        row = QHBoxLayout()
        row.addWidget(QLabel("<b>Method:</b>"))
        self.method_combo = QComboBox()
        self.method_combo.addItem(self.OC_METHOD_TEXT)
        self.method_combo.setToolTip(
            "OC Waverider: the classic X1–X4 optimization (CAD export and Gmsh mesh).\n"
            "Other methods appear here when you click \"Send to Optimization\" in their tab.")
        self.method_combo.currentIndexChanged.connect(self.on_method_changed)
        row.addWidget(self.method_combo, 1)
        layout.addLayout(row)

        self.method_note = QLabel(
            "<i>To optimize another method, click <b>Send to Optimization</b> in its tab.</i>")
        self.method_note.setWordWrap(True)
        self.method_note.setStyleSheet("color: #888888;")
        layout.addWidget(self.method_note)

        self.method_settings = QWidget()
        ms = QVBoxLayout(self.method_settings)
        ms.setContentsMargins(0, 0, 0, 0)
        mach_row = QHBoxLayout()
        mach_row.addWidget(QLabel("<b>Flight Mach number:</b>"))
        self.flight_mach_spin = QDoubleSpinBox()
        self.flight_mach_spin.setRange(1.1, 40.0)
        self.flight_mach_spin.setDecimals(2)
        self.flight_mach_spin.setSingleStep(0.5)
        self.flight_mach_spin.setToolTip(
            "Mach number of the PySAGAS analysis. Starts at the design Mach number;\n"
            "the geometry keeps its own design Mach number.")
        mach_row.addWidget(self.flight_mach_spin)
        mach_row.addStretch()
        ms.addLayout(mach_row)

        self.var_table = QTableWidget(0, 5)
        self.var_table.setHorizontalHeaderLabels(["Vary", "Variable", "Current", "Min", "Max"])
        self.var_table.verticalHeader().setVisible(False)
        self.var_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        for col in (0, 2, 3, 4):
            self.var_table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeToContents)
        self.var_table.setToolTip("Tick the parameters the optimizer may vary, within Min–Max.\n"
                                  "Unticked parameters keep their current value.")
        # As tall as its rows: the left panel scrolls, the table does not
        self.var_table.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.var_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.var_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.var_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        ms.addWidget(self.var_table)

        info = QLabel(
            "<i>The tab's current design is the starting point. Each design is meshed "
            "(closed, full span) and analysed with PySAGAS; A_ref is its planform area. "
            "Results: the designs, the Pareto front and, per Pareto design, an STL and "
            "its parameters.</i>")
        info.setWordWrap(True)
        info.setStyleSheet("color: #888888; font-size: 9px;")
        ms.addWidget(info)

        self.apply_best_btn = QPushButton("Apply best design to its tab")
        self.apply_best_btn.setEnabled(False)
        self.apply_best_btn.setToolTip(
            "Load the best design of the last run (first objective, among the Pareto\n"
            "designs) into the tab it came from and generate it.")
        self.apply_best_btn.clicked.connect(self.apply_best_design)
        ms.addWidget(self.apply_best_btn)
        layout.addWidget(self.method_settings)
        self.method_settings.setVisible(False)

        group.setLayout(layout)
        return group

    def is_oc_method(self) -> bool:
        return self.method_combo.currentIndex() <= 0

    def set_method_case(self, case, tab=None):
        """Make ``case`` (method_optimization.OptimizationCase) the method to
        optimize; ``tab`` is where an optimized design is applied back."""
        self.method_cases[case.title] = (case, tab)
        index = self.method_combo.findText(case.title)
        if index < 0:
            self.method_combo.addItem(case.title)
            index = self.method_combo.count() - 1
        if self.method_combo.currentIndex() == index:
            self.on_method_changed(index)
        else:
            self.method_combo.setCurrentIndex(index)
        self.log_message(f"✓ {case.title}: design received ({len(case.variables)} parameters)")

    def on_method_changed(self, index):
        oc = self.is_oc_method()
        self.design_vars_group.setVisible(oc)
        self.method_settings.setVisible(not oc)
        if hasattr(self, 'opt_match_shock_check'):
            self.opt_match_shock_check.setVisible(oc)
        if hasattr(self, 'opt_aref_spin'):
            for w in (self.aref_label, self.opt_aref_spin, self.aref_unit, self.sync_aref_btn):
                w.setVisible(oc)
        if hasattr(self, 'mesh_quality_combo'):
            self.mesh_quality_combo.setEnabled(oc)
        if oc:
            self.method_note.setText(
                "<i>To optimize another method, click <b>Send to Optimization</b> in its tab.</i>")
        else:
            case, _ = self.method_cases[self.method_combo.currentText()]
            note = f"Starting point: the current inputs of the <b>{case.title}</b> tab."
            if case.notes:
                note += f" {case.notes}"
            self.method_note.setText(f"<i>{note}</i>")
            self.flight_mach_spin.setValue(case.mach)
            self._fill_variable_table(case)
        if hasattr(self, 'time_estimate_label'):
            self.update_time_estimate()

    def _fill_variable_table(self, case):
        t = self.var_table
        t.setRowCount(len(case.variables))
        self._var_widgets = []
        for i, v in enumerate(case.variables):
            use = QCheckBox()
            use.setChecked(v.enabled)
            t.setCellWidget(i, 0, use)
            t.setItem(i, 1, QTableWidgetItem(v.label))
            t.item(i, 1).setFlags(Qt.ItemIsEnabled)
            t.setItem(i, 2, QTableWidgetItem(f"{v.value:.6g}"))
            t.item(i, 2).setFlags(Qt.ItemIsEnabled)
            spins = []
            for col, val in ((3, v.lower), (4, v.upper)):
                spin = QDoubleSpinBox()
                spin.setRange(-1e6, 1e6)
                spin.setDecimals(4)
                spin.setSingleStep(max(abs(v.upper - v.lower) / 20.0, 1e-4))
                spin.setValue(val)
                t.setCellWidget(i, col, spin)
                spins.append(spin)
            self._var_widgets.append((use, spins[0], spins[1]))
        t.resizeRowsToContents()
        self._fit_var_table()
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(0, self._fit_var_table)          # again once styled and shown

    def _fit_var_table(self):
        # A minimum height is respected up through the splitter and the
        # scroll area, where a fixed size policy alone was squeezed
        self.var_table.setMinimumHeight(self.var_table.sizeHint().height())

    def current_case(self):
        """The selected case with the table's choices, or None for the OC method."""
        if self.is_oc_method():
            return None
        from method_optimization import DesignVariable, OptimizationCase

        case, _ = self.method_cases[self.method_combo.currentText()]
        variables = [DesignVariable(v.name, v.label, v.value, lo.value(), hi.value(), use.isChecked())
                     for v, (use, lo, hi) in zip(case.variables, self._var_widgets)]
        return OptimizationCase(case.method, case.title, case.fixed, variables,
                                self.flight_mach_spin.value(), case.notes)

    def apply_best_design(self):
        """Load the best design of the last run into the tab it came from."""
        import pandas as pd

        if not self.last_results_folder or not (self.last_config or {}).get('method_case'):
            return
        from method_optimization import OptimizationCase

        case = OptimizationCase.from_dict(self.last_config['method_case'])    # the run's case
        _, tab = self.method_cases.get(case.title, (None, None))
        obj = (self.last_config or {}).get('objectives', [])
        folder = Path(self.last_results_folder)
        df = None
        for csv in (folder / "pareto_front.csv", folder / "designs.csv"):
            if csv.exists():
                df = pd.read_csv(csv)
                df = df[df['Success'] == True] if 'Success' in df else df
                if len(df):
                    break
        if not obj or df is None or not len(df):
            QMessageBox.warning(self, "Apply best design", "No successful design in the last run.")
            return
        name, mode = obj[0]['name'], obj[0]['mode'].lower()
        row = df.loc[df[name].idxmax() if mode == 'maximize' else df[name].idxmin()]
        names = [v['name'] for v in self.last_config['design_variables']]
        params = case.params([row[n] for n in names])
        if self.parent_gui is not None and hasattr(self.parent_gui, 'apply_optimized_design'):
            self.parent_gui.apply_optimized_design(case.title, params, tab)
        elif tab is not None:
            tab.apply_optimized_params(params)
        self.log_message(f"✓ Design {int(row['Design_ID'])} ({name} = {row[name]:.5g}) "
                         f"applied to the {case.title} tab")

    def create_design_variables_group(self):
        """Create design variable range controls with automatic constraint enforcement"""
        group = QGroupBox("Design Variable Ranges")
        layout = QGridLayout()
        
        # Safety margin for constraint (90% of theoretical max, as in paper)
        self.constraint_safety_margin = 0.90
        
        # Headers
        layout.addWidget(QLabel("<b>Variable</b>"), 0, 0)
        layout.addWidget(QLabel("<b>Min</b>"), 0, 1)
        layout.addWidget(QLabel("<b>Max</b>"), 0, 2)
        layout.addWidget(QLabel("<b>Description</b>"), 0, 3)
        
        # Design variables
        self.design_var_spins = {}
        
        variables = [
            ('X1', 0.0, 0.5, 'Flat region length (normalized to width)'),
            ('X2', 0.0, 0.5, 'Shockwave height (normalized to height)'),
            ('X3', 0.0, 1.0, 'Central upper surface shape'),
            ('X4', 0.0, 1.0, 'Side upper surface shape')
        ]
        
        for i, (var_name, default_min, default_max, description) in enumerate(variables, start=1):
            # Variable name
            layout.addWidget(QLabel(f"<b>{var_name}</b>"), i, 0)
            
            # Min value
            min_spin = QDoubleSpinBox()
            min_spin.setRange(0.0, 0.999)
            min_spin.setValue(default_min)
            min_spin.setSingleStep(0.01)
            min_spin.setDecimals(3)
            layout.addWidget(min_spin, i, 1)
            
            # Max value
            max_spin = QDoubleSpinBox()
            max_spin.setRange(0.001, 1.0)
            max_spin.setValue(default_max)
            max_spin.setSingleStep(0.01)
            max_spin.setDecimals(3)
            layout.addWidget(max_spin, i, 2)
            
            # Description
            desc_label = QLabel(description)
            desc_label.setWordWrap(True)
            layout.addWidget(desc_label, i, 3)
            
            # Store references
            self.design_var_spins[var_name] = {'min': min_spin, 'max': max_spin}
        
        # Connect X1 changes to auto-adjust X2 max
        self.design_var_spins['X1']['min'].valueChanged.connect(self.on_x1_changed)
        self.design_var_spins['X1']['max'].valueChanged.connect(self.on_x1_changed)
        
        # Connect X2 changes to validate against constraint
        self.design_var_spins['X2']['max'].valueChanged.connect(self.on_x2_max_changed)
        
        # Connect all to update hints
        for var_name in ['X1', 'X2', 'X3', 'X4']:
            self.design_var_spins[var_name]['min'].valueChanged.connect(self.update_design_var_hints)
            self.design_var_spins[var_name]['max'].valueChanged.connect(self.update_design_var_hints)
        
        # Design space constraint info
        constraint_label = QLabel(
            "📐 <i>Constraint: X2 / (1-X1)⁴ < 7/64 × (width/height)⁴</i>"
        )
        constraint_label.setWordWrap(True)
        constraint_label.setStyleSheet("color: #888888;")
        layout.addWidget(constraint_label, len(variables)+1, 0, 1, 4)
        
        # Auto-adjust info
        auto_adjust_label = QLabel(
            "🔄 <i>X2 max is automatically adjusted when X1 changes (90% safety margin per paper)</i>"
        )
        auto_adjust_label.setWordWrap(True)
        auto_adjust_label.setStyleSheet("color: #F59E0B; font-size: 9px;")
        layout.addWidget(auto_adjust_label, len(variables)+2, 0, 1, 4)
        
        # Dynamic constraint hint label
        self.design_var_hint_label = QLabel("")
        self.design_var_hint_label.setWordWrap(True)
        self.design_var_hint_label.setStyleSheet("color: #888888; font-size: 10px; padding: 2px;")
        layout.addWidget(self.design_var_hint_label, len(variables)+3, 0, 1, 4)
        
        group.setLayout(layout)
        return group
    
    def get_geometry_params(self):
        """Get width and height from parent GUI or use defaults"""
        if hasattr(self, 'parent_gui') and self.parent_gui:
            if hasattr(self.parent_gui, 'width_spin') and hasattr(self.parent_gui, 'height_spin'):
                return self.parent_gui.width_spin.value(), self.parent_gui.height_spin.value()
        return 3.0, 1.34  # Default values
    
    def calculate_max_x2(self, x1_max):
        """
        Calculate maximum allowed X2 given X1 max, based on paper constraint.
        
        Constraint: X2 / (1 - X1)^4 < (7/64) * (width/height)^4
        Rearranged: X2 < (7/64) * (width/height)^4 * (1 - X1)^4
        
        Parameters
        ----------
        x1_max : float
            Maximum X1 value
            
        Returns
        -------
        float
            Maximum allowed X2 value with safety margin
        """
        width, height = self.get_geometry_params()
        
        # Avoid division by zero
        one_minus_x1 = max(1.0 - x1_max, 0.001)
        
        # Theoretical maximum X2
        rhs = (7.0 / 64.0) * (width / height) ** 4
        max_x2_theoretical = rhs * (one_minus_x1 ** 4)
        
        # Apply safety margin
        max_x2_safe = max_x2_theoretical * self.constraint_safety_margin
        
        # Clamp to [0, 1]
        return min(max(max_x2_safe, 0.001), 1.0)
    
    def on_x1_changed(self):
        """Handle X1 min or max change - auto-adjust X2 max"""
        try:
            x1_max = self.design_var_spins['X1']['max'].value()
            x2_max_current = self.design_var_spins['X2']['max'].value()
            
            # Calculate new max X2 allowed
            max_x2_allowed = self.calculate_max_x2(x1_max)
            
            # If current X2 max exceeds allowed, adjust it
            if x2_max_current > max_x2_allowed:
                # Block signals to prevent recursion
                self.design_var_spins['X2']['max'].blockSignals(True)
                self.design_var_spins['X2']['max'].setValue(max_x2_allowed)
                self.design_var_spins['X2']['max'].blockSignals(False)
                
                # Also ensure X2 min doesn't exceed new max
                x2_min = self.design_var_spins['X2']['min'].value()
                if x2_min > max_x2_allowed:
                    self.design_var_spins['X2']['min'].blockSignals(True)
                    self.design_var_spins['X2']['min'].setValue(max_x2_allowed * 0.5)
                    self.design_var_spins['X2']['min'].blockSignals(False)
            
            # Update the hint
            self.update_design_var_hints()
            
        except Exception as e:
            print(f"Error in on_x1_changed: {e}")
    
    def on_x2_max_changed(self):
        """Handle X2 max change - enforce constraint"""
        try:
            x1_max = self.design_var_spins['X1']['max'].value()
            x2_max = self.design_var_spins['X2']['max'].value()
            
            # Calculate allowed max
            max_x2_allowed = self.calculate_max_x2(x1_max)
            
            # If user tries to set X2 max too high, clamp it
            if x2_max > max_x2_allowed:
                self.design_var_spins['X2']['max'].blockSignals(True)
                self.design_var_spins['X2']['max'].setValue(max_x2_allowed)
                self.design_var_spins['X2']['max'].blockSignals(False)
            
        except Exception as e:
            print(f"Error in on_x2_max_changed: {e}")
    
    def update_design_var_hints(self):
        """Update design variable constraint hints based on current values"""
        try:
            width, height = self.get_geometry_params()
            
            # Get current values
            x1_max = self.design_var_spins['X1']['max'].value()
            x2_max = self.design_var_spins['X2']['max'].value()
            
            # Calculate max X2 allowed
            max_x2_allowed = self.calculate_max_x2(x1_max)
            
            # Calculate how much "headroom" we have
            headroom = max_x2_allowed - x2_max
            headroom_pct = (headroom / max_x2_allowed * 100) if max_x2_allowed > 0 else 0
            
            # Update hint based on status
            if x2_max > max_x2_allowed * 1.01:  # Allow tiny floating point tolerance
                self.design_var_hint_label.setStyleSheet(
                    "color: #EF4444; font-size: 10px; padding: 2px; font-weight: bold;"
                )
                hint_text = (
                    f"⚠️ X2 max ({x2_max:.3f}) exceeds limit! "
                    f"Max allowed: {max_x2_allowed:.3f} for w={width:.2f}m, h={height:.2f}m"
                )
            elif headroom_pct < 5:
                self.design_var_hint_label.setStyleSheet(
                    "color: #F59E0B; font-size: 10px; padding: 2px;"
                )
                hint_text = (
                    f"⚡ Near boundary | w={width:.2f}m, h={height:.2f}m | "
                    f"X1≤{x1_max:.3f}, X2≤{x2_max:.3f} (max: {max_x2_allowed:.3f})"
                )
            else:
                self.design_var_hint_label.setStyleSheet(
                    "color: #4ADE80; font-size: 10px; padding: 2px;"
                )
                hint_text = (
                    f"✓ Valid | w={width:.2f}m, h={height:.2f}m | "
                    f"X1≤{x1_max:.3f}, X2≤{x2_max:.3f} (max: {max_x2_allowed:.3f}, {headroom_pct:.0f}% margin)"
                )
            
            self.design_var_hint_label.setText(hint_text)
            
        except Exception as e:
            self.design_var_hint_label.setText(f"Error: {e}")
        
    def create_objectives_group(self):
        """Create objective function selection with single/multi-objective mode"""
        group = QGroupBox("Optimization Objectives")
        main_layout = QVBoxLayout()
        
        # ===== Mode Selection =====
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("<b>Optimization Mode:</b>"))
        
        self.opt_mode_combo = QComboBox()
        self.opt_mode_combo.addItems([
            'Single-Objective',
            'Multi-Objective (Pareto Front)'
        ])
        self.opt_mode_combo.currentIndexChanged.connect(self.on_optimization_mode_changed)
        mode_layout.addWidget(self.opt_mode_combo)
        mode_layout.addStretch()
        main_layout.addLayout(mode_layout)
        
        # Mode description
        self.mode_description = QLabel(
            "<i>Single-objective: Optimize one objective (e.g., maximize CL/CD)</i>"
        )
        self.mode_description.setWordWrap(True)
        self.mode_description.setStyleSheet("color: #888888; margin: 5px 0;")
        main_layout.addWidget(self.mode_description)
        
        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(line)
        
        # ===== Objective Selection Grid =====
        obj_layout = QGridLayout()
        
        # Headers
        obj_layout.addWidget(QLabel("<b>Enable</b>"), 0, 0)
        obj_layout.addWidget(QLabel("<b>Objective</b>"), 0, 1)
        obj_layout.addWidget(QLabel("<b>Mode</b>"), 0, 2)
        obj_layout.addWidget(QLabel("<b>Description</b>"), 0, 3)
        
        # Objective options
        self.objective_controls = {}
        
        objectives = [
            ('CL', 'Lift Coefficient', 'Aerodynamic lift force'),
            ('CD', 'Drag Coefficient', 'Aerodynamic drag force'),
            ('Cm', 'Pitching Moment Coefficient', 'Rotational stability'),
            ('Volume', 'Internal Volume [m³]', 'Available internal space'),
            ('VolEff', 'Volumetric Efficiency', 'V^(2/3) / Planform Area'),
            ('CL/CD', 'Lift-to-Drag Ratio', 'Aerodynamic efficiency')
        ]
        
        for i, (obj_name, display_name, description) in enumerate(objectives, start=1):
            # Enable checkbox
            enable_check = QCheckBox()
            # Default: only CL/CD for single-objective mode
            enable_check.setChecked(obj_name == 'CL/CD')
            enable_check.stateChanged.connect(self.on_objective_selection_changed)
            obj_layout.addWidget(enable_check, i, 0)
            
            # Objective name
            obj_layout.addWidget(QLabel(f"<b>{display_name}</b>"), i, 1)
            
            # Mode combo box
            mode_combo = QComboBox()
            mode_combo.addItems(['Minimize', 'Maximize'])
            if obj_name in ['CL', 'Volume', 'VolEff', 'CL/CD']:
                mode_combo.setCurrentText('Maximize')
            else:
                mode_combo.setCurrentText('Minimize')
            obj_layout.addWidget(mode_combo, i, 2)
            
            # Description
            desc_label = QLabel(description)
            desc_label.setWordWrap(True)
            obj_layout.addWidget(desc_label, i, 3)
            
            # Store references
            self.objective_controls[obj_name] = {
                'enable': enable_check,
                'mode': mode_combo
            }
        
        main_layout.addLayout(obj_layout)
        
        # ===== Multi-objective preset button =====
        preset_layout = QHBoxLayout()
        self.pareto_preset_btn = QPushButton("📊 Set Classic Waverider Trade-off (CL/CD vs Volume)")
        self.pareto_preset_btn.clicked.connect(self.apply_pareto_preset)
        self.pareto_preset_btn.setVisible(False)  # Hidden in single-objective mode
        self.pareto_preset_btn.setStyleSheet("background-color: #78350F; padding: 5px;")
        preset_layout.addWidget(self.pareto_preset_btn)
        preset_layout.addStretch()
        main_layout.addLayout(preset_layout)
        
        # Selection status label
        self.obj_status_label = QLabel("")
        self.obj_status_label.setStyleSheet("color: #888888; font-style: italic;")
        main_layout.addWidget(self.obj_status_label)
        self.update_objective_status()
            
        group.setLayout(main_layout)
        return group
    
    def on_optimization_mode_changed(self, index):
        """Handle optimization mode change"""
        is_multi = index == 1  # Multi-objective
        
        # Update description
        if is_multi:
            self.mode_description.setText(
                "<i>Multi-objective: Generate Pareto front showing trade-offs between 2 objectives. "
                "Select exactly 2 objectives.</i>"
            )
            self.pareto_preset_btn.setVisible(True)
        else:
            self.mode_description.setText(
                "<i>Single-objective: Optimize one objective (e.g., maximize CL/CD)</i>"
            )
            self.pareto_preset_btn.setVisible(False)
        
        self.update_objective_status()
    
    def on_objective_selection_changed(self):
        """Handle objective checkbox changes"""
        self.update_objective_status()
    
    def update_objective_status(self):
        """Update the objective selection status label"""
        enabled = [name for name, ctrl in self.objective_controls.items() 
                   if ctrl['enable'].isChecked()]
        n_enabled = len(enabled)
        is_multi = self.opt_mode_combo.currentIndex() == 1
        
        if is_multi:
            if n_enabled == 0:
                self.obj_status_label.setText("⚠️ Select exactly 2 objectives for Pareto front")
                self.obj_status_label.setStyleSheet("color: #F59E0B; font-style: italic;")
            elif n_enabled == 1:
                self.obj_status_label.setText(f"⚠️ Select one more objective (currently: {enabled[0]})")
                self.obj_status_label.setStyleSheet("color: #F59E0B; font-style: italic;")
            elif n_enabled == 2:
                self.obj_status_label.setText(f"✓ Pareto front: {enabled[0]} vs {enabled[1]}")
                self.obj_status_label.setStyleSheet("color: #4ADE80; font-style: italic;")
            else:
                self.obj_status_label.setText(f"⚠️ Too many objectives ({n_enabled}). Select exactly 2.")
                self.obj_status_label.setStyleSheet("color: #EF4444; font-style: italic;")
        else:
            if n_enabled == 0:
                self.obj_status_label.setText("⚠️ Select one objective")
                self.obj_status_label.setStyleSheet("color: #F59E0B; font-style: italic;")
            elif n_enabled == 1:
                self.obj_status_label.setText(f"✓ Optimizing: {enabled[0]}")
                self.obj_status_label.setStyleSheet("color: #4ADE80; font-style: italic;")
            else:
                self.obj_status_label.setText(f"⚠️ Single-objective mode: select only 1 objective (or switch to Multi-Objective)")
                self.obj_status_label.setStyleSheet("color: #F59E0B; font-style: italic;")
    
    def apply_pareto_preset(self):
        """Apply the classic waverider Pareto trade-off preset"""
        # Uncheck all
        for ctrl in self.objective_controls.values():
            ctrl['enable'].setChecked(False)
        
        # Check CL/CD and Volume
        self.objective_controls['CL/CD']['enable'].setChecked(True)
        self.objective_controls['CL/CD']['mode'].setCurrentText('Maximize')
        self.objective_controls['Volume']['enable'].setChecked(True)
        self.objective_controls['Volume']['mode'].setCurrentText('Maximize')
        
        self.update_objective_status()
        
    def create_constraints_group(self):
        """Create constraint settings"""
        group = QGroupBox("Constraints")
        layout = QGridLayout()
        
        # Headers
        layout.addWidget(QLabel("<b>Enable</b>"), 0, 0)
        layout.addWidget(QLabel("<b>Constraint</b>"), 0, 1)
        layout.addWidget(QLabel("<b>Type</b>"), 0, 2)
        layout.addWidget(QLabel("<b>Value</b>"), 0, 3)
        
        # Constraint options
        self.constraint_controls = {}
        
        constraints = [
            ('CL_min', 'Minimum Lift Coefficient', '≥', 1.0, 0.0, 5.0),
            ('CD_max', 'Maximum Drag Coefficient', '≤', 0.5, 0.0, 2.0),
            ('Cm_max', 'Maximum Pitching Moment (abs)', '≤', 0.1, 0.0, 1.0),
            ('Volume_min', 'Minimum Volume [m³]', '≥', 1.0, 0.0, 10.0)
        ]
        
        for i, (const_name, display_name, symbol, default_val, min_val, max_val) in enumerate(constraints, start=1):
            # Enable checkbox
            enable_check = QCheckBox()
            # None by default: a waverider's CL is ~0.05-0.2, so a CL >= 1
            # default made every design infeasible
            enable_check.setChecked(False)
            layout.addWidget(enable_check, i, 0)
            
            # Constraint name
            layout.addWidget(QLabel(f"<b>{display_name}</b>"), i, 1)
            
            # Type symbol
            layout.addWidget(QLabel(symbol), i, 2)
            
            # Value spinbox
            value_spin = QDoubleSpinBox()
            value_spin.setRange(min_val, max_val)
            value_spin.setValue(default_val)
            value_spin.setSingleStep(0.1)
            value_spin.setDecimals(3)
            layout.addWidget(value_spin, i, 3)
            
            # Store references
            self.constraint_controls[const_name] = {
                'enable': enable_check,
                'value': value_spin
            }
            
        group.setLayout(layout)
        return group
    
    def create_simulation_params_group(self):
        """Create PySAGAS simulation parameters controls"""
        group = QGroupBox("Simulation Parameters (PySAGAS)")
        layout = QGridLayout()
        
        row = 0
        
        # Angle of attack
        layout.addWidget(QLabel("<b>Angle of Attack α:</b>"), row, 0)
        self.opt_aoa_spin = QDoubleSpinBox()
        self.opt_aoa_spin.setRange(-20.0, 20.0)
        self.opt_aoa_spin.setValue(0.0)
        self.opt_aoa_spin.setSingleStep(0.5)
        self.opt_aoa_spin.setDecimals(2)
        self.opt_aoa_spin.setToolTip("Angle of attack for aerodynamic analysis during optimization")
        layout.addWidget(self.opt_aoa_spin, row, 1)
        layout.addWidget(QLabel("deg"), row, 2)
        row += 1
        
        # Reference area (the OC method; other methods use each design's planform area)
        self.aref_label = QLabel("<b>Reference Area A_ref:</b>")
        layout.addWidget(self.aref_label, row, 0)
        self.opt_aref_spin = QDoubleSpinBox()
        self.opt_aref_spin.setRange(0.1, 1000.0)
        self.opt_aref_spin.setValue(19.65)  # Default for baseline waverider
        self.opt_aref_spin.setSingleStep(0.1)
        self.opt_aref_spin.setDecimals(4)
        self.opt_aref_spin.setToolTip(
            "Reference area for coefficient normalization.\n"
            "Tip: Use planform area from Aero Analysis tab."
        )
        layout.addWidget(self.opt_aref_spin, row, 1)
        self.aref_unit = QLabel("m²")
        layout.addWidget(self.aref_unit, row, 2)
        
        # Sync button
        self.sync_aref_btn = QPushButton("📋 Sync from Aero Tab")
        self.sync_aref_btn.clicked.connect(self.sync_aref_from_aero)
        self.sync_aref_btn.setToolTip("Copy reference area from Aero Analysis tab")
        layout.addWidget(self.sync_aref_btn, row, 3)
        row += 1
        
        # Freestream pressure
        layout.addWidget(QLabel("<b>Pressure P∞:</b>"), row, 0)
        self.opt_pressure_spin = QDoubleSpinBox()
        self.opt_pressure_spin.setRange(100, 1e7)
        self.opt_pressure_spin.setValue(2549.0)  # ~25km altitude
        self.opt_pressure_spin.setSingleStep(100)
        self.opt_pressure_spin.setDecimals(1)
        self.opt_pressure_spin.setToolTip("Freestream static pressure at flight altitude")
        layout.addWidget(self.opt_pressure_spin, row, 1)
        layout.addWidget(QLabel("Pa"), row, 2)
        row += 1
        
        # Freestream temperature
        layout.addWidget(QLabel("<b>Temperature T∞:</b>"), row, 0)
        self.opt_temperature_spin = QDoubleSpinBox()
        self.opt_temperature_spin.setRange(100, 500)
        self.opt_temperature_spin.setValue(221.55)  # ~25km altitude ISA
        self.opt_temperature_spin.setSingleStep(1)
        self.opt_temperature_spin.setDecimals(2)
        self.opt_temperature_spin.setToolTip("Freestream static temperature at flight altitude")
        layout.addWidget(self.opt_temperature_spin, row, 1)
        layout.addWidget(QLabel("K"), row, 2)
        row += 1
        
        # Altitude preset helper
        altitude_layout = QHBoxLayout()
        altitude_layout.addWidget(QLabel("Altitude preset:"))
        self.altitude_preset_combo = QComboBox()
        self.altitude_preset_combo.addItems([
            'Custom',
            'Sea Level (0 km)',
            '10 km (Troposphere)',
            '20 km (Stratosphere)',
            '25 km (High Altitude)',
            '30 km (Near Space)'
        ])
        self.altitude_preset_combo.setCurrentText('25 km (High Altitude)')
        self.altitude_preset_combo.currentTextChanged.connect(self.apply_altitude_preset)
        altitude_layout.addWidget(self.altitude_preset_combo)
        layout.addLayout(altitude_layout, row, 0, 1, 4)
        row += 1
        
        # Info label
        info_label = QLabel(
            "<i>💡 Altitude presets use ISA (International Standard Atmosphere) values. "
            "For optimization, ensure these match your design flight conditions.</i>"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #888888; font-size: 9px;")
        layout.addWidget(info_label, row, 0, 1, 4)
        row += 1
        
        # Geometry options separator
        layout.addWidget(QLabel(""), row, 0)  # spacer
        row += 1
        
        # Match lower surface to shockwave option
        self.opt_match_shock_check = QCheckBox("Match lower surface to shockwave (Max Volume)")
        self.opt_match_shock_check.setToolTip(
            "When enabled, the lower surface follows the shockwave curve\n"
            "instead of tracing streamlines through the conical flowfield.\n\n"
            "This maximizes internal volume for each design.\n"
            "Aerodynamic coefficients are still computed correctly."
        )
        self.opt_match_shock_check.setChecked(False)
        # Try to sync with main GUI checkbox
        if hasattr(self, 'parent_gui') and self.parent_gui:
            if hasattr(self.parent_gui, 'match_shock_check'):
                self.opt_match_shock_check.setChecked(self.parent_gui.match_shock_check.isChecked())
        layout.addWidget(self.opt_match_shock_check, row, 0, 1, 4)
        
        group.setLayout(layout)
        return group
    
    def sync_aref_from_aero(self):
        """Sync reference area from Aero Analysis tab"""
        try:
            if hasattr(self, 'parent_gui') and self.parent_gui:
                if hasattr(self.parent_gui, 'aref_spin'):
                    aref = self.parent_gui.aref_spin.value()
                    self.opt_aref_spin.setValue(aref)
                    self.log_message(f"✓ Synced A_ref = {aref:.4f} m² from Aero Analysis tab")
                else:
                    self.log_message("⚠ Aero Analysis tab not found")
            else:
                self.log_message("⚠ Parent GUI not available")
        except Exception as e:
            self.log_message(f"⚠ Could not sync A_ref: {e}")
    
    def apply_altitude_preset(self, preset_name):
        """Apply ISA atmospheric conditions for preset altitude"""
        # ISA values at various altitudes
        presets = {
            'Sea Level (0 km)': (101325.0, 288.15),
            '10 km (Troposphere)': (26500.0, 223.25),
            '20 km (Stratosphere)': (5529.0, 216.65),
            '25 km (High Altitude)': (2549.0, 221.55),
            '30 km (Near Space)': (1197.0, 226.51)
        }
        
        if preset_name in presets:
            pressure, temperature = presets[preset_name]
            self.opt_pressure_spin.setValue(pressure)
            self.opt_temperature_spin.setValue(temperature)
        
    def create_algorithm_settings_group(self):
        """Create algorithm parameter controls"""
        group = QGroupBox("Algorithm Settings")
        layout = QGridLayout()
        
        row = 0
        
        # Population size
        layout.addWidget(QLabel("<b>Population Size:</b>"), row, 0)
        self.pop_size_spin = QSpinBox()
        self.pop_size_spin.setRange(10, 200)
        self.pop_size_spin.setValue(20)
        self.pop_size_spin.setSingleStep(10)
        self.pop_size_spin.setToolTip("Number of designs per generation")
        layout.addWidget(self.pop_size_spin, row, 1)
        layout.addWidget(QLabel("designs/generation"), row, 2)
        row += 1
        
        # Number of generations
        layout.addWidget(QLabel("<b>Generations:</b>"), row, 0)
        self.n_gen_spin = QSpinBox()
        self.n_gen_spin.setRange(5, 100)
        self.n_gen_spin.setValue(10)
        self.n_gen_spin.setSingleStep(5)
        self.n_gen_spin.setToolTip("Number of evolutionary generations")
        layout.addWidget(self.n_gen_spin, row, 1)
        layout.addWidget(QLabel("generations"), row, 2)
        row += 1
        
        # Crossover probability
        layout.addWidget(QLabel("<b>Crossover Prob.:</b>"), row, 0)
        self.crossover_prob_spin = QDoubleSpinBox()
        self.crossover_prob_spin.setRange(0.0, 1.0)
        self.crossover_prob_spin.setValue(0.9)
        self.crossover_prob_spin.setSingleStep(0.05)
        self.crossover_prob_spin.setDecimals(2)
        self.crossover_prob_spin.setToolTip(
            "Probability of crossover between parents.\n"
            "Higher = more genetic mixing."
        )
        layout.addWidget(self.crossover_prob_spin, row, 1)
        layout.addWidget(QLabel("<i>(recommended: 0.8-0.95)</i>"), row, 2)
        row += 1
        
        # Mutation probability
        layout.addWidget(QLabel("<b>Mutation Prob.:</b>"), row, 0)
        self.mutation_prob_spin = QDoubleSpinBox()
        self.mutation_prob_spin.setRange(0.0, 1.0)
        self.mutation_prob_spin.setValue(0.1)
        self.mutation_prob_spin.setSingleStep(0.01)
        self.mutation_prob_spin.setDecimals(2)
        self.mutation_prob_spin.setToolTip(
            "Probability of mutation per gene.\n"
            "Higher = more exploration, may slow convergence.\n"
            "Lower = faster convergence, may get stuck."
        )
        layout.addWidget(self.mutation_prob_spin, row, 1)
        layout.addWidget(QLabel("<i>(recommended: 0.05-0.20)</i>"), row, 2)
        row += 1
        
        # CPU cores
        layout.addWidget(QLabel("<b>CPU Cores:</b>"), row, 0)
        self.n_cores_spin = QSpinBox()
        self.n_cores_spin.setRange(1, os.cpu_count() or 4)
        self.n_cores_spin.setValue(1)  # Default to 1 for safety
        self.n_cores_spin.setToolTip("Parallel evaluations (Windows: use 'spawn' mode)")
        layout.addWidget(self.n_cores_spin, row, 1)
        
        # Platform info
        if sys.platform == 'win32':
            info_label = QLabel(f"✅ Windows ({os.cpu_count()} cores)")
            info_label.setStyleSheet("color: #4ADE80; font-size: 9px;")
            info_label.setWordWrap(True)
            layout.addWidget(info_label, row, 2)
        else:
            info_label = QLabel(f"✅ Linux/Mac ({os.cpu_count()} cores)")
            info_label.setStyleSheet("color: #4ADE80; font-size: 9px;")
            layout.addWidget(info_label, row, 2)
        row += 1
        
        # Mesh size preset
        layout.addWidget(QLabel("<b>Mesh Quality:</b>"), row, 0)
        self.mesh_quality_combo = QComboBox()
        self.mesh_quality_combo.addItems(['Coarse (fast)', 'Medium', 'Fine (slow)'])
        self.mesh_quality_combo.setCurrentText('Coarse (fast)')
        self.mesh_quality_combo.setToolTip("Mesh density affects accuracy and speed")
        layout.addWidget(self.mesh_quality_combo, row, 1, 1, 2)
        row += 1
        
        # Estimated time
        self.time_estimate_label = QLabel("<i>Estimated time: calculating...</i>")
        self.time_estimate_label.setWordWrap(True)
        layout.addWidget(self.time_estimate_label, row, 0, 1, 3)
        
        # Connect signals to update estimate
        self.pop_size_spin.valueChanged.connect(self.update_time_estimate)
        self.n_gen_spin.valueChanged.connect(self.update_time_estimate)
        self.n_cores_spin.valueChanged.connect(self.update_time_estimate)
        self.mesh_quality_combo.currentTextChanged.connect(self.update_time_estimate)
        
        group.setLayout(layout)
        return group
        
    def create_run_controls_group(self):
        """Create run/stop/export controls"""
        group = QGroupBox("Run Control")
        layout = QVBoxLayout()
        
        # Button layout
        button_layout = QHBoxLayout()
        
        # Start button
        self.start_btn = QPushButton("▶ Start Optimization")
        self.start_btn.setStyleSheet("QPushButton { background-color: #F59E0B; color: #0A0A0A; font-weight: bold; padding: 10px; }")
        self.start_btn.clicked.connect(self.start_optimization)
        button_layout.addWidget(self.start_btn)
        
        # Stop button
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setStyleSheet("QPushButton { background-color: #EF4444; color: #FFFFFF; font-weight: bold; padding: 10px; }")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_optimization)
        button_layout.addWidget(self.stop_btn)
        
        layout.addLayout(button_layout)
        
        # Config buttons
        config_layout = QHBoxLayout()
        
        save_config_btn = QPushButton("💾 Save Config")
        save_config_btn.clicked.connect(self.save_configuration)
        config_layout.addWidget(save_config_btn)
        
        load_config_btn = QPushButton("📂 Load Config")
        load_config_btn.clicked.connect(self.load_configuration)
        config_layout.addWidget(load_config_btn)
        
        layout.addLayout(config_layout)
        
        group.setLayout(layout)
        return group
        
    def create_progress_group(self):
        """Create progress display"""
        group = QGroupBox("Progress")
        layout = QVBoxLayout()
        
        # Overall progress
        layout.addWidget(QLabel("<b>Overall Progress:</b>"))
        self.overall_progress = QProgressBar()
        self.overall_progress.setRange(0, 100)
        self.overall_progress.setValue(0)
        layout.addWidget(self.overall_progress)
        
        # Current generation progress
        layout.addWidget(QLabel("<b>Current Generation:</b>"))
        self.generation_progress = QProgressBar()
        self.generation_progress.setRange(0, 100)
        self.generation_progress.setValue(0)
        layout.addWidget(self.generation_progress)
        
        # Status label
        self.status_label = QLabel("Ready to start optimization")
        self.status_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(10)
        self.status_label.setFont(font)
        layout.addWidget(self.status_label)
        
        # Statistics display
        stats_layout = QGridLayout()
        stats_layout.addWidget(QLabel("<b>Designs Evaluated:</b>"), 0, 0)
        self.designs_evaluated_label = QLabel("0")
        stats_layout.addWidget(self.designs_evaluated_label, 0, 1)
        
        stats_layout.addWidget(QLabel("<b>Pareto Designs:</b>"), 1, 0)
        self.pareto_count_label = QLabel("0")
        stats_layout.addWidget(self.pareto_count_label, 1, 1)
        
        stats_layout.addWidget(QLabel("<b>Best CL/CD:</b>"), 2, 0)
        self.best_ld_label = QLabel("N/A")
        stats_layout.addWidget(self.best_ld_label, 2, 1)
        
        stats_layout.addWidget(QLabel("<b>Best Volume:</b>"), 3, 0)
        self.best_volume_label = QLabel("N/A")
        stats_layout.addWidget(self.best_volume_label, 3, 1)
        
        layout.addLayout(stats_layout)
        
        group.setLayout(layout)
        return group
        
    def create_console_group(self):
        """Create console output"""
        group = QGroupBox("Console Output")
        layout = QVBoxLayout()
        
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumHeight(200)
        self.console.setStyleSheet("background-color: #0A0A0A; color: #4ADE80; font-family: monospace;")
        layout.addWidget(self.console)
        
        # Clear button
        clear_btn = QPushButton("Clear Console")
        clear_btn.clicked.connect(self.console.clear)
        layout.addWidget(clear_btn)
        
        group.setLayout(layout)
        return group
        
    def update_time_estimate(self):
        """Update estimated optimization time"""
        pop_size = self.pop_size_spin.value()
        n_gen = self.n_gen_spin.value()
        n_cores = self.n_cores_spin.value()
        mesh_quality = self.mesh_quality_combo.currentText()
        
        # Estimate time per design (seconds)
        time_per_design = 16  # Base estimate for coarse mesh
        if 'Medium' in mesh_quality:
            time_per_design *= 2
        elif 'Fine' in mesh_quality:
            time_per_design *= 4
        if hasattr(self, 'method_combo') and not self.is_oc_method():
            # direct mesh + PySAGAS; LTOCs also solves every stream surface
            case, _ = self.method_cases[self.method_combo.currentText()]
            time_per_design = METHOD_SECONDS.get(case.method, 3)
            
        total_designs = pop_size * n_gen
        total_time_sec = (total_designs * time_per_design) / n_cores
        
        # Format time
        hours = int(total_time_sec // 3600)
        minutes = int((total_time_sec % 3600) // 60)
        
        if hours > 0:
            time_str = f"{hours}h {minutes}min"
        else:
            time_str = f"{minutes} minutes"
            
        self.time_estimate_label.setText(
            f"<i>Estimated time: ~{time_str} "
            f"({total_designs} designs × {time_per_design}s ÷ {n_cores} cores)</i>"
        )
        
    def log_message(self, message):
        """Add message to console"""
        self.console.append(message)
        
    def start_optimization(self):
        """Start the optimization process"""
        # Validate configuration
        if not self.validate_configuration():
            return
        
        # Build configuration
        config = self.build_configuration(*self._oc_flight_and_size())
        self.last_config = config  # Store for use in callbacks
        self.apply_best_btn.setEnabled(False)
        
        self.log_message("=" * 60)
        self.log_message("STARTING OPTIMIZATION")
        self.log_message("=" * 60)
        
        # Create and setup worker
        self.optimization_worker = OptimizationWorker(config)
        self.optimization_worker.progress_update.connect(self.on_progress_update)
        self.optimization_worker.design_evaluated.connect(self.on_design_evaluated)
        self.optimization_worker.generation_complete.connect(self.on_generation_complete)
        self.optimization_worker.optimization_complete.connect(self.on_optimization_complete)
        self.optimization_worker.error_occurred.connect(self.on_error_occurred)
        self.optimization_worker.log_message.connect(self.log_message)
        
        # Update UI state
        self.optimization_running = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Optimization running...")
        self.overall_progress.setValue(0)
        self.generation_progress.setValue(0)
        
        # Open plot windows
        self.open_plot_windows(config)
        
        # Start worker
        self.optimization_worker.start()
        
    def _oc_flight_and_size(self):
        """M_inf, beta, height, width of the OC Waverider tab (or defaults)."""
        if hasattr(self.parent_gui, 'm_inf_spin'):
            return (self.parent_gui.m_inf_spin.value(), self.parent_gui.beta_spin.value(),
                    self.parent_gui.height_spin.value(), self.parent_gui.width_spin.value())
        return 5.0, 15.0, 1.34, 3.0

    def build_configuration(self, M_inf, beta, height, width):
        """Build configuration dictionary from GUI settings"""
        case = self.current_case()
        # Design variables
        design_variables = []
        if case is not None:
            design_variables = [{'name': v.name, 'label': v.label, 'min': v.lower,
                                 'max': v.upper} for v in case.active()]
        else:
            for var_name in ['X1', 'X2', 'X3', 'X4']:
                spins = self.design_var_spins[var_name]
                design_variables.append({
                    'name': var_name,
                    'min': spins['min'].value(),
                    'max': spins['max'].value()
                })
        
        # Objectives
        objectives = []
        for obj_name, controls in self.objective_controls.items():
            if controls['enable'].isChecked():
                mode = controls['mode'].currentText().lower()  # Convert to lowercase
                objectives.append({
                    'name': obj_name,
                    'mode': mode
                })
        
        # Constraints
        constraints = []
        for const_name, controls in self.constraint_controls.items():
            if controls['enable'].isChecked():
                value = controls['value'].value()
                constraints.append({
                    'name': const_name,
                    'value': value,
                    'active': True  # Explicitly set active flag
                })
        
        # Algorithm parameters
        mesh_quality = self.mesh_quality_combo.currentText()
        if 'Coarse' in mesh_quality:
            mesh_size = 0.15
        elif 'Medium' in mesh_quality:
            mesh_size = 0.10
        else:  # Fine
            mesh_size = 0.05
            
        algorithm = {
            'pop_size': self.pop_size_spin.value(),
            'n_gen': self.n_gen_spin.value(),
            'n_cores': self.n_cores_spin.value(),
            'mesh_size': mesh_size,
            'crossover_prob': self.crossover_prob_spin.value(),
            'mutation_prob': self.mutation_prob_spin.value()
        }
        
        # Simulation parameters (PySAGAS)
        simulation_params = {
            'aoa': self.opt_aoa_spin.value(),
            'A_ref': self.opt_aref_spin.value(),
            'pressure': self.opt_pressure_spin.value(),
            'temperature': self.opt_temperature_spin.value(),
            'match_shockwave': self.opt_match_shock_check.isChecked(),
            'corrections': self.opt_corrections_group.corrections().to_dict()
        }
        
        # Fixed parameters (geometry + flow)
        if case is not None:
            fixed_parameters = {'M_inf': case.mach}
        else:
            fixed_parameters = {
                'M_inf': M_inf,
                'beta': beta,
                'height': height,
                'width': width
            }
        
        config = {
            'design_variables': design_variables,
            'objectives': objectives,
            'constraints': constraints,
            'algorithm': algorithm,
            'simulation_params': simulation_params,
            'fixed_parameters': fixed_parameters,
            'timestamp': datetime.now().isoformat()
        }
        if case is not None:
            config['method_case'] = case.to_dict()
        
        return config
    
    def open_plot_windows(self, config):
        """Open separate windows for live plots"""
        from plot_windows import ConvergencePlotWindow, ParetoPlotWindow, ConstraintPlotWindow
        
        # Convergence plot window
        self.convergence_window = ConvergencePlotWindow(config)
        self.convergence_window.show()
        
        # Pareto plot window (only if 2 objectives)
        if len(config['objectives']) == 2:
            self.pareto_window = ParetoPlotWindow(config)
            self.pareto_window.show()
        else:
            self.pareto_window = None
        
        # Constraint plot window (Figure 11 style; the OC design space only)
        if config.get('method_case'):
            self.constraint_window = None
            return
        width, height = self.get_geometry_params()
        self.constraint_window = ConstraintPlotWindow(config, width=width, height=height)
        self.constraint_window.show()
    
    # Slot methods for worker signals
    def on_progress_update(self, current_gen, total_gen):
        """Update progress bars"""
        overall_pct = int((current_gen / total_gen) * 100)
        self.overall_progress.setValue(overall_pct)
        self.status_label.setText(f"Generation {current_gen}/{total_gen}")
        
    def on_design_evaluated(self, design_id, metrics):
        """Handle design evaluation"""
        # Update statistics
        self.designs_evaluated_label.setText(str(design_id))
        
        # Skip failed designs (NaN values)
        if not metrics.get('success', False):
            return
        
        # Update best CL/CD value (maximize)
        ld_value = None
        if 'CL/CD' in metrics and not np.isnan(metrics['CL/CD']):
            ld_value = metrics['CL/CD']
        
        if ld_value is not None:
            current_best = self.best_ld_label.text()
            try:
                current_val = float(current_best) if current_best != 'N/A' else -1
                if ld_value > current_val:
                    self.best_ld_label.setText(f"{ld_value:.3f}")
            except:
                self.best_ld_label.setText(f"{ld_value:.3f}")
        
        # Update best Volume value (maximize)
        vol_value = None
        if 'Volume' in metrics and not np.isnan(metrics['Volume']):
            vol_value = metrics['Volume']
        elif 'Volume_min' in metrics and not np.isnan(metrics['Volume_min']):
            vol_value = metrics['Volume_min']
        
        if vol_value is not None:
            current_best = self.best_volume_label.text()
            try:
                current_val = float(current_best.split()[0]) if current_best != 'N/A' else -1
                if vol_value > current_val:
                    self.best_volume_label.setText(f"{vol_value:.3f} m³")
            except:
                self.best_volume_label.setText(f"{vol_value:.3f} m³")
        
        # Update plot windows if they exist
        if hasattr(self, 'convergence_window') and self.convergence_window:
            self.convergence_window.add_design(design_id, metrics)
        
        if hasattr(self, 'pareto_window') and self.pareto_window:
            self.pareto_window.add_design(design_id, metrics)
        
        # Update constraint window
        if hasattr(self, 'constraint_window') and self.constraint_window:
            X1 = metrics.get('X1', 0)
            X2 = metrics.get('X2', 0)
            obj_vals = {obj['name']: metrics.get(obj['name'], 0) 
                       for obj in self.last_config.get('objectives', []) if obj['name'] in metrics}
            self.constraint_window.add_design(X1, X2, obj_vals, is_pareto=False)
    
    def on_generation_complete(self, generation, stats):
        """Handle generation completion"""
        # Update generation progress
        self.generation_progress.setValue(100)
        
    def on_optimization_complete(self, results_folder):
        """Handle optimization completion"""
        self.optimization_running = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("✅ Optimization Complete!")
        self.overall_progress.setValue(100)
        self.last_results_folder = results_folder
        is_method_case = bool((self.last_config or {}).get('method_case'))
        self.apply_best_btn.setEnabled(is_method_case)
        
        # Update Pareto count from pareto_front.csv
        try:
            pareto_csv = Path(results_folder) / "pareto_front.csv"
            if pareto_csv.exists():
                import pandas as pd
                pareto_df = pd.read_csv(pareto_csv)
                n_pareto = len(pareto_df)
                self.pareto_count_label.setText(str(n_pareto))
        except Exception as e:
            print(f"Could not read Pareto count: {e}")
        
        # Generate constraint plot (similar to Figure 11 in paper; OC design space only)
        try:
            from optimization_utils import OptimizationResults
            
            # Get geometry parameters
            width, height = self.get_geometry_params()
            
            # Find the primary objective (first enabled objective)
            primary_objective = None
            minimize = True
            for obj_name, controls in self.objective_controls.items():
                if controls['enable'].isChecked():
                    primary_objective = obj_name
                    minimize = controls['mode'].currentText().lower() == 'minimize'
                    break
            
            if primary_objective and not is_method_case:
                # Create results object to access plot function
                results = OptimizationResults.__new__(OptimizationResults)
                results.plots_dir = Path(results_folder) / "plots"
                results.designs_csv = Path(results_folder) / "designs.csv"
                results.pareto_csv = Path(results_folder) / "pareto_front.csv"
                
                if results.designs_csv.exists():
                    results.plot_constraint_vs_objective(
                        objective_name=primary_objective,
                        width=width,
                        height=height,
                        minimize=minimize,
                        save_name="constraint_plot.png"
                    )
                    self.log_message("✓ Generated constraint plot (Figure 11 style)")
        except Exception as e:
            print(f"Could not generate constraint plot: {e}")
        
        # Load final plots in windows
        if hasattr(self, 'convergence_window') and self.convergence_window:
            self.convergence_window.load_final_plot(results_folder)
        
        if hasattr(self, 'pareto_window') and self.pareto_window:
            self.pareto_window.load_final_plot(results_folder)
        
        if hasattr(self, 'constraint_window') and self.constraint_window:
            self.constraint_window.load_final_plot(results_folder)
        
        QMessageBox.information(self, "Optimization Complete",
                              f"Optimization finished successfully!\n\n"
                              f"Results saved to:\n{results_folder}")
    
    def on_error_occurred(self, error_msg):
        """Handle optimization error"""
        self.optimization_running = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("❌ Optimization Failed")
        
        QMessageBox.critical(self, "Optimization Error", error_msg)
        
    def stop_optimization(self):
        """Stop the optimization process"""
        if self.optimization_worker and self.optimization_worker.isRunning():
            self.optimization_worker.stop()
            self.status_label.setText("Stopping optimization...")
        
    def validate_configuration(self):
        """Validate optimization configuration before starting"""
        is_multi_objective = self.opt_mode_combo.currentIndex() == 1
        
        enabled_objectives = [name for name, controls in self.objective_controls.items() 
                            if controls['enable'].isChecked()]
        n_objectives = len(enabled_objectives)
        
        if n_objectives == 0:
            QMessageBox.warning(self, "Configuration Error",
                              "Please enable at least one objective!")
            return False
        
        if is_multi_objective:
            # Multi-objective mode: need exactly 2 objectives
            if n_objectives != 2:
                QMessageBox.warning(self, "Configuration Error",
                                  f"Multi-objective mode requires exactly 2 objectives.\n\n"
                                  f"Currently selected: {n_objectives}\n\n"
                                  f"Recommended: CL/CD (maximize) + Volume (maximize)\n\n"
                                  f"Click the preset button or manually select 2 objectives.")
                return False
        else:
            # Single-objective mode: need exactly 1 objective
            if n_objectives != 1:
                QMessageBox.warning(self, "Configuration Error",
                                  f"Single-objective mode requires exactly 1 objective.\n\n"
                                  f"Currently selected: {n_objectives}\n\n"
                                  f"Either select only 1 objective, or switch to Multi-Objective mode.")
                return False
            
        # Check design variable ranges
        case = self.current_case()
        if case is not None:
            if not case.active():
                QMessageBox.warning(self, "Configuration Error",
                                    "Tick at least one parameter to vary.")
                return False
            for v in case.active():
                if v.lower >= v.upper:
                    QMessageBox.warning(self, "Configuration Error",
                                        f"{v.label}: Min value must be less than Max value!")
                    return False
                if not v.lower <= v.value <= v.upper:
                    self.log_message(f"⚠ {v.label}: the current value {v.value:g} is outside "
                                     f"{v.lower:g} to {v.upper:g}")
        for var_name, spins in (self.design_var_spins.items() if case is None else ()):
            min_val = spins['min'].value()
            max_val = spins['max'].value()
            if min_val >= max_val:
                QMessageBox.warning(self, "Configuration Error",
                                  f"{var_name}: Min value must be less than Max value!")
                return False
        
        mode_str = "Multi-objective" if is_multi_objective else "Single-objective"
        self.log_message(f"✓ Configuration valid: {mode_str} with {n_objectives} objective(s)")
        return True
        
    def save_configuration(self):
        """Save current configuration to JSON file"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Configuration", "", "JSON Files (*.json)"
        )
        if filename:
            try:
                config = self.build_configuration(*self._oc_flight_and_size())
                
                with open(filename, 'w') as f:
                    json.dump(config, f, indent=2)
                
                self.log_message(f"✓ Configuration saved: {filename}")
                QMessageBox.information(self, "Success", 
                                      f"Configuration saved to:\n{filename}")
            except Exception as e:
                self.log_message(f"❌ Error saving configuration: {str(e)}")
                QMessageBox.critical(self, "Error", 
                                   f"Failed to save configuration:\n{str(e)}")
            
    def load_configuration(self):
        """Load configuration from JSON file"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Configuration", "", "JSON Files (*.json)"
        )
        if filename:
            try:
                with open(filename, 'r') as f:
                    config = json.load(f)
                
                # A design tab's case: restore it (applying back finds the tab by name)
                if config.get('method_case'):
                    from method_optimization import OptimizationCase
                    case = OptimizationCase.from_dict(config['method_case'])
                    tab = None
                    if self.parent_gui is not None and hasattr(self.parent_gui, 'design_tab'):
                        tab = self.parent_gui.design_tab(case.title)
                    self.set_method_case(case, tab)
                else:
                    self.method_combo.setCurrentIndex(0)

                # Load design variables
                for var in config['design_variables']:
                    var_name = var['name']
                    if var_name in self.design_var_spins:
                        self.design_var_spins[var_name]['min'].setValue(var['min'])
                        self.design_var_spins[var_name]['max'].setValue(var['max'])
                
                # Load objectives
                for obj_name, controls in self.objective_controls.items():
                    # First, disable all
                    controls['enable'].setChecked(False)
                
                for obj in config['objectives']:
                    obj_name = obj['name']
                    if obj_name in self.objective_controls:
                        self.objective_controls[obj_name]['enable'].setChecked(True)
                        self.objective_controls[obj_name]['mode'].setCurrentText(obj['mode'])
                
                # Load constraints
                for const_name, controls in self.constraint_controls.items():
                    # First, disable all
                    controls['enable'].setChecked(False)
                
                for const in config['constraints']:
                    const_name = const['name']
                    if const_name in self.constraint_controls:
                        self.constraint_controls[const_name]['enable'].setChecked(True)
                        self.constraint_controls[const_name]['value'].setValue(const['value'])
                
                # Load algorithm parameters
                algo = config['algorithm']
                self.pop_size_spin.setValue(algo['pop_size'])
                self.n_gen_spin.setValue(algo['n_gen'])
                self.n_cores_spin.setValue(algo['n_cores'])
                
                # Set mesh quality
                mesh_size = algo.get('mesh_size', 0.15)
                if mesh_size >= 0.14:
                    self.mesh_quality_combo.setCurrentText('Coarse (fast)')
                elif mesh_size >= 0.08:
                    self.mesh_quality_combo.setCurrentText('Medium')
                else:
                    self.mesh_quality_combo.setCurrentText('Fine (slow)')

                # Base pressure and skin friction; configurations saved before
                # these settings existed ran PySAGAS as is
                from aero_corrections import INVISCID, AeroCorrections
                corrections = config.get('simulation_params', {}).get('corrections')
                self.opt_corrections_group.set_corrections(
                    AeroCorrections.from_dict(corrections) if corrections else INVISCID)

                self.log_message(f"✓ Configuration loaded: {filename}")
                QMessageBox.information(self, "Success",
                                      f"Configuration loaded from:\n{filename}")
            except Exception as e:
                self.log_message(f"❌ Error loading configuration: {str(e)}")
                QMessageBox.critical(self, "Error",
                                   f"Failed to load configuration:\n{str(e)}")


if __name__ == '__main__':
    """Test the optimization tab standalone"""
    from PyQt5.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    tab = OptimizationTab()
    tab.setWindowTitle("Waverider Optimization")
    tab.resize(1200, 800)
    tab.show()
    sys.exit(app.exec_())
