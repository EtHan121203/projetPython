import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
import logging
from typing import Dict, Any, List, Optional

from simulator.discrete_event_simulator import DiscreteEventSimulator
from ui.map_visualization import NetworkMapVisualization

class SimulatorGUI:
    """
    GUI for the Smart City Simulator.
    Provides controls and visualization for the simulation.
    """
    def __init__(self, simulator: DiscreteEventSimulator):
        self.simulator = simulator
        self.is_running = False
        self.is_paused = False
        self.simulation_thread = None
        self.update_interval = 100  # ms
        self.logger = logging.getLogger("GUI")
        
        # Create the main window
        self.root = tk.Tk()
        self.root.title("Smart City Simulator")
        self.root.geometry("1200x800")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Create frames
        self.create_frames()
        
        # Create controls
        self.create_control_panel()
        
        # Create visualization area
        self.create_visualization_area()
        
        # Create status bar
        self.create_status_bar()
        
        # Initialize visualization elements
        self.network_visualization = NetworkMapVisualization(self.simulator.network)
        self.time_series_data = {
            "time": [],
            "events_processed": [],
            "data_generated": [],
            "data_processed": []
        }
        
        # Setup automatic updates
        self.root.after(self.update_interval, self.update_display)
    
    def create_frames(self):
        """Create the main frames for the GUI layout"""
        # Main layout: 
        # - Left: Control Panel + Statistics 
        # - Right: Visualization (Map, Charts)
        
        # Left panel
        self.left_panel = ttk.Frame(self.root, padding=10)
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y)
        
        # Control panel (top of left panel)
        self.control_frame = ttk.LabelFrame(self.left_panel, text="Control Panel", padding=10)
        self.control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Statistics panel (bottom of left panel)
        self.stats_frame = ttk.LabelFrame(self.left_panel, text="Statistics", padding=10)
        self.stats_frame.pack(fill=tk.BOTH, expand=True)
        
        # Right panel (visualization)
        self.right_panel = ttk.Frame(self.root, padding=10)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Map frame (top of right panel)
        self.map_frame = ttk.LabelFrame(self.right_panel, text="Network Map", padding=10)
        self.map_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Charts frame (bottom of right panel)
        self.charts_frame = ttk.LabelFrame(self.right_panel, text="Metrics", padding=10)
        self.charts_frame.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_frame = ttk.Frame(self.root, padding=(10, 5))
        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X)
    
    def create_control_panel(self):
        """Create simulation control widgets"""
        # Speed controls
        speed_frame = ttk.Frame(self.control_frame)
        speed_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(speed_frame, text="Simulation Speed:").pack(side=tk.LEFT)
        self.speed_var = tk.DoubleVar(value=1.0)
        self.speed_scale = ttk.Scale(
            speed_frame, 
            from_=0.1, 
            to=10.0, 
            orient=tk.HORIZONTAL, 
            variable=self.speed_var,
            command=self.on_speed_change
        )
        self.speed_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(speed_frame, textvariable=self.speed_var).pack(side=tk.LEFT)
        
        # Buttons: Start, Pause, Stop, Step, Reset
        buttons_frame = ttk.Frame(self.control_frame)
        buttons_frame.pack(fill=tk.X, pady=10)
        
        self.start_button = ttk.Button(buttons_frame, text="Start", command=self.start_simulation)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.pause_button = ttk.Button(buttons_frame, text="Pause", command=self.pause_simulation)
        self.pause_button.pack(side=tk.LEFT, padx=5)
        self.pause_button.config(state=tk.DISABLED)
        
        self.stop_button = ttk.Button(buttons_frame, text="Stop", command=self.stop_simulation)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        self.stop_button.config(state=tk.DISABLED)
        
        self.step_button = ttk.Button(buttons_frame, text="Step", command=self.step_simulation)
        self.step_button.pack(side=tk.LEFT, padx=5)
        
        self.reset_button = ttk.Button(buttons_frame, text="Reset", command=self.reset_simulation)
        self.reset_button.pack(side=tk.LEFT, padx=5)
        
        # Configuration and Scenario controls
        config_frame = ttk.Frame(self.control_frame)
        config_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(
            config_frame, 
            text="Load Scenario", 
            command=self.load_scenario
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            config_frame, 
            text="Save Results", 
            command=self.save_results
        ).pack(side=tk.LEFT, padx=5)
        
        # Simulation parameters
        params_frame = ttk.Frame(self.control_frame)
        params_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(params_frame, text="Max Time:").pack(side=tk.LEFT)
        self.max_time_var = tk.StringVar(value="3600")
        ttk.Entry(
            params_frame, 
            textvariable=self.max_time_var, 
            width=10
        ).pack(side=tk.LEFT, padx=5)
        ttk.Label(params_frame, text="seconds").pack(side=tk.LEFT, padx=(0, 15))
        
        ttk.Label(params_frame, text="Log Level:").pack(side=tk.LEFT)
        self.log_level = tk.StringVar(value="INFO")
        log_combo = ttk.Combobox(
            params_frame, 
            textvariable=self.log_level, 
            values=["DEBUG", "INFO", "WARNING", "ERROR"],
            width=8
        )
        log_combo.pack(side=tk.LEFT, padx=5)
        log_combo.bind("<<ComboboxSelected>>", self.update_log_level)
    
    def create_visualization_area(self):
        """Create visualization widgets for network map and metrics"""
        # Network map visualization (using Matplotlib)
        self.map_figure = plt.Figure(figsize=(6, 4), dpi=100)
        self.map_canvas = FigureCanvasTkAgg(self.map_figure, self.map_frame)
        self.map_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.map_axes = self.map_figure.add_subplot(111)
        
        # Metrics charts
        charts_container = ttk.Notebook(self.charts_frame)
        charts_container.pack(fill=tk.BOTH, expand=True)
        
        # Time series tab
        time_series_tab = ttk.Frame(charts_container)
        charts_container.add(time_series_tab, text="Time Series")
        
        self.time_series_figure = plt.Figure(figsize=(6, 3), dpi=100)
        self.time_series_canvas = FigureCanvasTkAgg(self.time_series_figure, time_series_tab)
        self.time_series_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.time_series_axes = self.time_series_figure.add_subplot(111)
        
        # Bar charts tab
        bar_charts_tab = ttk.Frame(charts_container)
        charts_container.add(bar_charts_tab, text="Bar Charts")
        
        self.bar_chart_figure = plt.Figure(figsize=(6, 3), dpi=100)
        self.bar_chart_canvas = FigureCanvasTkAgg(self.bar_chart_figure, bar_charts_tab)
        self.bar_chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.bar_chart_axes = self.bar_chart_figure.add_subplot(111)
        
        # Statistics tree view
        self.stats_tree = ttk.Treeview(self.stats_frame, columns=("value",), show="tree")
        self.stats_tree.heading("#0", text="Metric")
        self.stats_tree.heading("value", text="Value")
        self.stats_tree.column("#0", width=150)
        self.stats_tree.column("value", width=100)
        self.stats_tree.pack(fill=tk.BOTH, expand=True)
    
    def create_status_bar(self):
        """Create status bar at bottom of window"""
        # Status variables
        self.status_text = tk.StringVar(value="Ready")
        self.time_text = tk.StringVar(value="Simulation time: 0.0s")
        self.events_text = tk.StringVar(value="Events: 0")
        
        # Status labels
        ttk.Label(self.status_frame, textvariable=self.status_text).pack(side=tk.LEFT)
        ttk.Separator(self.status_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill=tk.Y)
        ttk.Label(self.status_frame, textvariable=self.time_text).pack(side=tk.LEFT)
        ttk.Separator(self.status_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill=tk.Y)
        ttk.Label(self.status_frame, textvariable=self.events_text).pack(side=tk.LEFT)
    
    def start_simulation(self):
        """Start the simulation in a separate thread"""
        if self.is_running:
            return
        
        try:
            max_time = float(self.max_time_var.get())
            self.simulator.max_simulation_time = max_time
        except ValueError:
            messagebox.showerror("Error", "Invalid max simulation time")
            return
        
        # Update UI
        self.start_button.config(state=tk.DISABLED)
        self.pause_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.NORMAL)
        self.step_button.config(state=tk.DISABLED)
        self.status_text.set("Simulation running...")
        
        # Start simulation thread
        self.is_running = True
        self.is_paused = False
        self.simulation_thread = threading.Thread(target=self._run_simulation)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
    
    def _run_simulation(self):
        """Run the simulation (called in a separate thread)"""
        try:
            self.simulator.run()
        except Exception as e:
            self.logger.error(f"Simulation error: {str(e)}")
            # Update UI in main thread
            self.root.after(0, self._handle_simulation_error, str(e))
        finally:
            # Update UI in main thread
            if not self.is_paused:
                self.root.after(0, self._handle_simulation_complete)
    
    def _handle_simulation_error(self, error_msg):
        """Handle simulation errors (called in main thread)"""
        self.is_running = False
        messagebox.showerror("Simulation Error", f"An error occurred: {error_msg}")
        self.reset_ui_state()
    
    def _handle_simulation_complete(self):
        """Handle simulation completion (called in main thread)"""
        self.is_running = False
        self.status_text.set("Simulation complete")
        self.reset_ui_state()
    
    def pause_simulation(self):
        """Pause the running simulation"""
        if not self.is_running or self.is_paused:
            return
            
        self.is_paused = True
        self.simulator.pause()
        
        # Update UI
        self.status_text.set("Simulation paused")
        self.start_button.config(text="Resume", state=tk.NORMAL)
        self.step_button.config(state=tk.NORMAL)
    
    def stop_simulation(self):
        """Stop the simulation"""
        if not self.is_running:
            return
            
        self.simulator.stop()
        self.is_running = False
        self.is_paused = False
        
        # Update UI
        self.status_text.set("Simulation stopped")
        self.reset_ui_state()
    
    def step_simulation(self):
        """Execute a single step of the simulation"""
        if self.is_running and not self.is_paused:
            return
            
        try:
            event = self.simulator.step()
            if event:
                self.status_text.set(f"Processed: {event.event_type.value}")
            else:
                self.status_text.set("No more events to process")
        except Exception as e:
            self.logger.error(f"Step error: {str(e)}")
            messagebox.showerror("Error", f"Error in simulation step: {str(e)}")
    
    def reset_simulation(self):
        """Reset the simulation to initial state"""
        if self.is_running:
            self.stop_simulation()
            
        self.simulator.reset()
        self.time_series_data = {
            "time": [],
            "events_processed": [],
            "data_generated": [],
            "data_processed": []
        }
        
        # Update UI
        self.status_text.set("Simulation reset")
        self.time_text.set("Simulation time: 0.0s")
        self.events_text.set("Events: 0")
        self.reset_ui_state()
        self.update_display()
    
    def reset_ui_state(self):
        """Reset UI controls to initial state"""
        self.start_button.config(text="Start", state=tk.NORMAL)
        self.pause_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.DISABLED)
        self.step_button.config(state=tk.NORMAL)
    
    def load_scenario(self):
        """Load simulation scenario from file"""
        file_path = filedialog.askopenfilename(
            title="Load Scenario",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            # This would call your configuration loader
            messagebox.showinfo("Load Scenario", f"Loading {file_path}\n(Not implemented yet)")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading scenario: {str(e)}")
    
    def save_results(self):
        """Save simulation results to file"""
        file_path = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            # This would call your results saver
            messagebox.showinfo("Save Results", f"Saving to {file_path}\n(Not implemented yet)")
        except Exception as e:
            messagebox.showerror("Error", f"Error saving results: {str(e)}")
    
    def on_speed_change(self, *args):
        """Handle simulation speed change"""
        speed = self.speed_var.get()
        self.simulator.set_speed_factor(speed)
    
    def update_log_level(self, event):
        """Update logging level based on dropdown selection"""
        level_name = self.log_level.get()
        level = getattr(logging, level_name, logging.INFO)
        self.simulator.set_logging_level(level)
        logging.getLogger().setLevel(level)
    
    def update_display(self):
        """Update all display elements with current simulation data"""
        # Update status bar
        current_time = self.simulator.current_time
        self.time_text.set(f"Simulation time: {current_time:.2f}s")
        self.events_text.set(f"Events: {len(self.simulator.event_queue)}")
        
        # Update statistics
        self.update_statistics_display()
        
        # Update network map
        self.update_network_map()
        
        # Update charts
        self.update_charts()
        
        # Schedule next update
        if self.is_running or self.is_paused:
            self.root.after(self.update_interval, self.update_display)
    
    def update_statistics_display(self):
        """Update statistics tree view with current metrics"""
        # Clear existing items
        for item in self.stats_tree.get_children():
            self.stats_tree.delete(item)
            
        # Get metrics from simulator
        metrics = self.simulator.get_metrics()
        
        # Add metrics to tree
        simulation_id = self.stats_tree.insert("", "end", text="Simulation", open=True)
        self.stats_tree.insert(simulation_id, "end", text="Current Time", values=(f"{metrics.get('current_time', 0):.2f}s",))
        self.stats_tree.insert(simulation_id, "end", text="Events Processed", values=(metrics.get("processed_events", 0),))
        self.stats_tree.insert(simulation_id, "end", text="Real Time Elapsed", values=(f"{metrics.get('real_time_elapsed', 0):.2f}s",))
        
        data_id = self.stats_tree.insert("", "end", text="Data", open=True)
        self.stats_tree.insert(data_id, "end", text="Generated", values=(metrics.get("total_data_generated", 0),))
        self.stats_tree.insert(data_id, "end", text="Transmitted", values=(metrics.get("total_data_transmitted", 0),))
        self.stats_tree.insert(data_id, "end", text="Processed", values=(metrics.get("total_data_processed", 0),))
        
        # Add some network stats
        network_id = self.stats_tree.insert("", "end", text="Network", open=True)
        network_stats = self.simulator.network.get_network_stats()
        for key, value in network_stats.items():
            if isinstance(value, float):
                value_str = f"{value:.2f}"
            else:
                value_str = str(value)
            self.stats_tree.insert(network_id, "end", text=key, values=(value_str,))
    
    def update_network_map(self):
        """Update the network map visualization"""
        self.map_axes.clear()
        self.network_visualization.draw(self.map_axes)
        self.map_canvas.draw()
    
    def update_charts(self):
        """Update time series and bar charts"""
        # Update time series data
        current_time = self.simulator.current_time
        metrics = self.simulator.get_metrics()
        
        # Add current data point
        self.time_series_data["time"].append(current_time)
        self.time_series_data["events_processed"].append(metrics.get("processed_events", 0))
        self.time_series_data["data_generated"].append(metrics.get("total_data_generated", 0))
        self.time_series_data["data_processed"].append(metrics.get("total_data_processed", 0))
        
        # Limit data points to keep the chart readable
        max_points = 100
        if len(self.time_series_data["time"]) > max_points:
            for key in self.time_series_data:
                self.time_series_data[key] = self.time_series_data[key][-max_points:]
        
        # Draw time series
        self.time_series_axes.clear()
        self.time_series_axes.plot(
            self.time_series_data["time"], 
            self.time_series_data["events_processed"], 
            label="Events Processed"
        )
        self.time_series_axes.plot(
            self.time_series_data["time"], 
            self.time_series_data["data_generated"], 
            label="Data Generated"
        )
        self.time_series_axes.plot(
            self.time_series_data["time"], 
            self.time_series_data["data_processed"], 
            label="Data Processed"
        )
        self.time_series_axes.set_xlabel("Simulation Time (s)")
        self.time_series_axes.set_ylabel("Count")
        self.time_series_axes.legend()
        self.time_series_canvas.draw()
        
        # Draw bar chart for events by type
        events_by_type = metrics.get("events_by_type", {})
        self.bar_chart_axes.clear()
        if events_by_type:
            event_types = list(events_by_type.keys())
            event_counts = list(events_by_type.values())
            
            self.bar_chart_axes.bar(event_types, event_counts)
            self.bar_chart_axes.set_xlabel("Event Type")
            self.bar_chart_axes.set_ylabel("Count")
            self.bar_chart_axes.set_title("Events by Type")
            plt.setp(self.bar_chart_axes.get_xticklabels(), rotation=45, ha="right")
            self.bar_chart_figure.tight_layout()
            
        self.bar_chart_canvas.draw()
    
    def on_closing(self):
        """Handle window close event"""
        if self.is_running:
            self.stop_simulation()
        self.root.destroy()
    
    def run(self):
        """Start the GUI main loop"""
        self.root.mainloop()

def create_gui(simulator: DiscreteEventSimulator):
    """Create and return a simulator GUI instance"""
    return SimulatorGUI(simulator)