#!/usr/bin/env python3
"""
TEE Monitoring Data Visualization Script

Generates graphs from the monitoring data collected by monitor_tee_service.py

Usage:
    python3 visualize_monitoring_data.py [--session SESSION_ID] [--results-dir DIR]
"""

import argparse
import csv
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import sys

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.figure import Figure
except ImportError:
    print("Error: matplotlib is required for visualization")
    print("Install with: pip install matplotlib --break-system-packages")
    sys.exit(1)


class MonitoringDataVisualizer:
    def __init__(self, results_dir: str = "./monitoring_results", session_id: str = None):
        self.results_dir = Path(results_dir)
        self.session_id = session_id
        
        # Find session files
        if session_id:
            self.chat_file = self.results_dir / f"chat_requests_{session_id}.csv"
            self.stream_file = self.results_dir / f"stream_requests_{session_id}.csv"
            self.health_file = self.results_dir / f"health_checks_{session_id}.csv"
            self.summary_file = self.results_dir / f"session_summary_{session_id}.json"
        else:
            # Find most recent session
            health_files = list(self.results_dir.glob("health_checks_*.csv"))
            if not health_files:
                raise FileNotFoundError(f"No monitoring results found in {self.results_dir}")
            
            # Sort by modification time
            health_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            self.health_file = health_files[0]
            
            # Extract session ID from filename
            self.session_id = self.health_file.stem.replace("health_checks_", "")
            self.chat_file = self.results_dir / f"chat_requests_{self.session_id}.csv"
            self.stream_file = self.results_dir / f"stream_requests_{self.session_id}.csv"
            self.summary_file = self.results_dir / f"session_summary_{self.session_id}.json"
        
        print(f"Loading data from session: {self.session_id}")
        print(f"  Health file: {self.health_file.name}")
        print(f"  Chat file: {self.chat_file.name}")
        print(f"  Stream file: {self.stream_file.name}")
        
        # Load data
        self.health_data = self._load_csv(self.health_file)
        self.chat_data = self._load_csv(self.chat_file)
        self.stream_data = self._load_csv(self.stream_file)
        
        # Load summary if exists
        self.summary = {}
        if self.summary_file.exists():
            with open(self.summary_file, 'r') as f:
                self.summary = json.load(f)
    
    def _load_csv(self, filepath: Path) -> List[Dict[str, str]]:
        """Load CSV file into list of dicts"""
        if not filepath.exists():
            print(f"Warning: {filepath.name} not found")
            return []
        
        data = []
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        
        return data
    
    def _parse_timestamp(self, ts_str: str) -> datetime:
        """Parse ISO timestamp string"""
        return datetime.fromisoformat(ts_str)
    
    def _safe_float(self, value: str, default: float = 0.0) -> float:
        """Safely convert string to float"""
        try:
            return float(value) if value else default
        except (ValueError, TypeError):
            return default
    
    def plot_memory_usage(self, ax: plt.Axes):
        """Plot memory usage over time"""
        if not self.health_data:
            ax.text(0.5, 0.5, 'No health data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        timestamps = [self._parse_timestamp(row['timestamp']) for row in self.health_data]
        process_memory = [self._safe_float(row['process_memory_mb']) for row in self.health_data]
        system_memory = [self._safe_float(row['system_used_memory_mb']) for row in self.health_data]
        
        ax.plot(timestamps, process_memory, 'o-', label='Process Memory (MB)', 
               linewidth=2, markersize=4)
        ax.plot(timestamps, system_memory, 's-', label='System Memory (MB)', 
               linewidth=2, markersize=4, alpha=0.7)
        
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Memory (MB)', fontsize=12)
        ax.set_title('Memory Usage Over Time', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    def plot_memory_percentage(self, ax: plt.Axes):
        """Plot memory percentage over time"""
        if not self.health_data:
            ax.text(0.5, 0.5, 'No health data available',
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        timestamps = [self._parse_timestamp(row['timestamp']) for row in self.health_data]
        process_pct = [self._safe_float(row['process_memory_percent']) for row in self.health_data]
        system_pct = [self._safe_float(row['system_memory_percent']) for row in self.health_data]
        
        ax.plot(timestamps, process_pct, 'o-', label='Process Memory %',
               linewidth=2, markersize=4)
        ax.plot(timestamps, system_pct, 's-', label='System Memory %',
               linewidth=2, markersize=4, alpha=0.7)
        
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Memory (%)', fontsize=12)
        ax.set_title('Memory Percentage Over Time', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    def plot_request_latency(self, ax: plt.Axes):
        """Plot request latency over time"""
        if not self.chat_data and not self.stream_data:
            ax.text(0.5, 0.5, 'No request data available',
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Plot chat latency
        if self.chat_data:
            chat_timestamps = [self._parse_timestamp(row['timestamp']) for row in self.chat_data]
            chat_latencies = [self._safe_float(row['latency_ms']) for row in self.chat_data]
            ax.plot(chat_timestamps, chat_latencies, 'o-', label='Chat Request',
                   linewidth=2, markersize=5, alpha=0.8)
        
        # Plot stream latency
        if self.stream_data:
            stream_timestamps = [self._parse_timestamp(row['timestamp']) for row in self.stream_data]
            stream_latencies = [self._safe_float(row['latency_ms']) for row in self.stream_data]
            ax.plot(stream_timestamps, stream_latencies, 's-', label='Stream Request',
                   linewidth=2, markersize=5, alpha=0.8)
        
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Latency (ms)', fontsize=12)
        ax.set_title('Request Latency Over Time', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    def plot_system_resources(self, ax: plt.Axes):
        """Plot system resource usage (threads, FDs, connections)"""
        if not self.health_data:
            ax.text(0.5, 0.5, 'No health data available',
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        timestamps = [self._parse_timestamp(row['timestamp']) for row in self.health_data]
        threads = [self._safe_float(row['threads']) for row in self.health_data]
        fds = [self._safe_float(row['num_fds']) for row in self.health_data]
        connections = [self._safe_float(row['connections']) for row in self.health_data]
        
        ax.plot(timestamps, threads, 'o-', label='Threads', linewidth=2, markersize=4)
        ax.plot(timestamps, fds, 's-', label='File Descriptors', linewidth=2, markersize=4)
        ax.plot(timestamps, connections, '^-', label='Connections', linewidth=2, markersize=4)
        
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('System Resources Over Time', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    def plot_token_usage(self, ax: plt.Axes):
        """Plot token usage over time"""
        if not self.chat_data:
            ax.text(0.5, 0.5, 'No chat data available',
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        timestamps = [self._parse_timestamp(row['timestamp']) for row in self.chat_data]
        prompt_tokens = [self._safe_float(row['prompt_tokens']) for row in self.chat_data]
        completion_tokens = [self._safe_float(row['completion_tokens']) for row in self.chat_data]
        total_tokens = [self._safe_float(row['total_tokens']) for row in self.chat_data]
        
        ax.plot(timestamps, prompt_tokens, 'o-', label='Prompt Tokens',
               linewidth=2, markersize=4)
        ax.plot(timestamps, completion_tokens, 's-', label='Completion Tokens',
               linewidth=2, markersize=4)
        ax.plot(timestamps, total_tokens, '^-', label='Total Tokens',
               linewidth=2, markersize=4)
        
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Token Count', fontsize=12)
        ax.set_title('Token Usage Over Time', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    def plot_gc_objects(self, ax: plt.Axes):
        """Plot garbage collection objects over time"""
        if not self.health_data:
            ax.text(0.5, 0.5, 'No health data available',
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        timestamps = [self._parse_timestamp(row['timestamp']) for row in self.health_data]
        gc_objects = [self._safe_float(row['gc_objects']) for row in self.health_data]
        
        ax.plot(timestamps, gc_objects, 'o-', color='purple',
               linewidth=2, markersize=4)
        
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Object Count', fontsize=12)
        ax.set_title('Garbage Collector Objects Over Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    def generate_report(self):
        """Generate comprehensive visualization report"""
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle(f'TEE LLM Router Monitoring Report\nSession: {self.session_id}',
                    fontsize=16, fontweight='bold')
        
        # Create 3x2 grid of subplots
        gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, 0])
        self.plot_memory_usage(ax1)
        
        ax2 = fig.add_subplot(gs[0, 1])
        self.plot_memory_percentage(ax2)
        
        ax3 = fig.add_subplot(gs[1, 0])
        self.plot_request_latency(ax3)
        
        ax4 = fig.add_subplot(gs[1, 1])
        self.plot_system_resources(ax4)
        
        ax5 = fig.add_subplot(gs[2, 0])
        self.plot_token_usage(ax5)
        
        ax6 = fig.add_subplot(gs[2, 1])
        self.plot_gc_objects(ax6)
        
        # Save figure
        output_file = self.results_dir / f"monitoring_report_{self.session_id}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n✓ Report saved to: {output_file}")
        
        # Also save individual plots
        self._save_individual_plots()
        
        return output_file
    
    def _save_individual_plots(self):
        """Save each plot as individual file"""
        plots = [
            ('memory_usage', self.plot_memory_usage, 'Memory Usage'),
            ('memory_percentage', self.plot_memory_percentage, 'Memory Percentage'),
            ('request_latency', self.plot_request_latency, 'Request Latency'),
            ('system_resources', self.plot_system_resources, 'System Resources'),
            ('token_usage', self.plot_token_usage, 'Token Usage'),
            ('gc_objects', self.plot_gc_objects, 'GC Objects')
        ]
        
        for filename, plot_func, title in plots:
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_func(ax)
            
            output_file = self.results_dir / f"{filename}_{self.session_id}.png"
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"  ✓ {title}: {output_file.name}")
    
    def print_statistics(self):
        """Print summary statistics"""
        print(f"\n{'='*60}")
        print("Summary Statistics")
        print(f"{'='*60}\n")
        
        if self.summary:
            print(f"Session Duration: {self.summary.get('duration_hours_actual', 'N/A'):.2f} hours")
            print(f"Iterations Completed: {self.summary.get('iterations_completed', 'N/A')}")
            print(f"Errors: {len(self.summary.get('errors', []))}")
        
        if self.health_data:
            process_mem = [self._safe_float(row['process_memory_mb']) for row in self.health_data]
            system_mem_pct = [self._safe_float(row['system_memory_percent']) for row in self.health_data]
            
            print(f"\nMemory Usage:")
            print(f"  Process Memory (MB):")
            print(f"    Min: {min(process_mem):.2f}")
            print(f"    Max: {max(process_mem):.2f}")
            print(f"    Avg: {sum(process_mem)/len(process_mem):.2f}")
            print(f"    Growth: {max(process_mem) - min(process_mem):.2f} MB")
            
            print(f"  System Memory (%):")
            print(f"    Min: {min(system_mem_pct):.2f}")
            print(f"    Max: {max(system_mem_pct):.2f}")
            print(f"    Avg: {sum(system_mem_pct)/len(system_mem_pct):.2f}")
        
        if self.chat_data:
            chat_latencies = [self._safe_float(row['latency_ms']) for row in self.chat_data 
                            if row['success'] == 'True']
            
            if chat_latencies:
                print(f"\nChat Request Latency (ms):")
                print(f"  Min: {min(chat_latencies):.2f}")
                print(f"  Max: {max(chat_latencies):.2f}")
                print(f"  Avg: {sum(chat_latencies)/len(chat_latencies):.2f}")
                print(f"  Median: {sorted(chat_latencies)[len(chat_latencies)//2]:.2f}")
        
        if self.stream_data:
            stream_latencies = [self._safe_float(row['latency_ms']) for row in self.stream_data
                              if row['success'] == 'True']
            
            if stream_latencies:
                print(f"\nStream Request Latency (ms):")
                print(f"  Min: {min(stream_latencies):.2f}")
                print(f"  Max: {max(stream_latencies):.2f}")
                print(f"  Avg: {sum(stream_latencies)/len(stream_latencies):.2f}")
                print(f"  Median: {sorted(stream_latencies)[len(stream_latencies)//2]:.2f}")
        
        print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize TEE monitoring data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize most recent session
  python3 visualize_monitoring_data.py
  
  # Visualize specific session
  python3 visualize_monitoring_data.py --session 20260131_065500
  
  # Use custom results directory
  python3 visualize_monitoring_data.py --results-dir /path/to/results
        """
    )
    
    parser.add_argument(
        '--session',
        type=str,
        help='Session ID (timestamp) to visualize. If not provided, uses most recent.'
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default='./monitoring_results',
        help='Directory containing monitoring results (default: ./monitoring_results)'
    )
    
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Save plots without displaying them'
    )
    
    args = parser.parse_args()
    
    try:
        # Create visualizer
        viz = MonitoringDataVisualizer(
            results_dir=args.results_dir,
            session_id=args.session
        )
        
        # Print statistics
        viz.print_statistics()
        
        # Generate report
        print("\nGenerating visualization report...")
        output_file = viz.generate_report()
        
        if not args.no_display:
            print("\nDisplaying plots...")
            plt.show()
        
        print("\n✓ Visualization complete!")
        
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
