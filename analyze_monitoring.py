#!/usr/bin/env python3
"""
Quick Analysis Tool for TEE Monitoring Data

Analyzes monitoring data to detect memory leaks, performance degradation,
and other issues.

Usage:
    python3 analyze_monitoring.py [--session SESSION_ID] [--results-dir DIR]
"""

import argparse
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import statistics


class MonitoringAnalyzer:
    def __init__(self, results_dir: str = "./monitoring_results", session_id: str = None):
        self.results_dir = Path(results_dir)
        
        # Find session files
        if session_id:
            self.health_file = self.results_dir / f"health_checks_{session_id}.csv"
            self.chat_file = self.results_dir / f"chat_requests_{session_id}.csv"
            self.stream_file = self.results_dir / f"stream_requests_{session_id}.csv"
            self.session_id = session_id
        else:
            # Find most recent session
            health_files = list(self.results_dir.glob("health_checks_*.csv"))
            if not health_files:
                raise FileNotFoundError(f"No monitoring results found in {self.results_dir}")
            
            health_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            self.health_file = health_files[0]
            self.session_id = self.health_file.stem.replace("health_checks_", "")
            self.chat_file = self.results_dir / f"chat_requests_{self.session_id}.csv"
            self.stream_file = self.results_dir / f"stream_requests_{self.session_id}.csv"
        
        # Load data
        self.health_data = self._load_csv(self.health_file)
        self.chat_data = self._load_csv(self.chat_file)
        self.stream_data = self._load_csv(self.stream_file)
    
    def _load_csv(self, filepath: Path) -> List[Dict[str, str]]:
        if not filepath.exists():
            return []
        
        data = []
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        return data
    
    def _safe_float(self, value: str, default: float = 0.0) -> float:
        try:
            return float(value) if value else default
        except (ValueError, TypeError):
            return default
    
    def analyze_memory_leak(self) -> Dict:
        """Detect potential memory leaks"""
        if not self.health_data or len(self.health_data) < 2:
            return {'status': 'insufficient_data'}
        
        memory_values = [self._safe_float(row['process_memory_mb']) for row in self.health_data]
        timestamps = [datetime.fromisoformat(row['timestamp']) for row in self.health_data]
        
        # Calculate memory growth
        initial_memory = memory_values[0]
        final_memory = memory_values[-1]
        memory_growth = final_memory - initial_memory
        
        # Calculate growth rate
        duration_seconds = (timestamps[-1] - timestamps[0]).total_seconds()
        growth_rate_mb_per_hour = (memory_growth / duration_seconds) * 3600 if duration_seconds > 0 else 0
        
        # Calculate linear regression slope
        n = len(memory_values)
        x_values = list(range(n))
        x_mean = sum(x_values) / n
        y_mean = sum(memory_values) / n
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, memory_values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        slope = numerator / denominator if denominator != 0 else 0
        
        # Determine leak severity
        if growth_rate_mb_per_hour > 10:
            severity = "CRITICAL"
        elif growth_rate_mb_per_hour > 5:
            severity = "HIGH"
        elif growth_rate_mb_per_hour > 1:
            severity = "MEDIUM"
        elif growth_rate_mb_per_hour > 0.1:
            severity = "LOW"
        else:
            severity = "NONE"
        
        return {
            'status': 'analyzed',
            'initial_memory_mb': initial_memory,
            'final_memory_mb': final_memory,
            'memory_growth_mb': memory_growth,
            'growth_rate_mb_per_hour': growth_rate_mb_per_hour,
            'slope': slope,
            'severity': severity,
            'duration_hours': duration_seconds / 3600,
            'data_points': n
        }
    
    def analyze_latency_degradation(self) -> Dict:
        """Detect performance degradation over time"""
        if not self.chat_data or len(self.chat_data) < 2:
            return {'status': 'insufficient_data'}
        
        latencies = [self._safe_float(row['latency_ms']) 
                    for row in self.chat_data if row['success'] == 'True']
        
        if not latencies or len(latencies) < 2:
            return {'status': 'insufficient_data'}
        
        # Split into first half and second half
        mid = len(latencies) // 2
        first_half = latencies[:mid]
        second_half = latencies[mid:]
        
        avg_first = statistics.mean(first_half)
        avg_second = statistics.mean(second_half)
        
        degradation_pct = ((avg_second - avg_first) / avg_first * 100) if avg_first > 0 else 0
        
        # Calculate linear regression
        n = len(latencies)
        x_values = list(range(n))
        x_mean = sum(x_values) / n
        y_mean = sum(latencies) / n
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, latencies))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        slope = numerator / denominator if denominator != 0 else 0
        
        # Determine severity
        if degradation_pct > 50:
            severity = "CRITICAL"
        elif degradation_pct > 25:
            severity = "HIGH"
        elif degradation_pct > 10:
            severity = "MEDIUM"
        elif degradation_pct > 0:
            severity = "LOW"
        else:
            severity = "NONE"
        
        return {
            'status': 'analyzed',
            'avg_latency_first_half_ms': avg_first,
            'avg_latency_second_half_ms': avg_second,
            'degradation_pct': degradation_pct,
            'slope': slope,
            'severity': severity,
            'min_latency_ms': min(latencies),
            'max_latency_ms': max(latencies),
            'median_latency_ms': statistics.median(latencies),
            'data_points': len(latencies)
        }
    
    def analyze_resource_leaks(self) -> Dict:
        """Detect file descriptor and connection leaks"""
        if not self.health_data or len(self.health_data) < 2:
            return {'status': 'insufficient_data'}
        
        fds = [self._safe_float(row['num_fds']) for row in self.health_data]
        connections = [self._safe_float(row['connections']) for row in self.health_data]
        gc_objects = [self._safe_float(row['gc_objects']) for row in self.health_data]
        
        initial_fds = fds[0]
        final_fds = fds[-1]
        fd_growth = final_fds - initial_fds
        
        initial_gc = gc_objects[0]
        final_gc = gc_objects[-1]
        gc_growth = final_gc - initial_gc
        
        # Determine severity
        fd_severity = "NONE"
        if fd_growth > 50:
            fd_severity = "CRITICAL"
        elif fd_growth > 20:
            fd_severity = "HIGH"
        elif fd_growth > 10:
            fd_severity = "MEDIUM"
        elif fd_growth > 0:
            fd_severity = "LOW"
        
        gc_severity = "NONE"
        gc_growth_pct = (gc_growth / initial_gc * 100) if initial_gc > 0 else 0
        if gc_growth_pct > 100:
            gc_severity = "HIGH"
        elif gc_growth_pct > 50:
            gc_severity = "MEDIUM"
        elif gc_growth_pct > 25:
            gc_severity = "LOW"
        
        return {
            'status': 'analyzed',
            'initial_fds': initial_fds,
            'final_fds': final_fds,
            'fd_growth': fd_growth,
            'fd_severity': fd_severity,
            'avg_connections': statistics.mean(connections),
            'max_connections': max(connections),
            'initial_gc_objects': initial_gc,
            'final_gc_objects': final_gc,
            'gc_growth': gc_growth,
            'gc_growth_pct': gc_growth_pct,
            'gc_severity': gc_severity
        }
    
    def analyze_stability(self) -> Dict:
        """Analyze overall system stability"""
        if not self.chat_data and not self.stream_data:
            return {'status': 'insufficient_data'}
        
        # Request success rates
        chat_total = len(self.chat_data)
        chat_success = sum(1 for row in self.chat_data if row['success'] == 'True')
        chat_success_rate = (chat_success / chat_total * 100) if chat_total > 0 else 0
        
        stream_total = len(self.stream_data)
        stream_success = sum(1 for row in self.stream_data if row['success'] == 'True')
        stream_success_rate = (stream_success / stream_total * 100) if stream_total > 0 else 0
        
        # Collect error types
        errors = {}
        for row in self.chat_data:
            if row['success'] == 'False' and row['error_message']:
                errors[row['error_message']] = errors.get(row['error_message'], 0) + 1
        
        for row in self.stream_data:
            if row['success'] == 'False' and row['error_message']:
                errors[row['error_message']] = errors.get(row['error_message'], 0) + 1
        
        return {
            'status': 'analyzed',
            'chat_requests_total': chat_total,
            'chat_requests_success': chat_success,
            'chat_success_rate_pct': chat_success_rate,
            'stream_requests_total': stream_total,
            'stream_requests_success': stream_success,
            'stream_success_rate_pct': stream_success_rate,
            'unique_errors': len(errors),
            'error_summary': errors
        }
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        print(f"\n{'='*70}")
        print(f"TEE LLM Router - Monitoring Analysis Report")
        print(f"Session: {self.session_id}")
        print(f"{'='*70}\n")
        
        # Memory leak analysis
        print("━" * 70)
        print("MEMORY LEAK ANALYSIS")
        print("━" * 70)
        memory_result = self.analyze_memory_leak()
        
        if memory_result['status'] == 'analyzed':
            print(f"\nSeverity: {memory_result['severity']}")
            print(f"  Initial Memory: {memory_result['initial_memory_mb']:.2f} MB")
            print(f"  Final Memory: {memory_result['final_memory_mb']:.2f} MB")
            print(f"  Total Growth: {memory_result['memory_growth_mb']:.2f} MB")
            print(f"  Growth Rate: {memory_result['growth_rate_mb_per_hour']:.2f} MB/hour")
            print(f"  Linear Slope: {memory_result['slope']:.4f}")
            print(f"  Duration: {memory_result['duration_hours']:.2f} hours")
            print(f"  Data Points: {memory_result['data_points']}")
            
            if memory_result['severity'] in ['CRITICAL', 'HIGH']:
                print(f"\n  ⚠️  WARNING: Potential memory leak detected!")
                print(f"      At this rate, memory will increase by "
                      f"{memory_result['growth_rate_mb_per_hour'] * 24:.1f} MB per day")
            elif memory_result['severity'] == 'MEDIUM':
                print(f"\n  ⚠️  CAUTION: Moderate memory growth detected")
            elif memory_result['severity'] == 'LOW':
                print(f"\n  ℹ️  INFO: Slight memory growth detected")
            else:
                print(f"\n  ✓ No significant memory leak detected")
        else:
            print("  Insufficient data for analysis")
        
        # Latency degradation analysis
        print(f"\n{'━' * 70}")
        print("PERFORMANCE DEGRADATION ANALYSIS")
        print("━" * 70)
        latency_result = self.analyze_latency_degradation()
        
        if latency_result['status'] == 'analyzed':
            print(f"\nSeverity: {latency_result['severity']}")
            print(f"  First Half Avg: {latency_result['avg_latency_first_half_ms']:.2f} ms")
            print(f"  Second Half Avg: {latency_result['avg_latency_second_half_ms']:.2f} ms")
            print(f"  Degradation: {latency_result['degradation_pct']:.2f}%")
            print(f"  Min Latency: {latency_result['min_latency_ms']:.2f} ms")
            print(f"  Max Latency: {latency_result['max_latency_ms']:.2f} ms")
            print(f"  Median Latency: {latency_result['median_latency_ms']:.2f} ms")
            print(f"  Data Points: {latency_result['data_points']}")
            
            if latency_result['severity'] in ['CRITICAL', 'HIGH']:
                print(f"\n  ⚠️  WARNING: Significant performance degradation!")
            elif latency_result['severity'] == 'MEDIUM':
                print(f"\n  ⚠️  CAUTION: Moderate performance degradation")
            elif latency_result['severity'] == 'LOW':
                print(f"\n  ℹ️  INFO: Slight performance variation")
            else:
                print(f"\n  ✓ No significant performance degradation")
        else:
            print("  Insufficient data for analysis")
        
        # Resource leak analysis
        print(f"\n{'━' * 70}")
        print("RESOURCE LEAK ANALYSIS")
        print("━" * 70)
        resource_result = self.analyze_resource_leaks()
        
        if resource_result['status'] == 'analyzed':
            print(f"\nFile Descriptors:")
            print(f"  Severity: {resource_result['fd_severity']}")
            print(f"  Initial FDs: {resource_result['initial_fds']:.0f}")
            print(f"  Final FDs: {resource_result['final_fds']:.0f}")
            print(f"  Growth: {resource_result['fd_growth']:.0f}")
            
            if resource_result['fd_severity'] in ['CRITICAL', 'HIGH']:
                print(f"  ⚠️  WARNING: File descriptor leak detected!")
            elif resource_result['fd_severity'] in ['MEDIUM', 'LOW']:
                print(f"  ℹ️  INFO: File descriptor count increased")
            else:
                print(f"  ✓ No file descriptor leak")
            
            print(f"\nConnections:")
            print(f"  Average: {resource_result['avg_connections']:.1f}")
            print(f"  Maximum: {resource_result['max_connections']:.0f}")
            
            print(f"\nGarbage Collection:")
            print(f"  Severity: {resource_result['gc_severity']}")
            print(f"  Initial Objects: {resource_result['initial_gc_objects']:.0f}")
            print(f"  Final Objects: {resource_result['final_gc_objects']:.0f}")
            print(f"  Growth: {resource_result['gc_growth']:.0f} ({resource_result['gc_growth_pct']:.1f}%)")
            
            if resource_result['gc_severity'] == 'HIGH':
                print(f"  ⚠️  WARNING: Significant increase in GC objects")
            elif resource_result['gc_severity'] in ['MEDIUM', 'LOW']:
                print(f"  ℹ️  INFO: Moderate increase in GC objects")
            else:
                print(f"  ✓ GC object count stable")
        else:
            print("  Insufficient data for analysis")
        
        # Stability analysis
        print(f"\n{'━' * 70}")
        print("SYSTEM STABILITY ANALYSIS")
        print("━" * 70)
        stability_result = self.analyze_stability()
        
        if stability_result['status'] == 'analyzed':
            print(f"\nChat Requests:")
            print(f"  Total: {stability_result['chat_requests_total']}")
            print(f"  Successful: {stability_result['chat_requests_success']}")
            print(f"  Success Rate: {stability_result['chat_success_rate_pct']:.2f}%")
            
            print(f"\nStream Requests:")
            print(f"  Total: {stability_result['stream_requests_total']}")
            print(f"  Successful: {stability_result['stream_requests_success']}")
            print(f"  Success Rate: {stability_result['stream_success_rate_pct']:.2f}%")
            
            if stability_result['unique_errors'] > 0:
                print(f"\nErrors Encountered: {stability_result['unique_errors']} unique types")
                for error_msg, count in sorted(stability_result['error_summary'].items(), 
                                              key=lambda x: x[1], reverse=True)[:5]:
                    print(f"  • {error_msg[:60]}... ({count} times)")
            else:
                print(f"\n  ✓ No errors encountered")
            
            if stability_result['chat_success_rate_pct'] < 95 or \
               stability_result['stream_success_rate_pct'] < 95:
                print(f"\n  ⚠️  WARNING: Success rate below 95%")
            else:
                print(f"\n  ✓ System stable")
        else:
            print("  Insufficient data for analysis")
        
        # Overall assessment
        print(f"\n{'='*70}")
        print("OVERALL ASSESSMENT")
        print(f"{'='*70}\n")
        
        issues = []
        
        if memory_result.get('severity') in ['CRITICAL', 'HIGH']:
            issues.append(f"• Memory leak detected ({memory_result['growth_rate_mb_per_hour']:.2f} MB/hour)")
        
        if latency_result.get('severity') in ['CRITICAL', 'HIGH']:
            issues.append(f"• Performance degradation ({latency_result['degradation_pct']:.1f}% slower)")
        
        if resource_result.get('fd_severity') in ['CRITICAL', 'HIGH']:
            issues.append(f"• File descriptor leak ({resource_result['fd_growth']:.0f} FDs leaked)")
        
        if resource_result.get('gc_severity') == 'HIGH':
            issues.append(f"• Excessive GC object growth ({resource_result['gc_growth_pct']:.1f}% increase)")
        
        if stability_result.get('chat_success_rate_pct', 100) < 95:
            issues.append(f"• Low request success rate ({stability_result['chat_success_rate_pct']:.1f}%)")
        
        if issues:
            print("⚠️  ISSUES DETECTED:\n")
            for issue in issues:
                print(f"  {issue}")
            print("\n  Recommendation: Investigate and fix the identified issues")
        else:
            print("✓ System appears healthy")
            print("  No critical issues detected during monitoring period")
        
        print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze TEE monitoring data for issues',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--session',
        type=str,
        help='Session ID to analyze. If not provided, uses most recent.'
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default='./monitoring_results',
        help='Directory containing monitoring results'
    )
    
    args = parser.parse_args()
    
    try:
        analyzer = MonitoringAnalyzer(
            results_dir=args.results_dir,
            session_id=args.session
        )
        analyzer.generate_report()
    except Exception as e:
        print(f"\nError: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
