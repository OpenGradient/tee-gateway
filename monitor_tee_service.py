#!/usr/bin/env python3
"""
TEE LLM Router Monitoring Script

Monitors the TEE service by:
1. Periodically making chat/stream requests
2. Tracking request latency
3. Recording health metrics (especially memory usage)
4. Storing results to persistent CSV files

Usage:
    python3 monitor_tee_service.py [--duration HOURS] [--interval SECONDS]
"""

import argparse
import csv
import json
import os
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
import signal
import sys

# Configuration
HOST = "https://13.59.207.188:443"
GOOGLE_MODEL = "gemini-2.5-flash-lite"
PROMPT = "Describe to me the 7 layers of the network stack"
TEMPERATURE = 0.7
MAX_TOKENS = 150


class MonitoringSession:
    def __init__(self, duration_hours: float = 6.0, interval_seconds: int = 60, 
                 output_dir: str = "./monitoring_results"):
        self.duration_hours = duration_hours
        self.interval_seconds = interval_seconds
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Generate timestamp for this session
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Output files
        self.chat_results_file = self.output_dir / f"chat_requests_{self.session_timestamp}.csv"
        self.stream_results_file = self.output_dir / f"stream_requests_{self.session_timestamp}.csv"
        self.health_results_file = self.output_dir / f"health_checks_{self.session_timestamp}.csv"
        self.summary_file = self.output_dir / f"session_summary_{self.session_timestamp}.json"
        
        # Initialize CSV files
        self._init_csv_files()
        
        # Session state
        self.start_time = None
        self.end_time = None
        self.iteration_count = 0
        self.errors = []
        self.should_stop = False
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
    
    def _init_csv_files(self):
        """Initialize CSV files with headers"""
        # Chat results
        with open(self.chat_results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'iteration', 'success', 'latency_ms', 
                'http_status', 'model', 'prompt_tokens', 'completion_tokens',
                'total_tokens', 'finish_reason', 'error_message'
            ])
        
        # Stream results
        with open(self.stream_results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'iteration', 'success', 'latency_ms',
                'http_status', 'model', 'chunks_received', 
                'total_content_length', 'error_message'
            ])
        
        # Health results
        with open(self.health_results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'iteration', 'success', 'status',
                'uptime_seconds', 'process_memory_mb', 'process_memory_percent',
                'system_total_memory_mb', 'system_used_memory_mb', 
                'system_available_memory_mb', 'system_memory_percent',
                'threads', 'open_files', 'num_fds', 'connections',
                'gc_objects', 'model_cache_size', 'error_message'
            ])
    
    def _handle_shutdown(self, signum, frame):
        """Handle graceful shutdown on SIGINT/SIGTERM"""
        print(f"\n\n{'='*60}")
        print("Shutdown signal received. Saving final results...")
        print(f"{'='*60}\n")
        self.should_stop = True
        self._save_summary()
        sys.exit(0)
    
    def _run_curl_command(self, cmd: list, stream: bool = False) -> Dict[str, Any]:
        """Execute curl command and parse response"""
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Parse output
            output = result.stdout
            
            # Extract HTTP status
            http_status = None
            if 'HTTP/2' in output or 'HTTP/1' in output:
                status_line = [line for line in output.split('\n') if 'HTTP/' in line]
                if status_line:
                    http_status = int(status_line[0].split()[1])
            
            # Extract response body (after headers)
            response_body = None
            if '\n\n' in output:
                response_body = output.split('\n\n', 1)[1]
            elif output.strip().startswith('{'):
                response_body = output
            
            return {
                'success': result.returncode == 0 and http_status == 200,
                'latency_ms': latency_ms,
                'http_status': http_status,
                'response_body': response_body,
                'stderr': result.stderr,
                'returncode': result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'latency_ms': (time.time() - start_time) * 1000,
                'http_status': None,
                'response_body': None,
                'stderr': 'Request timeout',
                'returncode': -1
            }
        except Exception as e:
            return {
                'success': False,
                'latency_ms': (time.time() - start_time) * 1000,
                'http_status': None,
                'response_body': None,
                'stderr': str(e),
                'returncode': -1
            }
    
    def test_chat_completion(self) -> Dict[str, Any]:
        """Test regular chat completion"""
        cmd = [
            'curl', '-i', '-k', '-X', 'POST',
            f'{HOST}/v1/chat/completions',
            '-H', 'Content-Type: application/json',
            '-d', json.dumps({
                'model': GOOGLE_MODEL,
                'messages': [{'role': 'user', 'content': PROMPT}],
                'temperature': TEMPERATURE,
                'max_tokens': MAX_TOKENS
            })
        ]
        
        result = self._run_curl_command(cmd)
        
        # Parse JSON response
        parsed_data = {
            'prompt_tokens': None,
            'completion_tokens': None,
            'total_tokens': None,
            'finish_reason': None
        }
        
        if result['response_body']:
            try:
                json_data = json.loads(result['response_body'])
                if 'usage' in json_data:
                    parsed_data['prompt_tokens'] = json_data['usage'].get('prompt_tokens')
                    parsed_data['completion_tokens'] = json_data['usage'].get('completion_tokens')
                    parsed_data['total_tokens'] = json_data['usage'].get('total_tokens')
                parsed_data['finish_reason'] = json_data.get('finish_reason')
            except json.JSONDecodeError:
                pass
        
        return {**result, **parsed_data}
    
    def test_stream_completion(self) -> Dict[str, Any]:
        """Test streaming completion"""
        cmd = [
            'curl', '-i', '-X', 'POST',
            f'{HOST}/v1/chat/completions/stream',
            '-H', 'Content-Type: application/json',
            '-N', '--insecure',
            '-d', json.dumps({
                'model': GOOGLE_MODEL,
                'messages': [{'role': 'user', 'content': PROMPT}],
                'temperature': TEMPERATURE,
                'max_tokens': MAX_TOKENS
            })
        ]
        
        result = self._run_curl_command(cmd, stream=True)
        
        # Parse streaming response
        chunks_received = 0
        total_content_length = 0
        
        if result['response_body']:
            # Count SSE chunks (lines starting with "data:")
            for line in result['response_body'].split('\n'):
                if line.startswith('data:'):
                    chunks_received += 1
                    total_content_length += len(line)
        
        return {
            **result,
            'chunks_received': chunks_received,
            'total_content_length': total_content_length
        }
    
    def check_health(self) -> Dict[str, Any]:
        """Check service health"""
        cmd = [
            'curl', '-i', '-k',
            f'{HOST}/health'
        ]
        
        result = self._run_curl_command(cmd)
        
        # Parse health data
        health_data = {}
        if result['response_body']:
            try:
                health_data = json.loads(result['response_body'])
            except json.JSONDecodeError:
                pass
        
        return {**result, 'health_data': health_data}
    
    def record_chat_result(self, iteration: int, result: Dict[str, Any]):
        """Record chat completion result to CSV"""
        timestamp = datetime.now().isoformat()
        
        with open(self.chat_results_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                iteration,
                result['success'],
                f"{result['latency_ms']:.2f}",
                result['http_status'],
                GOOGLE_MODEL,
                result.get('prompt_tokens', ''),
                result.get('completion_tokens', ''),
                result.get('total_tokens', ''),
                result.get('finish_reason', ''),
                result['stderr'] if not result['success'] else ''
            ])
    
    def record_stream_result(self, iteration: int, result: Dict[str, Any]):
        """Record stream completion result to CSV"""
        timestamp = datetime.now().isoformat()
        
        with open(self.stream_results_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                iteration,
                result['success'],
                f"{result['latency_ms']:.2f}",
                result['http_status'],
                GOOGLE_MODEL,
                result.get('chunks_received', ''),
                result.get('total_content_length', ''),
                result['stderr'] if not result['success'] else ''
            ])
    
    def record_health_result(self, iteration: int, result: Dict[str, Any]):
        """Record health check result to CSV"""
        timestamp = datetime.now().isoformat()
        health_data = result.get('health_data', {})
        
        with open(self.health_results_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                iteration,
                result['success'],
                health_data.get('status', ''),
                health_data.get('uptime_seconds', ''),
                health_data.get('process_memory_mb', ''),
                health_data.get('process_memory_percent', ''),
                health_data.get('system_total_memory_mb', ''),
                health_data.get('system_used_memory_mb', ''),
                health_data.get('system_available_memory_mb', ''),
                health_data.get('system_memory_percent', ''),
                health_data.get('threads', ''),
                health_data.get('open_files', ''),
                health_data.get('num_fds', ''),
                health_data.get('connections', ''),
                health_data.get('gc_objects', ''),
                health_data.get('model_cache_size', ''),
                result['stderr'] if not result['success'] else ''
            ])
    
    def run_iteration(self, iteration: int):
        """Run a single monitoring iteration"""
        print(f"\n{'='*60}")
        print(f"Iteration {iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        # Health check
        print("\n[1/3] Checking health...")
        health_result = self.check_health()
        self.record_health_result(iteration, health_result)
        
        if health_result['success']:
            health_data = health_result['health_data']
            print(f"  ✓ Status: {health_data.get('status', 'unknown')}")
            print(f"  ✓ Process Memory: {health_data.get('process_memory_mb', 0):.2f} MB "
                  f"({health_data.get('process_memory_percent', 0):.2f}%)")
            print(f"  ✓ System Memory: {health_data.get('system_used_memory_mb', 0):.2f} MB "
                  f"({health_data.get('system_memory_percent', 0):.2f}%)")
            print(f"  ✓ Uptime: {health_data.get('uptime_seconds', 0):.2f}s")
        else:
            print(f"  ✗ Health check failed: {health_result['stderr']}")
            self.errors.append({
                'iteration': iteration,
                'type': 'health_check',
                'error': health_result['stderr']
            })
        
        # Chat completion test
        print("\n[2/3] Testing chat completion...")
        chat_result = self.test_chat_completion()
        self.record_chat_result(iteration, chat_result)
        
        if chat_result['success']:
            print(f"  ✓ Latency: {chat_result['latency_ms']:.2f}ms")
            print(f"  ✓ Tokens: {chat_result.get('total_tokens', 'N/A')}")
        else:
            print(f"  ✗ Chat request failed: {chat_result['stderr']}")
            self.errors.append({
                'iteration': iteration,
                'type': 'chat_completion',
                'error': chat_result['stderr']
            })
        
        # Stream completion test
        print("\n[3/3] Testing stream completion...")
        stream_result = self.test_stream_completion()
        self.record_stream_result(iteration, stream_result)
        
        if stream_result['success']:
            print(f"  ✓ Latency: {stream_result['latency_ms']:.2f}ms")
            print(f"  ✓ Chunks: {stream_result.get('chunks_received', 0)}")
        else:
            print(f"  ✗ Stream request failed: {stream_result['stderr']}")
            self.errors.append({
                'iteration': iteration,
                'type': 'stream_completion',
                'error': stream_result['stderr']
            })
        
        print(f"\n{'='*60}")
    
    def _save_summary(self):
        """Save session summary to JSON"""
        summary = {
            'session_timestamp': self.session_timestamp,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_hours_planned': self.duration_hours,
            'duration_hours_actual': (
                (self.end_time - self.start_time).total_seconds() / 3600 
                if self.start_time and self.end_time else None
            ),
            'interval_seconds': self.interval_seconds,
            'iterations_completed': self.iteration_count,
            'errors': self.errors,
            'output_files': {
                'chat_results': str(self.chat_results_file),
                'stream_results': str(self.stream_results_file),
                'health_results': str(self.health_results_file)
            }
        }
        
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Summary saved to: {self.summary_file}")
    
    def run(self):
        """Run the monitoring session"""
        self.start_time = datetime.now()
        end_target = self.start_time + timedelta(hours=self.duration_hours)
        
        print(f"\n{'#'*60}")
        print(f"# TEE LLM Router Monitoring Session")
        print(f"{'#'*60}")
        print(f"\nStart time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration: {self.duration_hours} hours")
        print(f"Interval: {self.interval_seconds} seconds")
        print(f"Expected end: {end_target.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nResults directory: {self.output_dir.absolute()}")
        print(f"  - Chat results: {self.chat_results_file.name}")
        print(f"  - Stream results: {self.stream_results_file.name}")
        print(f"  - Health results: {self.health_results_file.name}")
        print(f"  - Summary: {self.summary_file.name}")
        print(f"\n{'#'*60}\n")
        
        iteration = 0
        
        try:
            while not self.should_stop:
                current_time = datetime.now()
                
                # Check if we've reached the duration
                if current_time >= end_target:
                    print(f"\n\nTarget duration of {self.duration_hours} hours reached.")
                    break
                
                iteration += 1
                self.iteration_count = iteration
                
                # Run tests
                self.run_iteration(iteration)
                
                # Calculate time until next iteration
                time_remaining = (end_target - datetime.now()).total_seconds()
                if time_remaining <= 0:
                    break
                
                sleep_time = min(self.interval_seconds, time_remaining)
                
                # Show progress
                elapsed = (datetime.now() - self.start_time).total_seconds()
                progress_pct = (elapsed / (self.duration_hours * 3600)) * 100
                
                print(f"\nProgress: {progress_pct:.1f}% complete")
                print(f"Next iteration in {sleep_time:.0f} seconds...")
                print(f"Press Ctrl+C to stop early and save results")
                
                time.sleep(sleep_time)
        
        except Exception as e:
            print(f"\n\nUnexpected error: {e}")
            self.errors.append({
                'iteration': iteration,
                'type': 'system_error',
                'error': str(e)
            })
        
        finally:
            self.end_time = datetime.now()
            
            # Final summary
            print(f"\n\n{'#'*60}")
            print(f"# Monitoring Session Complete")
            print(f"{'#'*60}")
            print(f"\nStart time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"End time: {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Duration: {(self.end_time - self.start_time).total_seconds() / 3600:.2f} hours")
            print(f"Iterations: {self.iteration_count}")
            print(f"Errors: {len(self.errors)}")
            
            self._save_summary()
            
            print(f"\n{'#'*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Monitor TEE LLM Router service health and performance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run for 6 hours with 60 second intervals (default)
  python3 monitor_tee_service.py
  
  # Run for 2 hours with 30 second intervals
  python3 monitor_tee_service.py --duration 2 --interval 30
  
  # Short test run (5 minutes, 10 second intervals)
  python3 monitor_tee_service.py --duration 0.083 --interval 10
        """
    )
    
    parser.add_argument(
        '--duration',
        type=float,
        default=6.0,
        help='Duration to run monitoring in hours (default: 6.0)'
    )
    
    parser.add_argument(
        '--interval',
        type=int,
        default=60,
        help='Interval between checks in seconds (default: 60)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./monitoring_results',
        help='Directory to store results (default: ./monitoring_results)'
    )
    
    args = parser.parse_args()
    
    # Create and run monitoring session
    session = MonitoringSession(
        duration_hours=args.duration,
        interval_seconds=args.interval,
        output_dir=args.output_dir
    )
    
    session.run()


if __name__ == '__main__':
    main()
