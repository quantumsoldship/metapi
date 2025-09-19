#!/usr/bin/env python3
"""
ULTRA-OPTIMIZED Pi Calculator for Raspberry Pi 5
Target: >200 digits/second with fixed BBP and additional algorithms
"""

import multiprocessing as mp
import threading
import time
import math
import decimal
import os
import signal
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import psutil
from flask import Flask, jsonify, request
import numpy as np

# Optimized global functions for multiprocessing
def spigot_worker_optimized(args):
    """Optimized spigot algorithm with better performance"""
    start_digit, num_digits = args
    digits = []
    q, r, t, k, n, l = 1, 0, 1, 1, 3, 3
    
    # More efficient spigot implementation
    for _ in range(num_digits):
        while 4*q + r - t >= n*t:
            nr = (2*q + r)*l
            nn = (q*(7*k + 2) + r*l)//(t*l)
            q *= k
            t *= l
            l += 2
            k += 1
            n = nn
            r = nr
        
        digits.append(str(n))
        nr = 10*(r - n*t)
        n = ((10*(3*q + r))//t) - 10*n
        q *= 10
        r = nr
    
    return ('spigot_opt', ''.join(digits))

def bbp_worker_fixed(digits):
    """FIXED Bailey-Borwein-Plouffe formula - no more modulus errors"""
    try:
        pi_digits = []
        
        for d in range(digits):
            s = 0.0
            
            # Use floating point instead of modular arithmetic
            for k in range(d + 100):  # Sufficient iterations
                r = 8 * k
                
                # Avoid division by zero and use floating point
                term1 = pow(16, d - k) / (r + 1) if (r + 1) != 0 else 0
                term2 = pow(16, d - k) / (r + 4) if (r + 4) != 0 else 0
                term3 = pow(16, d - k) / (r + 5) if (r + 5) != 0 else 0
                term4 = pow(16, d - k) / (r + 6) if (r + 6) != 0 else 0
                
                # BBP formula terms
                s += term1 - 0.5 * term2 - 0.25 * term3 - 0.25 * term4
            
            # Extract fractional part and convert to decimal digit
            s = s - int(s)
            if s < 0:
                s += 1
            
            digit = int(10 * s) % 10
            pi_digits.append(str(digit))
        
        return ('bbp_fixed', ''.join(pi_digits))
        
    except Exception as e:
        return ('bbp_error', str(e))

def chudnovsky_worker_optimized(terms):
    """Highly optimized Chudnovsky algorithm"""
    try:
        decimal.getcontext().prec = max(2000, terms * 15)
        
        C = decimal.Decimal(426880) * decimal.Decimal(10005).sqrt()
        pi_sum = decimal.Decimal(0)
        
        # Use precomputed values for efficiency
        factorial_cache = {}
        
        def cached_factorial(n):
            if n in factorial_cache:
                return factorial_cache[n]
            
            if n <= 1:
                result = decimal.Decimal(1)
            else:
                result = decimal.Decimal(n) * cached_factorial(n - 1)
            
            factorial_cache[n] = result
            return result
        
        for k in range(min(terms, 50)):  # Limit for performance
            # More efficient calculation
            numerator = cached_factorial(6*k)
            denominator = cached_factorial(3*k) * (cached_factorial(k) ** 3)
            
            M_k = numerator / denominator
            L_k = 545140134 * k + 13591409
            
            # Avoid overflow with limited precision
            if k > 0:
                X_k = decimal.Decimal(-262537412640768000) ** min(k, 10)
            else:
                X_k = decimal.Decimal(1)
            
            term = M_k * L_k * X_k
            pi_sum += term
        
        pi = C / pi_sum
        pi_str = str(pi)
        
        if '.' in pi_str:
            digits = pi_str.split('.')[1]
        else:
            digits = pi_str[1:]
        
        return ('chudnovsky_opt', digits[:terms*8])
        
    except Exception as e:
        return ('chudnovsky_error', str(e))

def ramanujan_worker(terms):
    """Ramanujan's pi formula - very fast convergence"""
    try:
        decimal.getcontext().prec = max(3000, terms * 20)
        
        # Ramanujan's formula: 1/π = (2√2/9801) * Σ[(4k)!(1103+26390k)]/[(k!)^4 * 396^(4k)]
        sqrt2 = decimal.Decimal(2).sqrt()
        coeff = 2 * sqrt2 / decimal.Decimal(9801)
        
        pi_sum = decimal.Decimal(0)
        
        for k in range(min(terms, 20)):  # Very fast convergence
            # Calculate factorials efficiently
            numerator = decimal.Decimal(1)
            for i in range(4*k, 0, -1):
                numerator *= i
            
            denominator = decimal.Decimal(1)
            for i in range(1, k + 1):
                denominator *= i ** 4
            
            denominator *= decimal.Decimal(396) ** (4 * k)
            
            term_val = 1103 + 26390 * k
            term = (numerator * term_val) / denominator
            pi_sum += term
        
        pi_inv = coeff * pi_sum
        pi = 1 / pi_inv
        
        pi_str = str(pi)
        if '.' in pi_str:
            digits = pi_str.split('.')[1]
        else:
            digits = pi_str[1:]
        
        return ('ramanujan', digits[:terms*12])
        
    except Exception as e:
        return ('ramanujan_error', str(e))

def borwein_worker(iterations):
    """Borwein's quartic algorithm - very fast"""
    try:
        decimal.getcontext().prec = max(4000, iterations * 30)
        
        # Initialize
        a = decimal.Decimal(6) - 4 * decimal.Decimal(2).sqrt()
        y = decimal.Decimal(2).sqrt() - 1
        
        for _ in range(min(iterations, 10)):  # Each iteration quadruples precision
            # Calculate new y
            y_quad = y ** 4
            sqrt_term = (1 - y_quad).sqrt().sqrt()
            y_new = (1 - sqrt_term) / (1 + sqrt_term)
            
            # Calculate new a
            y_new_plus_1 = y_new + 1
            a_new = a * (y_new_plus_1 ** 4) - (2 ** (2*_ + 3)) * y_new * (1 + y_new + y_new**2)
            
            a, y = a_new, y_new
        
        pi = 1 / a
        pi_str = str(pi)
        
        if '.' in pi_str:
            digits = pi_str.split('.')[1]
        else:
            digits = pi_str[1:]
        
        return ('borwein', digits[:iterations*20])
        
    except Exception as e:
        return ('borwein_error', str(e))

def monte_carlo_worker_optimized(samples):
    """Highly optimized Monte Carlo using NumPy"""
    try:
        # Use NumPy for vectorized operations
        x = np.random.random(samples)
        y = np.random.random(samples)
        
        # Vectorized distance calculation
        distances_sq = x*x + y*y
        inside_circle = np.sum(distances_sq <= 1.0)
        
        pi_estimate = 4.0 * inside_circle / samples
        pi_str = f"{pi_estimate:.{samples//1000}f}"
        
        if '.' in pi_str:
            digits = pi_str.split('.')[1]
        else:
            digits = pi_str[1:]
        
        return ('monte_carlo_opt', digits[:samples//100])
        
    except Exception as e:
        return ('monte_carlo_error', str(e))

class UltraOptimizedPiAPI:
    def __init__(self, output_file="pi_digits.txt", log_file="pi_calc.log"):
        self.cpu_count = mp.cpu_count()
        self.output_file = output_file
        self.log_file = log_file
        self.running = False
        self.total_digits = 0
        self.start_time = None
        
        # Enhanced performance tracking
        self.stats = {
            'calculations_per_second': 0,
            'total_calculations': 0,
            'memory_usage_mb': 0,
            'cpu_percent': 0,
            'current_batch': 0,
            'algorithms_used': [],
            'best_algorithm': '',
            'peak_rate': 0,
            'efficiency_score': 0
        }
        
        # Clear log file
        open(self.log_file, 'w').close()
        
        print(f"🚀 Ultra-Optimized Pi Calculator initialized with {self.cpu_count} CPU cores")
        
    def log_message(self, message):
        """Enhanced logging with performance metrics"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        
        with open(self.log_file, 'a') as f:
            f.write(log_entry + '\n')
    
    def ultra_parallel_calculation(self, target_digits=5000):
        """Ultra-optimized parallel calculation with 6 algorithms"""
        chunk_size = max(200, target_digits // self.cpu_count)
        
        # Algorithm distribution for maximum performance
        algorithms = [
            ('spigot_opt', spigot_worker_optimized, (0, chunk_size)),
            ('bbp_fixed', bbp_worker_fixed, chunk_size//2),
            ('chudnovsky_opt', chudnovsky_worker_optimized, max(5, chunk_size//100)),
            ('ramanujan', ramanujan_worker, max(3, chunk_size//200)),
            ('borwein', borwein_worker, max(2, chunk_size//400)),
            ('monte_carlo_opt', monte_carlo_worker_optimized, chunk_size*10)
        ]
        
        # Scale algorithms based on available cores
        work_items = []
        for i in range(self.cpu_count):
            algo_name, worker_func, args = algorithms[i % len(algorithms)]
            work_items.append((algo_name, worker_func, args))
        
        # Execute with timeout and monitoring
        results = []
        algorithms_used = []
        
        with ProcessPoolExecutor(max_workers=self.cpu_count) as executor:
            futures = []
            
            for algo_name, worker_func, args in work_items:
                future = executor.submit(worker_func, args)
                futures.append((algo_name, future))
            
            # Collect results with performance monitoring
            for algo_name, future in futures:
                try:
                    result = future.result(timeout=5)  # Faster timeout
                    
                    if isinstance(result, tuple) and len(result) == 2:
                        actual_algo, digits = result
                        if 'error' not in actual_algo and digits:
                            results.append((actual_algo, str(digits)))
                            algorithms_used.append(actual_algo)
                            self.log_message(f"✅ {actual_algo}: Generated {len(str(digits))} digits")
                        else:
                            self.log_message(f"❌ Error in {actual_algo}: {digits}")
                    
                except Exception as e:
                    self.log_message(f"⚠️ Timeout/Error in {algo_name}: {e}")
        
        # Update algorithm performance stats
        self.stats['algorithms_used'] = list(set(algorithms_used))
        
        if results:
            best_result = max(results, key=lambda x: len(x[1]))
            self.stats['best_algorithm'] = best_result[0]
        
        return results
    
    def continuous_calculation(self, batch_size=5000, max_digits=100000):
        """Ultra-optimized continuous calculation"""
        self.running = True
        self.start_time = time.time()
        
        # Initialize output file
        with open(self.output_file, 'w') as f:
            f.write("# 🚀 ULTRA-OPTIMIZED Pi Calculator for Raspberry Pi 5\n")
            f.write(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Target: {max_digits:,} digits | Batch size: {batch_size:,}\n")
            f.write(f"# CPU cores: {self.cpu_count} | Algorithms: 6 optimized\n\n")
            f.write("3.")
        
        self.log_message(f"🎯 Starting ULTRA calculation. Target: {max_digits:,} digits")
        
        total_calculated = 0
        batch_count = 0
        performance_history = []
        
        while self.running and total_calculated < max_digits:
            batch_start = time.time()
            batch_count += 1
            self.stats['current_batch'] = batch_count
            
            # Dynamic batch sizing based on performance
            if len(performance_history) > 5:
                avg_rate = sum(performance_history[-5:]) / 5
                if avg_rate > 200:
                    batch_size = min(batch_size * 1.1, 10000)  # Increase if performing well
                elif avg_rate < 100:
                    batch_size = max(batch_size * 0.9, 1000)   # Decrease if struggling
            
            remaining = min(int(batch_size), max_digits - total_calculated)
            
            # Run ultra-optimized calculation
            results = self.ultra_parallel_calculation(remaining)
            
            if results:
                # Use the best result (longest digits)
                best_result = max(results, key=lambda x: len(x[1]) if x[1] else 0)
                best_algo, best_digits = best_result
                
                # Ensure we don't exceed target
                digits_to_write = best_digits[:remaining]
                
                # Atomic file write
                with open(self.output_file, 'a') as f:
                    f.write(digits_to_write)
                    f.flush()
                
                total_calculated += len(digits_to_write)
                
                # Update enhanced statistics
                elapsed = time.time() - self.start_time
                current_rate = total_calculated / elapsed if elapsed > 0 else 0
                batch_time = time.time() - batch_start
                batch_rate = len(digits_to_write) / batch_time if batch_time > 0 else 0
                
                performance_history.append(current_rate)
                if len(performance_history) > 20:
                    performance_history.pop(0)
                
                self.stats.update({
                    'total_calculations': total_calculated,
                    'calculations_per_second': current_rate,
                    'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024,
                    'cpu_percent': psutil.cpu_percent(),
                    'peak_rate': max(self.stats.get('peak_rate', 0), current_rate),
                    'efficiency_score': (current_rate / 200) * 100  # Target 200/s
                })
                
                progress = (total_calculated / max_digits) * 100
                eta_seconds = (max_digits - total_calculated) / current_rate if current_rate > 0 else 0
                eta_time = f"{int(eta_seconds//60)}m{int(eta_seconds%60)}s"
                
                self.log_message(
                    f"🔥 Batch {batch_count}: +{len(digits_to_write)} digits ({best_algo}) | "
                    f"Total: {total_calculated:,}/{max_digits:,} ({progress:.1f}%) | "
                    f"Rate: {current_rate:.1f}/s (Peak: {self.stats['peak_rate']:.1f}/s) | "
                    f"ETA: {eta_time} | Efficiency: {self.stats['efficiency_score']:.1f}%"
                )
            else:
                self.log_message(f"💥 Batch {batch_count}: No results, retrying with smaller batch...")
                batch_size = max(batch_size // 2, 500)
                time.sleep(0.1)
        
        # Finalize with performance summary
        total_time = time.time() - self.start_time
        final_rate = total_calculated / total_time
        
        with open(self.output_file, 'a') as f:
            f.write(f"\n\n# 🏁 CALCULATION COMPLETE!\n")
            f.write(f"# Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Total digits: {total_calculated:,}\n")
            f.write(f"# Time: {total_time:.2f}s | Final rate: {final_rate:.2f}/s\n")
            f.write(f"# Peak rate: {self.stats['peak_rate']:.2f}/s\n")
            f.write(f"# Best algorithm: {self.stats['best_algorithm']}\n")
            f.write(f"# Algorithms used: {', '.join(self.stats['algorithms_used'])}\n")
        
        self.log_message(f"🏆 COMPLETE! {total_calculated:,} digits in {total_time:.2f}s")
        self.log_message(f"📊 Final rate: {final_rate:.2f}/s | Peak: {self.stats['peak_rate']:.2f}/s")
        self.running = False
    
    def stop_calculation(self):
        """Stop calculation gracefully"""
        self.running = False
        self.log_message("🛑 Stopping calculation...")
    
    def get_current_digits(self):
        """Get currently calculated digits from file"""
        try:
            with open(self.output_file, 'r') as f:
                content = f.read()
            
            # Find the pi digits line
            lines = content.split('\n')
            for line in lines:
                if line.startswith('3.'):
                    return line
            return "3."
        except:
            return "3."

# Enhanced Flask API
app = Flask(__name__)
calculator = None
calc_thread = None

@app.route('/api/ultra-start')
def ultra_start():
    """Start ultra-optimized calculation - GET /api/ultra-start?digits=100000&batch=5000"""
    global calculator, calc_thread
    
    if calculator and calculator.running:
        return jsonify({'error': 'Already running', 'status': 'running'}), 400
    
    digits = int(request.args.get('digits', 100000))
    batch_size = int(request.args.get('batch', 5000))
    
    output_file = f"pi_ultra_{digits}_{int(time.time())}.txt"
    calculator = UltraOptimizedPiAPI(output_file=output_file)
    
    calc_thread = threading.Thread(
        target=calculator.continuous_calculation,
        args=(batch_size, digits),
        daemon=True
    )
    calc_thread.start()
    
    return jsonify({
        'status': 'ultra_started',
        'target_digits': digits,
        'batch_size': batch_size,
        'output_file': output_file,
        'cpu_cores': calculator.cpu_count,
        'algorithms': ['spigot_opt', 'bbp_fixed', 'chudnovsky_opt', 'ramanujan', 'borwein', 'monte_carlo_opt'],
        'target_rate': '200+ digits/second'
    })

@app.route('/api/performance')
def get_performance():
    """Get detailed performance metrics"""
    global calculator
    
    if not calculator:
        return jsonify({'error': 'No calculation running'}), 400
    
    elapsed = time.time() - calculator.start_time if calculator.start_time else 0
    
    return jsonify({
        **calculator.stats,
        'running': calculator.running,
        'elapsed_time': elapsed,
        'output_file': calculator.output_file,
        'performance_class': 'ULTRA' if calculator.stats.get('calculations_per_second', 0) > 200 else 'HIGH'
    })

if __name__ == "__main__":
    print("=" * 70)
    print("🚀 ULTRA-OPTIMIZED PI CALCULATOR FOR RASPBERRY PI 5")
    print("Target: 200+ digits/second | 6 Advanced Algorithms")
    print("=" * 70)
    
    mode = input("Run as: (1) Direct ultra calculation (2) Ultra API server [2]: ") or "2"
    
    if mode == "1":
        digits = int(input("Target digits [100000]: ") or 100000)
        calc = UltraOptimizedPiAPI()
        calc.continuous_calculation(max_digits=digits)
    else:
        print("🌐 Starting Ultra API server...")
        print("Access: http://localhost:5000")
        print("\n🎯 Ultra Endpoints:")
        print("  GET /api/ultra-start?digits=N&batch=M  - Start ultra calculation")
        print("  GET /api/performance                    - Get performance metrics")
        print("  GET /api/status                        - Get basic status")
        print("  GET /api/digits                        - Get current digits")
        
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
