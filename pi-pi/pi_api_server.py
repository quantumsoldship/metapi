#!/usr/bin/env python3
"""
Simple API server for pi calculation status
Lightweight REST API - no web interface
"""

from flask import Flask, jsonify
import threading
import time
from pi_calculator_api import MaxPowerPiAPI

app = Flask(__name__)
calculator = None
calc_thread = None

@app.route('/api/start/<int:digits>')
def start_calculation(digits):
    """Start pi calculation for specified digits"""
    global calculator, calc_thread
    
    if calculator and calculator.running:
        return jsonify({'error': 'Calculation already running'}), 400
    
    calculator = MaxPowerPiAPI(output_file=f"pi_{digits}_digits.txt")
    calc_thread = threading.Thread(
        target=calculator.continuous_calculation,
        args=(5000, digits),
        daemon=True
    )
    calc_thread.start()
    
    return jsonify({
        'status': 'started',
        'target_digits': digits,
        'output_file': calculator.output_file
    })

@app.route('/api/stop')
def stop_calculation():
    """Stop current calculation"""
    global calculator
    
    if calculator:
        calculator.stop_calculation()
        return jsonify({'status': 'stopped'})
    
    return jsonify({'error': 'No calculation running'}), 400

@app.route('/api/status')
def get_status():
    """Get current calculation status"""
    global calculator
    
    if not calculator:
        return jsonify({'status': 'not_started'})
    
    return jsonify(calculator.get_stats_api())

@app.route('/api/digits')
def get_current_digits():
    """Get currently calculated digits"""
    global calculator
    
    if not calculator:
        return jsonify({'error': 'No calculation running'}), 400
    
    try:
        with open(calculator.output_file, 'r') as f:
            content = f.read()
        
        # Extract just the digits (skip header)
        lines = content.split('\n')
        pi_line = next((line for line in lines if line.startswith('3.')), '3.')
        
        return jsonify({
            'digits': pi_line,
            'length': len(pi_line) - 2,  # Subtract '3.'
            'file': calculator.output_file
        })
    
    except FileNotFoundError:
        return jsonify({'error': 'Output file not found'}), 404

if __name__ == '__main__':
    print("Pi Calculator API Server")
    print("Endpoints:")
    print("  GET  /api/start/<digits>  - Start calculation")
    print("  GET  /api/stop           - Stop calculation")
    print("  GET  /api/status         - Get status")
    print("  GET  /api/digits         - Get current digits")
    print("\nStarting server on port 5000...")
    
    app.run(host='0.0.0.0', port=5000, debug=False)