#!/usr/bin/env python3
"""
Pi Calculation GitHub Uploader
Automatically uploads completed pi calculation files to quantumsoldship/metapi
"""

import os
import requests
import json
import base64
from datetime import datetime
import hashlib

class PiGitHubUploader:
    def __init__(self, token: str, repo: str = "quantumsoldship/metapi"):
        self.token = token
        self.repo = repo
        self.base_url = "https://api.github.com"
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "Content-Type": "application/json"
        }
        
    def upload_file(self, file_path: str, github_path: str, commit_message: str = None):
        """Upload a file to GitHub repository"""
        
        # Read file content
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return False
        
        with open(file_path, 'rb') as f:
            content = f.read()
        
        # Encode content to base64
        content_b64 = base64.b64encode(content).decode('utf-8')
        
        # Generate commit message if not provided
        if not commit_message:
            file_size = len(content)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            commit_message = f"Upload pi calculation results - {file_size:,} bytes - {timestamp}"
        
        # Check if file exists (to get SHA for updates)
        url = f"{self.base_url}/repos/{self.repo}/contents/{github_path}"
        response = requests.get(url, headers=self.headers)
        
        data = {
            "message": commit_message,
            "content": content_b64
        }
        
        if response.status_code == 200:
            # File exists, need SHA for update
            existing_file = response.json()
            data["sha"] = existing_file["sha"]
            print(f"📝 Updating existing file: {github_path}")
        else:
            print(f"📄 Creating new file: {github_path}")
        
        # Upload/update file
        response = requests.put(url, headers=self.headers, data=json.dumps(data))
        
        if response.status_code in [200, 201]:
            result = response.json()
            print(f"✅ Successfully uploaded: {github_path}")
            print(f"🔗 View at: {result['content']['html_url']}")
            return True
        else:
            print(f"❌ Upload failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False
    
    def upload_pi_results(self, pi_file: str, log_file: str = None, stats_file: str = None):
        """Upload pi calculation results with metadata"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        upload_success = []
        
        # Upload main pi digits file
        if os.path.exists(pi_file):
            github_path = f"pi-pi/calculations/pi_digits_{timestamp}.txt"
            
            # Get file stats for commit message
            file_size = os.path.getsize(pi_file)
            
            # Count digits by reading first few lines
            digit_count = 0
            with open(pi_file, 'r') as f:
                for line in f:
                    if line.startswith('3.'):
                        digit_count = len(line.strip()) - 2  # Remove '3.'
                        break
            
            commit_msg = f"🔢 Pi calculation complete: {digit_count:,} digits ({file_size:,} bytes) - Raspberry Pi 5"
            
            if self.upload_file(pi_file, github_path, commit_msg):
                upload_success.append(github_path)
        
        # Upload log file if exists
        if log_file and os.path.exists(log_file):
            github_path = f"pi-pi/logs/calculation_log_{timestamp}.txt"
            commit_msg = f"📝 Pi calculation logs - {timestamp}"
            
            if self.upload_file(log_file, github_path, commit_msg):
                upload_success.append(github_path)
        
        # Create and upload summary stats
        if pi_file and os.path.exists(pi_file):
            stats = self.generate_stats(pi_file, log_file)
            stats_content = self.format_stats(stats, timestamp)
            
            # Write stats to temp file
            temp_stats_file = f"pi_stats_{timestamp}.md"
            with open(temp_stats_file, 'w') as f:
                f.write(stats_content)
            
            github_path = f"pi-pi/stats/calculation_stats_{timestamp}.md"
            commit_msg = f"📊 Pi calculation statistics - {stats['digit_count']:,} digits - {timestamp}"
            
            if self.upload_file(temp_stats_file, github_path, commit_msg):
                upload_success.append(github_path)
            
            # Clean up temp file
            os.remove(temp_stats_file)
        
        # Update main README
        self.update_readme(upload_success, timestamp)
        
        return upload_success
    
    def generate_stats(self, pi_file: str, log_file: str = None):
        """Generate statistics from pi calculation files"""
        stats = {
            "timestamp": datetime.now().isoformat(),
            "file_size": os.path.getsize(pi_file),
            "digit_count": 0,
            "calculation_time": 0,
            "rate": 0,
            "algorithms_used": [],
            "peak_rate": 0,
            "cpu_info": "Raspberry Pi 5"
        }
        
        # Analyze pi digits file
        with open(pi_file, 'r') as f:
            content = f.read()
            
            # Count digits
            for line in content.split('\n'):
                if line.startswith('3.'):
                    stats["digit_count"] = len(line.strip()) - 2
                    break
            
            # Extract metadata from comments
            if "Total digits:" in content:
                import re
                time_match = re.search(r'Calculation time: ([\d.]+) seconds', content)
                if time_match:
                    stats["calculation_time"] = float(time_match.group(1))
                
                rate_match = re.search(r'Average rate: ([\d.]+) digits/second', content)
                if rate_match:
                    stats["rate"] = float(rate_match.group(1))
        
        # Analyze log file for algorithms and peak rate
        if log_file and os.path.exists(log_file):
            with open(log_file, 'r') as f:
                log_content = f.read()
                
                # Extract algorithms used
                algorithms = set()
                peak_rate = 0
                
                for line in log_content.split('\n'):
                    if "Generated" in line:
                        for algo in ["spigot", "chudnovsky", "machin", "bbp", "ramanujan", "borwein", "leibniz", "monte_carlo"]:
                            if algo in line.lower():
                                algorithms.add(algo)
                    
                    if "Rate:" in line:
                        import re
                        rate_match = re.search(r'Rate: ([\d.]+)', line)
                        if rate_match:
                            rate = float(rate_match.group(1))
                            peak_rate = max(peak_rate, rate)
                
                stats["algorithms_used"] = list(algorithms)
                stats["peak_rate"] = peak_rate
        
        return stats
    
    def format_stats(self, stats: dict, timestamp: str):
        """Format statistics as markdown"""
        return f"""# Pi Calculation Results - {timestamp}

## 📊 Summary Statistics

- **Total Digits Calculated**: {stats['digit_count']:,}
- **File Size**: {stats['file_size']:,} bytes
- **Calculation Time**: {stats['calculation_time']:.2f} seconds
- **Average Rate**: {stats['rate']:.2f} digits/second
- **Peak Rate**: {stats['peak_rate']:.2f} digits/second
- **Hardware**: {stats['cpu_info']}

## 🧮 Algorithms Used

{', '.join(stats['algorithms_used']) if stats['algorithms_used'] else 'Multiple algorithms'}

## ⚡ Performance Metrics

- **Efficiency**: {(stats['rate']/200)*100:.1f}% (target: 200 digits/sec)
- **Data Rate**: {(stats['file_size']/1024/1024/stats['calculation_time']):.2f} MB/s
- **Computational Intensity**: {stats['digit_count']/stats['calculation_time']:.0f} operations/sec

## 🕒 Calculation Details

- **Started**: {datetime.fromisoformat(stats['timestamp']).strftime('%Y-%m-%d %H:%M:%S UTC')}
- **Duration**: {timedelta(seconds=int(stats['calculation_time']))}
- **Platform**: Raspberry Pi 5 with maximum optimization

## 📁 Generated Files

- Pi digits: `/pi-pi/calculations/pi_digits_{timestamp}.txt`
- Calculation logs: `/pi-pi/logs/calculation_log_{timestamp}.txt`
- This statistics file: `/pi-pi/stats/calculation_stats_{timestamp}.md`

---
*Generated automatically by Pi Calculator GitHub Uploader*
"""
    
    def update_readme(self, uploaded_files: list, timestamp: str):
        """Update main README with latest calculation info"""
        readme_content = f"""# 🥧 MetAPI - Pi Calculation Project

Calculating pi digits on a Raspberry Pi 5 with maximum performance optimization.

## 🎯 Latest Calculation

**Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}

### 📁 Recent Files
{chr(10).join([f"- `{file}`" for file in uploaded_files])}

## 🚀 Project Structure
