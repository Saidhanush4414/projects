#!/usr/bin/env python3
"""
Simple runner script for the Crypto Portfolio AI Dashboard
"""

import sys
import os
import subprocess

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'flask', 'pandas', 'numpy', 'scikit-learn', 
        'matplotlib', 'textblob', 'requests'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Installing missing packages...")
        
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
            print("✅ All packages installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install packages. Please run:")
            print(f"   pip install {' '.join(missing_packages)}")
            return False
    
    return True

def main():
    """Main function to run the application"""
    print("🚀 Crypto Portfolio AI Dashboard")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check if app.py exists
    if not os.path.exists('app.py'):
        print("❌ app.py not found in current directory")
        print("   Please run this script from the project root directory")
        sys.exit(1)
    
    print("✅ All dependencies satisfied")
    print("🌐 Starting web server...")
    print("\n📊 Available pages:")
    print("   - http://localhost:5000/ (Dashboard)")
    print("   - http://localhost:5000/coins (Crypto Analytics)")
    print("   - http://localhost:5000/news (AI News Monitoring)")
    print("   - http://localhost:5000/whale (Whale Detection & Trading)")
    print("\n🛑 Press Ctrl+C to stop the server")
    print("=" * 40)
    
    # Run the Flask app
    try:
        import app
        app.app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Server stopped. Goodbye!")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
