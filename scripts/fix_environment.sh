#!/bin/bash
#
# Stock Market Forecasting — Complete Environment Repair Script
# Run this on macOS to create a clean, working Python environment
#
# Usage:
#   chmod +x scripts/fix_environment.sh
#   ./scripts/fix_environment.sh
#

set -e

# Resolve the repo root as the parent of this script's directory (scripts/..)
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Stock Market Forecasting — Environment Repair Script     ║"
echo "║  macOS + Apple Silicon / Intel Compatible                 ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Verify current directory
echo "Step 1/7: Verifying project directory..."
if [ ! -d "$PROJECT_ROOT" ]; then
    echo "❌ Project directory not found: $PROJECT_ROOT"
    exit 1
fi
cd "$PROJECT_ROOT"
echo "  ✓ Working directory: $(pwd)"
echo ""

# Step 2: Backup old environments
echo "Step 2/7: Backing up old environments..."
if [ -d "venv" ]; then
    BACKUP_TIME=$(date +%s)
    mv venv "venv_broken_$BACKUP_TIME"
    echo "  ✓ Old venv backed up: venv_broken_$BACKUP_TIME"
fi
if [ -d ".venv" ]; then
    BACKUP_TIME=$(date +%s)
    mv .venv "venv_old_$BACKUP_TIME"
    echo "  ✓ Old .venv backed up: venv_old_$BACKUP_TIME"
fi
echo ""

# Step 3: Ensure Python 3.11+ is available
echo "Step 3/7: Ensuring Python 3.11+ is available..."
PYTHON_CMD=""
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
    echo "  ✓ Using python3.11"
elif command -v python3.12 &> /dev/null; then
    PYTHON_CMD="python3.12"
    echo "  ✓ Using python3.12"
elif command -v python3 &> /dev/null; then
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    if [[ $PYTHON_VERSION == 3.1[0-4]* ]]; then
        PYTHON_CMD="python3"
        echo "  ✓ Using system python3 ($PYTHON_VERSION)"
    else
        echo "⚠ System python3 is $PYTHON_VERSION (need 3.10+)"
        echo "  Installing Python 3.11 via Homebrew..."
        brew install python@3.11 2>/dev/null
        PYTHON_CMD="python3.11"
    fi
fi

if [ -z "$PYTHON_CMD" ]; then
    echo "❌ Could not find Python 3.10+"
    exit 1
fi

$PYTHON_CMD --version
echo ""

# Step 4: Create fresh virtual environment
echo "Step 4/7: Creating fresh virtual environment..."
$PYTHON_CMD -m venv venv
source venv/bin/activate
python --version
echo "  ✓ Virtual environment created and activated"
echo ""

# Step 5: Upgrade pip and build tools
echo "Step 5/7: Upgrading pip, setuptools, wheel..."
pip install --quiet --upgrade pip setuptools wheel
pip --version
echo "  ✓ pip upgraded"
echo ""

# Step 6: Install all dependencies
echo "Step 6/7: Installing all dependencies (this may take 5-10 minutes)..."
echo "  Installing: numpy, pandas, scipy, scikit-learn, statsmodels..."
pip install --quiet --no-cache-dir \
    "numpy>=1.24.0,<3.0" \
    "pandas>=2.0.0,<3.0" \
    "scipy>=1.10.0" \
    "scikit-learn>=1.3.0,<2.0" \
    "statsmodels>=0.14.0" \
    "joblib>=1.3.0"
echo "  ✓ Core scientific stack installed"

echo "  Installing: PyTorch with MPS support..."
pip install --quiet --no-cache-dir \
    "torch>=2.2.0" \
    "torchvision>=0.17.0" \
    "torchaudio>=2.2.0"
echo "  ✓ PyTorch installed"

echo "  Installing: Financial and data libraries..."
pip install --quiet --no-cache-dir \
    "yfinance>=0.2.40" \
    "openpyxl>=3.1.0"
echo "  ✓ Financial libraries installed"

echo "  Installing: Google Cloud integration..."
pip install --quiet --no-cache-dir \
    "google-auth>=2.29.0" \
    "google-auth-httplib2>=0.2.0" \
    "google-auth-oauthlib>=1.2.0" \
    "gspread>=6.0.0" \
    "cryptography>=42.0.0"
echo "  ✓ Google Cloud integration installed"

echo "  Installing: Web frameworks and API tools..."
pip install --quiet --no-cache-dir \
    "requests>=2.31.0" \
    "urllib3>=2.0.0" \
    "httpx>=0.27.0" \
    "fastapi>=0.110.0" \
    "uvicorn[standard]>=0.27.0" \
    "pydantic>=2.6.0,<3.0" \
    "python-dotenv>=1.0.0" \
    "langchain-core>=0.1.0"
echo "  ✓ Web frameworks installed"

echo "  Installing: Visualization and ML optimization tools..."
pip install --quiet --no-cache-dir \
    "matplotlib>=3.7.0" \
    "seaborn>=0.13.0" \
    "plotly>=5.18.0" \
    "tqdm>=4.66.0" \
    "beautifulsoup4>=4.12.0" \
    "optuna>=3.5.0" \
    "pytest>=8.0.0"
echo "  ✓ Visualization and ML tools installed"

echo ""
echo "Step 7/7: Verifying all imports..."
python << 'VERIFY_IMPORTS'
import sys
test_packages = [
    ('torch', 'PyTorch'),
    ('pandas', 'pandas'),
    ('numpy', 'numpy'),
    ('scipy', 'scipy'),
    ('sklearn', 'scikit-learn'),
    ('statsmodels', 'statsmodels'),
    ('yfinance', 'yfinance'),
    ('openpyxl', 'openpyxl'),
    ('google.auth', 'google-auth'),
    ('gspread', 'gspread'),
    ('fastapi', 'fastapi'),
    ('pydantic', 'pydantic'),
    ('matplotlib', 'matplotlib'),
    ('optuna', 'optuna'),
    ('pytest', 'pytest'),
]

failed = []
for module_name, display_name in test_packages:
    try:
        __import__(module_name)
        print(f"  ✓ {display_name}")
    except ImportError as e:
        print(f"  ✗ {display_name}: {e}")
        failed.append(display_name)

print()
if failed:
    print(f"❌ {len(failed)} packages failed: {', '.join(failed)}")
    sys.exit(1)
else:
    print("✅ All core packages verified!")

# Extra validation
try:
    import torch
    print(f"\n📊 PyTorch version: {torch.__version__}")
    if torch.backends.mps.is_available():
        print("   MPS (Apple Silicon acceleration) is AVAILABLE ✓")
    else:
        print("   MPS not available (using CPU)")
except Exception as e:
    print(f"⚠ PyTorch check failed: {e}")
VERIFY_IMPORTS

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  ✅ Environment Repair Complete!                          ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "📝 Next Steps:"
echo "   1. Keep your venv activated:"
echo "      source venv/bin/activate"
echo ""
echo "   2. Run the complete forecasting pipeline:"
echo "      ./scripts/run_full_workflow.sh"
echo ""
echo "   3. Check the output for any remaining issues"
echo ""
echo "📋 To verify the environment manually:"
echo "   python -c \"import torch; print(f'PyTorch {torch.__version__}')\""
echo "   python -c \"import pandas; print(f'pandas {pandas.__version__}')\""
echo ""
