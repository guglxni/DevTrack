# ASD Assessment API - Enhanced Version

This package includes several significant enhancements to the ASD Assessment API, focusing on improving the scoring accuracy, optimizing for Apple Silicon, and providing more intuitive testing tools.

## 🆕 What's New

1. **Enhanced Scoring Accuracy**
   - Improved response interpretation for more accurate milestone scoring
   - Better handling of negative responses and negations
   - More nuanced understanding of caregiver feedback
   - Properly distinguishes between "not yet," "emerging," "with support," and "independent" responses

2. **Apple Silicon Optimization**
   - Automatically detects and leverages Apple M-series chips
   - Uses Metal Performance Shaders (MPS) when available
   - Optimized thread management for better performance
   - Reduced memory usage and faster processing

3. **Improved Developer Experience**
   - New scripts for easier testing and deployment
   - Better error handling and debug information
   - Clearer feedback on system status

## 📋 Scripts

### `enhance_model.py`

This script improves the scoring model by enhancing the language understanding capabilities of the assessment engine:

```bash
# Run directly to enhance the model
./enhance_model.py
```

The script:
- Creates a backup of your original assessment engine
- Adds a sophisticated response interpretation function
- Ensures compatibility with the existing API

### `start_api.sh`

An optimized launcher for the API with better configuration options:

```bash
# Basic usage
./start_api.sh

# With options
./start_api.sh --port 8000 --workers 2 --reload
```

Features:
- Automatic Python environment detection
- Apple Silicon optimization
- Dependency checking
- Configurable workers, port, timeout, etc.
- Better error handling and status reporting

### `test_enhanced.sh`

A dedicated testing tool to evaluate the enhanced scoring model:

```bash
# Start the API server
./test_enhanced.sh start

# Test various responses
./test_enhanced.sh test "yes, he does this independently"
./test_enhanced.sh test "no, he cannot do this yet"
./test_enhanced.sh test "sometimes but needs help"

# Change the milestone being tested
./test_enhanced.sh milestone "draws a person with 2-4 body parts"

# Stop the server
./test_enhanced.sh stop
```

## 🚀 Getting Started

1. **Enhance the model**:
   ```bash
   ./enhance_model.py
   ```

2. **Start the API server**:
   ```bash
   ./start_api.sh
   ```
   or
   ```bash
   ./test_enhanced.sh start
   ```

3. **Test the enhanced scoring**:
   ```bash
   ./test_enhanced.sh test "your response here"
   ```

## 💡 Understanding the Scoring

The enhanced scoring system provides more accurate interpretations:

| Score | Label | Description |
|-------|-------|-------------|
| 4 | INDEPENDENT | Child performs the skill independently without help |
| 3 | WITH_SUPPORT | Child can perform with assistance or prompting |
| 2 | EMERGING | Child is beginning to develop this skill, but inconsistently |
| 1 | NOT_YET | Child is not yet demonstrating this skill |
| 0 | CANNOT_DO | Child is definitely unable to perform this skill |

## 🛠️ Technical Details

### Enhanced Response Processing

The new scoring model:
- Uses regular expressions to detect patterns in caregiver responses
- Applies linguistic analysis to understand negations and qualifiers
- Handles ambiguous or uncertain responses more appropriately
- Falls back to the original scoring method when needed for compatibility

### Apple Silicon Optimizations

When running on Apple Silicon:
- Enables Metal Performance Shaders (MPS) for faster tensor operations
- Sets optimal thread count based on available cores
- Enables memory optimizations specific to the M-series architecture
- Configures PyTorch for optimal performance

## 🔍 Troubleshooting

If you encounter issues:

1. Check if the API server is running:
   ```bash
   ./test_enhanced.sh start
   ```

2. Verify the model was enhanced properly:
   ```bash
   ./enhance_model.py
   ```

3. If problems persist, restart the server with debug logging:
   ```bash
   ./start_api.sh --log-level debug
   ```

## 📊 Performance Improvements

The enhanced version provides:
- More consistent and accurate scoring
- Faster response times on Apple Silicon
- Better handling of edge cases
- Improved resource utilization 