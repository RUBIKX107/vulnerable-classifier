# vulnerable-classifier

Step 1: Environment Setup
First, make sure you have the required libraries:
bashpip install tensorflow numpy matplotlib scikit-learn

step 2: build th basic calsiffer
check the mnist basic code 

step 3: understanding the baseline
1. train the basic mnist classifer
2. see how well the perfoms on the clean data
3. understand what "normal" prediciton looks like

Questions to think about while training:

What's the accuracy on clean test data?
How confident is the model in its predictions?
Are there any images it struggles with naturally?

Step 4: Next - Creating Adversarial Examples
Once you have the basic classifier working, we'll create adversarial examples to see how small changes can fool it.

Before moving forward:

Run the classifier code
Check the test accuracy (should be ~99%+)
Look at a few predictions

Let me know how the training goes and what accuracy you get! Then we'll move to the fun part - breaking it with tiny modifications.

# AI Security MNIST Exercise - Requirements
# =====================================

# Core ML libraries
tensorflow>=2.10.0
numpy>=1.21.0
scikit-learn>=1.1.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0

# Data processing
pandas>=1.4.0
pillow>=9.0.0

# Jupyter notebook support
jupyter>=1.0.0
ipykernel>=6.0.0

# Additional utilities
tqdm>=4.64.0  # Progress bars
opencv-python>=4.6.0  # Image processing

# Development tools (optional)
pytest>=7.0.0  # Testing
black>=22.0.0  # Code formatting
flake8>=4.0.0  # Linting

# For advanced attacks (optional)
# cleverhans>=4.0.0  # Additional adversarial attacks library
# foolbox>=3.3.0     # Adversarial attacks framework

What this gives you:
✅ Professional GitHub structure with proper documentation
✅ Modular, well-documented code that's easy to understand and modify
✅ Multiple attack methods (FGSM, PGD ready for extension)
✅ Comprehensive visualization of results
✅ Experiment logging to track your research
✅ Robustness evaluation across different attack strengths
Expected Results:

Baseline accuracy: ~99%+ on clean images
FGSM attack success:

ε=0.01: ~10-20% success rate
ε=0.1: ~60-80% success rate
ε=0.3: ~90%+ success rate

