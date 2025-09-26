"""
Setup script to download all required NLTK data
"""
import nltk
import ssl

# Handle SSL certificate issues
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

print("Downloading required NLTK data packages...")

packages = [
    'punkt',
    'punkt_tab',
    'stopwords',
    'averaged_perceptron_tagger',
    'wordnet'
]

for package in packages:
    print(f"Downloading {package}...")
    try:
        nltk.download(package, quiet=False)
        print(f"✓ {package} downloaded successfully")
    except Exception as e:
        print(f"✗ Failed to download {package}: {e}")

print("\nNLTK setup complete!")
print("You can now run: python -m streamlit run app_lite.py")