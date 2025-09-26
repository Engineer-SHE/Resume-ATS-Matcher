@echo off
echo Installing Resume-ATS Matcher dependencies...
echo.
echo Step 1: Installing basic dependencies...
pip install --upgrade pip
pip install -r requirements-simple.txt

echo.
echo Step 2: Installing ML packages (this may take a while)...
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers
pip install transformers

echo.
echo Step 3: Installing spaCy...
pip install spacy
python -m spacy download en_core_web_sm

echo.
echo Installation complete!
echo Run the application with: streamlit run app.py
pause