@echo off
echo Installing Resume-ATS Matcher Lite dependencies...
echo.

echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Installing basic packages...
pip install streamlit
pip install pandas
pip install plotly
pip install PyPDF2
pip install python-docx
pip install pdfplumber
pip install nltk
pip install rake-nltk
pip install textstat

echo.
echo Installation complete!
echo.
echo To run the lite version: python -m streamlit run app_lite.py
echo.
pause