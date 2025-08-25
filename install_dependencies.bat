@echo off
echo Installing dependencies for CUMBHv2 Emergency Alert System...
echo.

echo Installing Python packages...
pip install twilio
pip install ultralytics
pip install scikit-image
pip install folium
pip install streamlit-folium
pip install plotly
pip install pandas
pip install opencv-python
pip install numpy
pip install streamlit

echo.
echo Installation complete!
echo.
echo Next steps:
echo 1. Set up Twilio credentials as environment variables
echo 2. Run: streamlit run streamlit_app.py
echo 3. Navigate to Emergency Alerts page to configure contacts
echo.
pause
