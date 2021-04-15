call "C:\ProgramData\Anaconda3\Scripts\activate.bat" main 
cd "D:\Pandemix\Github\DanskeData" 
python get_data.py
python antigen_test.py
python export_data_for_web.py