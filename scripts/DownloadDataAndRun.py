
## Main file for downloading data and running scripts to make current overview figures

print('-------- Downloading new data --------')

exec(open("download_data_script.py",'r',encoding='utf-8').read())

print('-------- Done downloading new data --------')

print('-------- Late 2022 overview figures --------')

exec(open("Overview_Late2022.py",'r',encoding='utf-8').read())

print('-------- Done making late 2022 overview figures --------')