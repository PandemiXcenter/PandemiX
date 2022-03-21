# import codecs 
# exec(open("./RegionalFigures.py").read())
# import RegionalFigures.py
# print(open("./RegionalFigures.py").read())

print('-------- Winter overview figures --------')

exec(open("Winter2022Overview.py",'r',encoding='utf-8').read())

print('-------- Done with winter overview figures --------')


print('-------- Regional figures --------')

exec(open("RegionalFigures.py",'r',encoding='utf-8').read())

print('-------- Done with regional figures --------')


print('-------- Age-distribution figures --------')

exec(open("DanishAgeDistributions.py",'r',encoding='utf-8').read())

print('-------- Done with age-distribution figures --------')


print('-------- Immunity-overview figures --------')

exec(open("ImmunityOverview.py",'r',encoding='utf-8').read())

print('-------- Done with immunity-overview figures --------')
