
Data taget fra SSIs historiske opgørelser af COVID-19, downloadet fra fanen "Filer med overvågningsdata (zip-csv), fra den 6. maj 2020 til 31. marts 2023" på 
https://www.ssi.dk/sygdomme-beredskab-og-forskning/sygdomsovervaagning/c/historiske-covid-19-opgoerelser 

NyePositive_PCR: Kolonnen "NewPositive" fra "Test_pos_over_time"
AntalTest_PCR: Kolonnen "Tested" fra "Test_pos_over_time"
NyePositive_Antigen: Kolonnen "NewPositive" fra "Test_pos_over_time_antigen"
AntalTest_Antigen: Kolonnen "NewPositive" fra "Test_pos_over_time_antigen"
Dødfald: Kolonnen "antal_døde" i "Deaths_over_time"
Nyindlæggelser: Kolonnen "Total" i "Newly_admitted_over_time"


Følgende er kopieret fra SSIs egne forklaringer:

------------------------------------------------------

Fil 7: Test_pos_over_time: 

Tabellen indeholder kun personer testet ved PCR-test. 
Denne tabel fokuserer på testede personer per dag frem for personer testet i hele perioden. I modsætning til de andre tabeller kan en person derfor bidrage flere gange til denne tabel, dog kun en gang per dag.

Dette er modsat dashboardet (www.ssi.dk/covid19data), hvor positivprocenten beregnes over en uge, med antal personer som er testet positive seneste syv komplette dage over antallet af alle personer testet seneste syv komplette dage.

Date: Datoer i formatet YYYY-MM-DD som der stratificeres efter.
NewPositive: Antallet af personer, som for første gang er testet positive for covid-19, på en given dag. Hvis det er en reinfektion skal der være gået mindst 60 dage siden starten af forrige infektion for at en person tælles med i denne kategori.
NotPrevPos: Antallet af personer testet på en given dag, hvor sidste infektion ligger mere end 60 dage tilbage i tiden. Indeholder negative tests for personer, som ikke har testet positive før, samt den første positive test for hver infektion.
PosPct: Andelen (%) af personer som er testet positive. Dette er beregnet som NewPositive divideret med NotPrevPos. 
PrevPos: Antallet af personer testet på en given dag, som tidligere har testet positive hvor sidste infektion ligger indenfor de sidste 60 dage. Den positive test som indikerer infektionsstart indgår ikke her.
Tested: Det samlede antal testede personer på en given dag, som har fået konklusivt svar (positiv eller negativ). Dette kan udregnes som NotPrevPos plus PrevPos.
Tested_kumulativ: Det kumulerede antal personer testet op til og med en given dag, som har fået konklusivt svar (positiv eller negativ). Udregnes som summen af værdierne i Tested frem til den givne dag. En person, der er testet på flere forskellige dage, kan bidrage mere end en gang i denne sum.

Noter: I den sidste række (Antal personer) er den totale opgørelse opgjort således, at en person kun kan indgå en gang i hver søjle, i modsætning til de andre. Af denne grund er Tested det samme som NotPrevPos i denne række, og ikke en sum af NotPrevPos og PrevPos, som i de andre.

------------------------------------------------------

Fil 10: Test_pos_over_time_antigen: 
Denne tabel indeholder - i modsætning til test_pos_over_time - kun personer testet ved antigen-test og personer tæller bestandigt som tidligere positive efter den allerførste positive antigen-test.
Denne tabel fokuserer ligeledes på testede personer per dag frem for personer testet i hele perioden. I modsætning til de andre tabeller kan en person derfor bidrage flere gange til denne tabel, dog kun en gang per dag. I denne fil indgår udelukkende personer testet med antigen-test.
Fra 1/9-2021 ændres filen til at indeholde samtlige personer med gyldigt CPR nummer uanset om de har været testet med PCR før. Før denne dato har filerne udelukkende indeholdt personer med gyldigt CPR nummer som samtidig også havde fået foretaget en PCR test. Personer uden gyldigt CPR nummer kan ikke følges korrekt og tæller derfor ikke med i denne opgørelse.


Date: Datoer i formatet YYYY-MM-DD som der stratificeres efter.
NewPositive: Antallet af personer, som for første gang er testet positive for covid-19, på en given dag.
NotPrevPos: Antallet af personer testet på en given dag, som ikke har testet positive på en tidligere dato. Indeholder negative tests for folk, som ikke har testet positive før, samt første positive test.
PosPct: Andelen af personer som er testet positive. Dette er beregnet som NewPositive divideret med NotPrevPos. Bemærk at prøver taget på folk, som tidligere har testet positive ikke er medregnet her.
PrevPos: Antallet af personer testet på en given dag blandt personer, som tidligere har testet positive. Den første test, som er positiv, indgår ikke her.
Tested: Det samlede antal testede personer på en given dag, som har fået konklusivt svar (positiv eller negativ). Dette kan udregnes som NotPrevPos plus PrevPos.
Tested_kumulativ: Det kumulerede antal personer testet op til og med en given dag, som har fået konklusivt svar (positiv eller negativ). Udregnes som summen af værdierne i Tested frem til den givne dag. En person, der er testet på flere forskellige dage, kan bidrage mere end en gang i denne sum.

Noter: I den sidste række (Antal personer) er den totale opgørelse opgjort således, at en person kun kan indgå en gang i hver søjle, i modsætning til de andre. Af denne grund er Tested det samme som NotPrevPos i denne række, og ikke en sum af NotPrevPos og PrevPos, som i de andre.

------------------------------------------------------

Fil 2: Deaths_over_time:

Dato: Datoer i formatet YYYY-MM-DD, som der stratificeres efter.
Antal_døde: Antal døde registreret på en given dag. En person indgår, hvis de er registreret i enten CPR eller Dødsårsagsregisteret. Er en person registreret både i CPR og Dødsårsagsregisteret med forskellige dødsdatoer, bruges datoen angivet i CPR. Dødsfald relateret til covid-19 defineres som et covid-19 bekræftet tilfælde, der er død indenfor 30 dage efter påvist covid-19-infektion. Covid-19 er ikke nødvendigvis den tilgrundliggende årsag til dødsfaldet.

------------------------------------------------------

Fil 6: Newly_admitted_over_time:

Indlæggelsestal i denne tabel er baseret både på LPR og de daglige øjebliksbilleder, som indrapporteres af regionerne hver dag, og en person medregnes, når blot de er registreret indlagt i en af disse. Er en person registreret i begge med forskellige datoer, så bruges den første. Denne opgørelse er udelukkende baseret på covid-19 bekræftede tilfælde. Læs mere om definitionen for covid-19 relateret indlæggelse på Datakilder og definitioner (ssi.dk)

Dato: Datoer i formatet YYYY-MM-DD som der stratificeres efter.
Hovedstaden: Antal nyindlagte på en given dag i Region Hovedstaden.
Sjælland: Antal nyindlagte på en given dag i Region Sjælland.
Syddanmark: Antal nyindlagte på en given dag i Region Syddanmark.
Midtjylland: Antal nyindlagte på en given dag i Region Midtjylland.
Nordjylland: Antal nyindlagte på en given dag i Region Nordjylland.
Ukendt Region: Antal nyindlagte på en given dag, som ikke har en registreret bopæl i nogen region.
Total: Det totale antal nyindlagte på en given dag, dette tal er summen af værdierne i Hovedstaden, Sjælland, Syddanmark, Midtjylland, Nordjylland og Ukendt Region.

