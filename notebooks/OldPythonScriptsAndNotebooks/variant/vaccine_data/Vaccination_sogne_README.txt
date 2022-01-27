Dette dokument beskriver indholdet af filen Vaccination_sogne. Variabelnavnene, som beskrives nedenfor, refererer til søjlenavne.
Filen bliver opdateret hver onsdag, med data fra tirsdag. 

Datakilder og opgørelsesfrekvenser:
•	CPR-registeret: Opgjort tirsdag
•	Målgruppedata (TcDK): Opgjort mandag
•	Det Danske Vaccinationsregister: Opgjort tirsdag
•	Bookingdata: Opdateres mandag aften


Generel beskrivelse:
I opgørelsen indgår personer, der er inviteret til vaccination mod COVID-19 for mindst 7 dage siden. Det vil sige, at personer, der er inviteret inden for de sidste 7 dage ikke indgår. 

I opgørelsen tages der udgangspunkt i de forskellige trin i vaccinationsforløbet. 
Opgørelsen er lavet, så personer kun kan optræde på ét af disse trin, og de placeres på det højest muligt opnåede trin i vaccinationsforløbet, hvor påbegyndt vaccination er et højere trin
end inviteret til og booket tid til vaccination. Derfor summer andelene på de fire trin til 100 pct. 
Personer, der i forbindelse med tilvalgsordningen bliver vaccineret med vaccinen fra Johnson & Johnson vil indgå under færdigvaccinerede.

Antal inviterede er vist i antal personer, da det er personer på dette trin, som kommunerne pt. retter en særlig indsats overfor. 

Vær opmærksom på, at det afviger fra opgørelserne på det interaktive dashboard og andre opgørelser for vaccinationstilslutning på SSIs hjemmeside, 
hvor personer kan indgå i både gruppen for påbegyndt vaccination og færdigvaccineret.

Diskretionering:
Opgørelsen vil blive diskretioneret, hvis befolkningen i sognet er færre end 20 personer. I de fleste tilfælde vil personerne fortsat indgår i opgørelsen af kommunen, 
men uden at være tilknyttet et sogn. Er der under 20 personer, der ikke kan tilknyttes et sogn vil data dog blive fjernet fra opgørelsen. 
Hvis der er mindre end 10 personer i gruppen af antal inviterede, der ikke har booket eller er vaccineret, vil det fremgå som <10 og andelene vil være diskretioneret.

------------------------------------------------------

Beskrivelse af variable:

Bo_kom_today:			Bopælskommune på opgørelsestidspunktet.
Sognekode:			Koden på sognet.
Sogn: 				Bopælssogn på opgørelsestidspunktet. Enkelte sogne fordeler sig geografisk over flere kommuner og vil derfor fremgå flere gange.
				Personer, der ikke har en adresse, kan ikke placeres i et sogn. De vil fremgå under kommunen, men uden et sogn (se også under diskretionering).
Inviterede_i_sognet:            Det samlede antal personer, som opgørelsen tager udgangspunkt i.
Antal_inviteret_ikke_booket:    Antal personer, som er inviteret, men endnu ikke har booket tid, er påbegyndt eller har færdiggjort vaccination.
Andel_inviteret_ikke_booket:    Andel personer, som er inviteret, men endnu ikke har booket tid, er påbegyndt eller har færdiggjort vaccination.
Andel_booket:                   Andel personer, som har booket tid, men som ikke er påbegyndt eller har færdiggjort vaccination.
Andel_påbegyndt:                Andel personer, som er påbegyndt vaccination, men ikke har færdiggjort vaccination.
Andel_færdigvaccineret:         Andel personer, som har færdiggjort vaccination.

-----------------------------------------------------