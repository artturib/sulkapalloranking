# Sulkapallon nelinpelin pelaajaratingit

## Tavoite

Kaveriporukan sulkapallovuoron nelinpelituloksista lasketaan pelaajakohtaiset vahvuusestimaatit ja ennusteet tuleville otteluille. Parit kierrätetään systemaattisesti, joten yksilövahvuudet ovat tunnistettavissa.

Toteutus pelkästään Pythonilla.

## Data

Noin 12 erää viikossa, keskimäärin noin 35 rallia per erä (noin 420 rallia viikossa). Pelaajien määrä: TÄYDENNÄ.

Syöteformaatti (CSV, yksi rivi per erä):

```
pvm,a1,a2,b1,b2,pisteet_a,pisteet_b
2026-10-01,Antti,Ville,Matti,Jussi,21,15
```

Erän pistemäärät tallennetaan aina, koska koko analyysi perustuu pallotason informaatioon.

## Malli 1: TrueSkill Through Time pallotasolla (päämalli)

Kirjasto: `trueskillthroughtime`.

Jokainen pallo käsitellään omana pelinään. Tulos 21-15 tuottaa 21 peliä, joissa A voittaa, ja 15, joissa B voittaa.

Tärkeää:
- Kaikki saman viikon pallot saavat SAMAN aikaleiman (viikkonumero). Järjestystä ei randomisoida. Erillinen aikaleima per pallo sallisi taidon driftata pallojen välillä, mikä on väärin.
- Aikayksikkö on viikko.
- γ (taidon drift per aikayksikkö) valitaan datasta log-evidenssillä, ei oletusarvolla. Jos paras γ on 0, taidot eivät mitattavasti muutu.

```python
from trueskillthroughtime import History

comp, times = [], []
for viikko, a, b, pa, pb in pelit:
    comp += [[a, b]] * pa   # voittaja ensin
    comp += [[b, a]] * pb
    times += [viikko] * (pa + pb)

for g in [0.0, 0.005, 0.01, 0.02, 0.05]:
    h = History(comp, times=times, gamma=g)
    h.convergence()
    print(g, h.log_evidence())
```

Tulkinta (β = 1): μ-luvuilla on merkitystä vain erotuksina. Pallon voittotodennäköisyys:

P(A voittaa pallon) = Φ( (Σ_A μ − Σ_B μ) / sqrt(4β² + Σ σ²) )

Rajoite: pallojen riippumattomuusoletus tekee σ:ista liian kapeita. Piste-estimaatit ovat kunnossa.

## Malli 2: binomi-logit (vertailu ja diagnostiikka)

Staattinen ML-estimaatti, jolla tarkistetaan TTT:n tulokset ja pallojen riippuvuus.

- Vaste: voitetut pallot / pelatut pallot joukkueen A näkökulmasta.
- Selittäjät: pelaajadummyt, A:n pelaajille +1, B:n pelaajille −1, muut 0.
- Ei vakiota. Normalisointi: θ:iden summa nolla.
- Esim. `statsmodels` GLM Binomial.
- Ylihajonta: Pearson χ²/df. Jos selvästi yli 1, raportoi quasi-binomi-keskivirheet (`scale='X2'`).

Odotettu tarkkuus n pelaajalla N pallon jälkeen: SE(θ_i) ≈ (n−1)/sqrt(nN) logit-asteikolla, pallotodennäköisyytenä noin 0,25 kertaa tämä. Kuudella pelaajalla noin 2,5 pp viikossa, 1,2 pp neljässä viikossa.

## Ennusteet

Pallotodennäköisyydestä erän voittotodennäköisyys sulkapallon säännöillä (21 pistettä, kahden pisteen ero, katto 30):

```python
from functools import lru_cache

def era_voitto(p):
    @lru_cache(None)
    def f(a, b):
        if (a >= 21 and a - b >= 2) or a == 30: return 1.0
        if (b >= 21 and b - a >= 2) or b == 30: return 0.0
        return p * f(a + 1, b) + (1 - p) * f(a, b + 1)
    return f(0, 0)
```

## Tulosteet

1. Ranking-taulukko: pelaaja, μ, σ, konservatiivinen arvo μ − 3σ.
2. Oppimiskäyrät ajan yli (TTT:n `learning_curves()`).
3. Otteluennustin: kaksi paria sisään, pallon ja erän voittotodennäköisyys ulos.
4. Parhaat parijaot vuorolle: kaikki jaot, joissa erät ovat mahdollisimman tasaisia.

## Myöhemmin (ei nyt)

Pariyhteensopivuus: binomi-logitiin parikohtaiset interaktiotermit ridge-penaltilla. Data riittää tähän aikaisintaan puolen vuoden jälkeen ja vain isoille vaikutuksille.
