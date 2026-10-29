"""Sulkapallon nelinpelin pelaajaratingit (SPEC.md).

Asennus: python3.13 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt
Ajo:     .venv/bin/python koodit/ranking.py [--csv polku] [--out polku]

Hakee erätulokset Google Sheetsistä (tai paikallisesta CSV:stä), laskee
TrueSkill Through Time -ratingit pallotasolla ja binomi-logitin vertailuksi,
tulostaa taulukot konsoliin ja kirjoittaa jaettavan HTML-raportin.
"""

import argparse
import csv
import hashlib
import io
import itertools
import json
import math
import os
import ssl
import sys
import urllib.request
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path

import certifi
import numpy as np
import statsmodels.api as sm
from scipy.stats import norm
from trueskillthroughtime import History

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = Path(__file__).resolve().parent / "raportti_pohja.html"
DEFAULT_OUT = ROOT / "tulokset" / "sulkapalloranking.html"
VALIMUISTI = ROOT / "tulokset" / ".valimuisti.json"
# Sheetin julkaisulinkki ei ole repossa: ympäristömuuttuja tai gitignoroitu tiedosto.
DATA_URL_TIEDOSTO = ROOT / ".data_url"

BETA = 1.0
MU0, SIGMA0 = 0.0, 6.0
GAMMAS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]


# ---------------------------------------------------------------- data

def lue_csv(lahde):
    if lahde:
        teksti = Path(lahde).read_text(encoding="utf-8")
    else:
        # Varmenteet certifistä eikä käyttöjärjestelmästä: macOS:n varasto voi
        # sisältää vaarannetun emSign ECC Root CA - G3:n (eMudhra-tapaus 10/2026).
        konteksti = ssl.create_default_context(cafile=certifi.where())
        url = os.environ.get("DATA_URL") or (
            DATA_URL_TIEDOSTO.read_text().strip() if DATA_URL_TIEDOSTO.exists() else "")
        if not url:
            sys.exit(f"Aseta DATA_URL-ympäristömuuttuja tai kirjoita osoite tiedostoon {DATA_URL_TIEDOSTO}.")
        with urllib.request.urlopen(url, timeout=30, context=konteksti) as r:
            teksti = r.read().decode("utf-8")
    return list(csv.reader(io.StringIO(teksti)))


def voittajan_pisteet(haviaja):
    """Sulkapallon säännöt: 21 pisteeseen, kahden pisteen ero, katto 30."""
    if haviaja <= 19:
        return 21
    return min(haviaja + 2, 30)


def jasenna(rivit):
    """Muuntaa lomakkeen rivit eriksi: pvm, viikko, a (voittajat), b, pa, pb."""
    otsikot = rivit[0]
    pelaajasarakkeet = [
        (i, h[h.index("[") + 1:h.index("]")])
        for i, h in enumerate(otsikot) if h.startswith("Pelaajat [")
    ]
    i_pisteet = otsikot.index("Häviäjien pisteet")
    i_vier = otsikot.index("Vierailija") if "Vierailija" in otsikot else None
    i_vier2 = next((i for i, h in enumerate(otsikot) if h.startswith("Jos vierailija")), None)

    erat, varoitukset = [], []
    for nro, r in enumerate(rivit[1:], start=2):
        if not any(c.strip() for c in r):
            continue
        r = r + [""] * (len(otsikot) - len(r))
        try:
            ts = datetime.strptime(r[0].strip(), "%m/%d/%Y %H:%M:%S")
            haviaja = int(r[i_pisteet])
        except ValueError:
            varoitukset.append(f"rivi {nro}: aikaleima tai pisteet puuttuu")
            continue
        voittajat, haviajat = [], []
        for i, nimi in pelaajasarakkeet:
            if nimi == "Vierailija":
                nimi = (r[i_vier2].strip() if i_vier2 is not None else "") or \
                       (r[i_vier].strip() if i_vier is not None else "") or "Vierailija"
            if r[i] == "Voittajat":
                voittajat.append(nimi)
            elif r[i] == "Häviäjät":
                haviajat.append(nimi)
        if len(voittajat) != 2 or len(haviajat) != 2:
            varoitukset.append(f"rivi {nro}: ei 2 vs 2 ({voittajat} / {haviajat}), ohitettu")
            continue
        maanantai = ts.date() - timedelta(days=ts.weekday())
        erat.append({
            "aika": ts, "pvm": ts.date(), "maanantai": maanantai,
            "a": sorted(voittajat), "b": sorted(haviajat),
            "pa": voittajan_pisteet(haviaja), "pb": haviaja,
        })
    erat.sort(key=lambda e: e["aika"])
    if erat:
        eka = erat[0]["maanantai"]
        for e in erat:
            e["viikko"] = (e["maanantai"] - eka).days // 7
    return erat, varoitukset


# ---------------------------------------------------------------- malli 1: TTT

def ttt_pelit(erat):
    comp, times = [], []
    for e in erat:
        comp += [[e["a"], e["b"]]] * e["pa"]   # voittaja ensin
        comp += [[e["b"], e["a"]]] * e["pb"]
        times += [e["viikko"]] * (e["pa"] + e["pb"])
    return comp, times


def _keskitetyt(h):
    """Pelaajien μ erotuksina viikon keskiarvosta sekä σ, joka viikolle."""
    viikoittain = {}
    for p, c in h.learning_curves().items():
        for t, g in c:
            viikoittain.setdefault(t, {})[p] = (g.mu, g.sigma)
    out = {}
    for t, d in viikoittain.items():
        ka = sum(m for m, _ in d.values()) / len(d)
        for p, (m, s) in d.items():
            out[(t, p)] = (m - ka, s)
    return out


def sovita_ttt(erat, gamma, tol=1e-4, max_iter=40):
    """TTT-sovitus omalla pysähtymisehdolla.

    Kirjaston oma convergence() jää junnaamaan kaikille yhteiseen tasoon
    (kaikkien μ:t liukuvat hitaasti yhdessä kohti priorin keskiarvoa). Se ei
    vaikuta erotuksiin, ennusteisiin eikä evidenssiin, joten konvergenssi
    tarkistetaan vain keskitetyistä μ:istä ja σ:ista.
    """
    comp, times = ttt_pelit(erat)
    h = History(comp, times=times, mu=MU0, sigma=SIGMA0, beta=BETA, gamma=gamma)
    vanha = _keskitetyt(h)
    for _ in range(max_iter):
        h.iteration()
        uusi = _keskitetyt(h)
        muutos = max(max(abs(uusi[k][0] - vanha[k][0]), abs(uusi[k][1] - vanha[k][1]))
                     for k in uusi)
        vanha = uusi
        if muutos < tol:
            break
    return h


def valitse_gamma(erat):
    tulokset = []
    for g in GAMMAS:
        h = sovita_ttt(erat, g)
        tulokset.append((g, h.log_evidence(), h))
    # Tasapelissä (esim. vain yksi viikko dataa) pienin γ.
    paras = max(tulokset, key=lambda t: (round(t[1], 6), -t[0]))
    return paras, [(g, le) for g, le, _ in tulokset]


def kayrat_keskitetyt(h):
    """Oppimiskäyrät keskitettyinä: pelaaja -> [(viikko, μ, σ), ...]."""
    out = {}
    for (t, p), (mu, s) in sorted(_keskitetyt(h).items()):
        out.setdefault(p, []).append((t, mu, s))
    return out


def viimeisimmat(kayrat):
    return {p: c[-1] for p, c in kayrat.items()}


def tila_ennen_viikkoa(erat, w, gamma, valimuisti):
    """Ratingit viikon w alussa: sovitus vain viikkoja < w käyttäen (välimuistissa)."""
    aiemmat = [e for e in erat if e["viikko"] < w]
    if not aiemmat:
        return {}
    avain = hashlib.sha1(json.dumps(
        [gamma, BETA, SIGMA0] + [[e["viikko"], e["a"], e["b"], e["pa"], e["pb"]] for e in aiemmat]
    ).encode()).hexdigest()
    if avain not in valimuisti:
        valimuisti[avain] = viimeisimmat(kayrat_keskitetyt(sovita_ttt(aiemmat, gamma)))
    return {p: tuple(v) for p, v in valimuisti[avain].items()}


# ---------------------------------------------------------------- ennusteet

def ero_jakauma(a, b, tila, gamma=0.0, viikko=None):
    """Joukkueiden taitoeron jakauma N(dmu, var): A:n μ-summa − B:n μ-summa."""
    dmu, var = 0.0, 0.0
    for merkki, joukkue in ((1, a), (-1, b)):
        for p in joukkue:
            t, mu, s = tila.get(p, (viikko or 0, MU0, SIGMA0))
            drift = max((viikko - t) if viikko is not None else 0, 0) * gamma ** 2
            dmu += merkki * mu
            var += s ** 2 + drift
    return dmu, var


def pallo_p(a, b, tila, gamma=0.0, viikko=None):
    """P(A voittaa pallon), taitojen epävarmuus integroituna pois."""
    dmu, var = ero_jakauma(a, b, tila, gamma, viikko)
    return float(norm.cdf(dmu / math.sqrt(4 * BETA ** 2 + var)))


def era_taulu(p):
    """Erän voittotodennäköisyys mistä tahansa tilanteesta (a, b), kun pallon p on tunnettu."""
    @lru_cache(None)
    def f(a, b):
        if (a >= 21 and a - b >= 2) or a == 30: return 1.0
        if (b >= 21 and b - a >= 2) or b == 30: return 0.0
        return p * f(a + 1, b) + (1 - p) * f(a, b + 1)
    return f


def era_voitto(p, alku_a=0, alku_b=0):
    return era_taulu(p)(alku_a, alku_b)


_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(32)
_GH_W = _GH_W / _GH_W.sum()


def era_epavarma(dmu, var):
    """Erän voittotodennäköisyys taitojen epävarmuus huomioiden.

    Sama (tuntematon) taitoero pätee erän jokaiseen palloon, joten
    epävarmuus integroidaan erätasolla: E[erä(Φ(Δ / 2β))], Δ ~ N(dmu, var).
    Palauttaa funktion (alku_a, alku_b) -> todennäköisyys.
    """
    if var <= 0:
        f = era_taulu(float(norm.cdf(dmu / (2 * BETA))))
        return f
    taulut = [era_taulu(float(norm.cdf((dmu + math.sqrt(var) * x) / (2 * BETA)))) for x in _GH_X]
    return lambda a, b: float(sum(w * t(a, b) for w, t in zip(_GH_W, taulut)))


def tasoitus(g):
    """Altavastaajan etumatka (pisteinä), joka vie erän lähimmäs 50-50.

    g: funktion (alku_a, alku_b) -> A:n erävoitto. Palauttaa (etumatka,
    erän voittotodennäköisyys suosikin näkökulmasta etumatkan jälkeen,
    saako etumatkan B).
    """
    b_altavastaaja = g(0, 0) >= 0.5
    suosikki = (lambda h: g(0, h)) if b_altavastaaja else (lambda h: 1 - g(h, 0))
    paras = (0, suosikki(0))
    for h in range(1, 20):
        v = suosikki(h)
        if abs(v - 0.5) < abs(paras[1] - 0.5):
            paras = (h, v)
        if v < 0.5:
            break
    return paras[0], paras[1], b_altavastaaja


def ennuste(a, b, tila, gamma=0.0, viikko=None):
    dmu, var = ero_jakauma(a, b, tila, gamma, viikko)
    g = era_epavarma(dmu, var)
    h, v, b_alta = tasoitus(g)
    return {"pallo": float(norm.cdf(dmu / math.sqrt(4 * BETA ** 2 + var))), "era": g(0, 0),
            "tasoitus": h, "tasoitukselle": "b" if b_alta else "a", "era_tasoituksella": v}


# ---------------------------------------------------------------- malli 2: binomi-logit

def binomi_logit(erat, pelaajat):
    n = len(pelaajat)
    idx = {p: i for i, p in enumerate(pelaajat)}
    X, y, m = [], [], []
    for e in erat:
        x = np.zeros(n)
        for p in e["a"]: x[idx[p]] += 1
        for p in e["b"]: x[idx[p]] -= 1
        X.append(x); y.append(e["pa"]); m.append(e["pa"] + e["pb"])
    X, y, m = np.array(X), np.array(y), np.array(m)
    # Summa nolla: θ_n = −Σ θ_i, joten sarake j = x_j − x_n.
    Z = X[:, :-1] - X[:, [-1]]
    endog = np.column_stack([y, m - y])
    malli = sm.GLM(endog, Z, family=sm.families.Binomial())
    tulos = malli.fit()
    df = max(len(y) - Z.shape[1], 1)
    hajonta = float(np.sum(tulos.resid_pearson ** 2) / df)
    quasi = hajonta > 1.5
    if quasi:
        tulos = malli.fit(scale="X2")
    T = np.vstack([np.eye(n - 1), -np.ones(n - 1)])   # θ = T β
    theta = T @ tulos.params
    cov = T @ tulos.cov_params() @ T.T
    se = np.sqrt(np.diag(cov))
    N = int(m.sum())
    return {
        "theta": dict(zip(pelaajat, theta.tolist())),
        "se": dict(zip(pelaajat, se.tolist())),
        "hajonta": hajonta, "df": df, "quasi": quasi,
        "pallot": N, "se_odotettu": (n - 1) / math.sqrt(n * N) if N else None,
    }


# ---------------------------------------------------------------- malli 3: Kalman (taito + jaksaminen)
#
# Jokaisella pelaajalla on kaksi piilotilaa: taito s ja jaksamiskerroin f.
# Erän havainto on pallo-osuus TTT:n asteikolle muunnettuna,
#     y = 2β Φ⁻¹(pa / n)  ≈  Σ_A (s + f·(x − c)) − Σ_B (s + f·(x − c)) + kohina,
# missä x on pelaajan samana iltana jo pelaamat erät ja c_i pelaajan oma
# keskimääräinen x (taito = pelaajan taso hänen tyypillisessä kohdassaan iltaa). Illan sisällä tilat eivät muutu;
# viikkojen välillä s muuttuu nopeudella γ ja f nopeudella η. Yhteinen
# jaksaminen g on mukana staattisena taustamuuttujana, koska se erottuu vain
# parien kuormituseroista. Raportoidaan s ja f suhteessa porukan keskiarvoon.

KALMAN_GAMMAS = [0.0, 0.01, 0.02, 0.05, 0.1]
KALMAN_ETAS = [0.0, 0.005, 0.01, 0.02]
KALMAN_TAUS = [0.01, 0.02, 0.05, 0.1]   # jaksamiskertoimien priorihajonta (kutistettu malli)
TAU_VAPAA = 0.5                          # käytännössä kutistamaton
G0 = 0.5                                 # yhteisen jaksamisen priorihajonta


def kuormitukset(erat):
    """Lisää jokaiseen erään 'x': pelaaja -> montako erää hän on jo pelannut samana iltana."""
    laskuri = {}
    for e in erat:
        ilta = laskuri.setdefault(e["pvm"], {})
        e["x"] = {p: ilta.get(p, 0) for p in e["a"] + e["b"]}
        for p in e["a"] + e["b"]:
            ilta[p] = ilta.get(p, 0) + 1


def _havainto(e):
    n = e["pa"] + e["pb"]
    osuus = min(max(e["pa"] / n, 0.5 / n), 1 - 0.5 / n)
    z = norm.ppf(osuus)
    y = 2 * BETA * z
    r = 4 * BETA ** 2 * osuus * (1 - osuus) / (n * norm.pdf(z) ** 2)
    return y, r


def kalman(erat, pelaajat, gamma, eta, tau_f, c, tasoita=True):
    """Kalman-suodatin ja RTS-tasoitin viikoittain. Palauttaa (log-evidenssi, viikot, tilat)."""
    # numpy 2.0 + macOS Accelerate antaa matmulista aiheettomia FP-varoituksia.
    with np.errstate(all="ignore"):
        return _kalman(erat, pelaajat, gamma, eta, tau_f, c, tasoita)


def _kalman(erat, pelaajat, gamma, eta, tau_f, c, tasoita):
    n_p = len(pelaajat)
    idx = {p: i for i, p in enumerate(pelaajat)}
    dim = 2 * n_p + 1
    m = np.zeros(dim)
    P = np.diag([SIGMA0 ** 2] * n_p + [tau_f ** 2] * n_p + [G0 ** 2])
    Q1 = np.diag([gamma ** 2] * n_p + [eta ** 2] * n_p + [0.0])
    viikot = sorted({e["viikko"] for e in erat})
    ennuste, suodatettu = [], []
    ll, edellinen = 0.0, None
    for w in viikot:
        if edellinen is not None:
            P = P + Q1 * (w - edellinen)
        ennuste.append((m.copy(), P.copy()))
        for e in (e for e in erat if e["viikko"] == w):
            h = np.zeros(dim)
            for merkki, joukkue in ((1, e["a"]), (-1, e["b"])):
                for p in joukkue:
                    h[idx[p]] += merkki
                    h[n_p + idx[p]] += merkki * (e["x"][p] - c[p])
                    h[-1] += merkki * (e["x"][p] - c[p])
            y, r = _havainto(e)
            Ph = P @ h
            S = h @ Ph + r
            v = y - h @ m
            ll += -0.5 * (math.log(2 * math.pi * S) + v * v / S)
            K = Ph / S
            m = m + K * v
            P = P - np.outer(K, Ph)
            P = (P + P.T) / 2
        suodatettu.append((m.copy(), P.copy()))
        edellinen = w
    if not tasoita:
        return ll, viikot, suodatettu
    tasoitettu = [None] * len(viikot)
    tasoitettu[-1] = suodatettu[-1]
    for k in range(len(viikot) - 2, -1, -1):
        mf, Pf = suodatettu[k]
        mp, Pp = ennuste[k + 1]
        ms, Ps = tasoitettu[k + 1]
        C = Pf @ np.linalg.pinv(Pp)
        tasoitettu[k] = (mf + C @ (ms - mp), Pf + C @ (Ps - Pp) @ C.T)
    return ll, viikot, tasoitettu


def _kalman_raportti(pelaajat, viikot, tilat, ensimmainen):
    n_p = len(pelaajat)
    dim = 2 * n_p + 1
    out = {}
    for i, p in enumerate(pelaajat):
        cs = np.zeros(dim); cs[:n_p] = -1 / n_p; cs[i] += 1
        cf = np.zeros(dim); cf[n_p:2 * n_p] = -1 / n_p; cf[n_p + i] += 1
        rivit = []
        for w, (m, P) in zip(viikot, tilat):
            if w < ensimmainen[p]:
                continue
            rivit.append({"viikko": w,
                          "taito": float(cs @ m), "taito_sd": float(math.sqrt(max(cs @ P @ cs, 0))),
                          "jaksaminen": float(cf @ m), "jaksaminen_sd": float(math.sqrt(max(cf @ P @ cf, 0)))})
        out[p] = rivit
    return out


def kalman_malli(erat, pelaajat, odotettu):
    kuormitukset(erat)
    c = {p: float(np.mean([e["x"][p] for e in erat if p in e["x"]])) for p in pelaajat}
    ensimmainen = {}
    for e in erat:
        for p in e["a"] + e["b"]:
            ensimmainen.setdefault(p, e["viikko"])

    # γ ja η kutistamattomalla jaksamisella, sitten kutistus τ samoilla γ, η.
    haku = [(g, h, kalman(erat, pelaajat, g, h, TAU_VAPAA, c, tasoita=False)[0])
            for g in KALMAN_GAMMAS for h in KALMAN_ETAS]
    gamma, eta, _ = max(haku, key=lambda t: (round(t[2], 6), -t[0], -t[1]))
    tau_haku = [(t, kalman(erat, pelaajat, gamma, eta, t, c, tasoita=False)[0]) for t in KALMAN_TAUS]
    tau = max(tau_haku, key=lambda t: (round(t[1], 6), -t[0]))[0]

    _, viikot, vapaa = kalman(erat, pelaajat, gamma, eta, TAU_VAPAA, c)
    _, _, vapaa_ajantasa = kalman(erat, pelaajat, gamma, eta, TAU_VAPAA, c, tasoita=False)
    _, _, kutistettu = kalman(erat, pelaajat, gamma, eta, tau, c)

    # Havainnot: toteutunut − TTT:n odottama pallo-osuus, pelaajan erän järjestysnumeron mukaan.
    havainnot = {p: [] for p in pelaajat}
    for e, odo in zip(erat, odotettu):
        osuus = e["pa"] / (e["pa"] + e["pb"])
        for joukkue, ero in ((e["a"], osuus - odo), (e["b"], odo - osuus)):
            for p in joukkue:
                havainnot[p].append({"x": e["x"][p], "ero": ero, "pvm": e["pvm"].isoformat()})

    return {
        "gamma": gamma, "eta": eta, "tau": tau, "c": c,
        "max_x": max(max(e["x"].values()) for e in erat) + 1,
        "max_erat": {p: max(e["x"][p] for e in erat if p in e["x"]) + 1 for p in pelaajat},
        "pp_per_yksikko": 100 * float(norm.pdf(0)) / (2 * BETA),
        "haku": [{"gamma": g, "eta": h, "log_evidenssi": v} for g, h, v in haku],
        "tau_haku": [{"tau": t, "log_evidenssi": v} for t, v in tau_haku],
        "kutistamaton": _kalman_raportti(pelaajat, viikot, vapaa, ensimmainen),
        "kutistamaton_ajantasa": _kalman_raportti(pelaajat, viikot, vapaa_ajantasa, ensimmainen),
        "kutistettu": _kalman_raportti(pelaajat, viikot, kutistettu, ensimmainen),
        "havainnot": havainnot,
    }


# ---------------------------------------------------------------- koonti

def laske(erat, valimuisti):
    pelaajat = sorted({p for e in erat for p in e["a"] + e["b"]})
    viikot = sorted({e["viikko"] for e in erat})
    viikko_pvm = {e["viikko"]: e["maanantai"] for e in erat}
    viikko_pelipvm = {}
    for e in erat:
        viikko_pelipvm.setdefault(e["viikko"], set()).add(e["pvm"])

    (gamma, le, h), evidenssit = valitse_gamma(erat)
    kayrat = kayrat_keskitetyt(h)
    tila = viimeisimmat(kayrat)
    nyt = max(viikot)

    # Kulloisenkin rankingin ennusteet: viikon w erät ennustetaan vain viikkoja < w
    # käyttäen (aito ennuste), ja lisäksi koko datan tasoitetulla arviolla (jälkikäteen).
    ennen = {w: tila_ennen_viikkoa(erat, w, gamma, valimuisti) for w in viikot}
    # Ranking sellaisena kuin se oli kunkin viikon jälkeen (vain siihen asti kertynyt data).
    jalkeen = {w: tila_ennen_viikkoa(erat, w + 1, gamma, valimuisti) for w in viikot}
    kayra_viikolla = {p: {k[0]: k for k in c} for p, c in kayrat.items()}

    eralista = []
    for i, e in enumerate(erat):
        w = e["viikko"]
        tila_w = {p: kayra_viikolla[p][w] for p in e["a"] + e["b"]}
        etuk = ennuste(e["a"], e["b"], ennen[w], gamma, w)
        jalk = ennuste(e["a"], e["b"], tila_w)
        eralista.append({
            "nro": i + 1, "pvm": e["pvm"].isoformat(), "klo": e["aika"].strftime("%H:%M"),
            "viikko": w, "a": e["a"], "b": e["b"], "pa": e["pa"], "pb": e["pb"],
            "ennuste": etuk, "jalkikateen": jalk,
        })

    # Pelaajatilastot
    pt = {p: {"erat": 0, "voitot": 0, "pallot_v": 0, "pallot_h": 0} for p in pelaajat}
    for e in erat:
        for p in e["a"]:
            pt[p]["erat"] += 1; pt[p]["voitot"] += 1
            pt[p]["pallot_v"] += e["pa"]; pt[p]["pallot_h"] += e["pb"]
        for p in e["b"]:
            pt[p]["erat"] += 1
            pt[p]["pallot_v"] += e["pb"]; pt[p]["pallot_h"] += e["pa"]

    logit = binomi_logit(erat, pelaajat)
    odotettu = [pallo_p(e["a"], e["b"], {p: kayra_viikolla[p][e["viikko"]] for p in e["a"] + e["b"]})
                for e in erat]
    kalm = kalman_malli(erat, pelaajat, odotettu)

    ranking = []
    for p in pelaajat:
        t, mu, s = tila[p]
        ranking.append({
            "pelaaja": p, "mu": mu, "sigma": s, "konservatiivinen": mu - 3 * s,
            # Keskitasoisen (μ = 0) parin kanssa keskitasoista paria vastaan.
            "pallo_ka": float(norm.cdf(mu / math.sqrt(4 * BETA ** 2 + s ** 2))),
            "viimeksi": t, **pt[p],
            "theta": logit["theta"][p], "theta_se": logit["se"][p],
        })
    for r in ranking:
        r["era_ka"] = era_epavarma(r["mu"], r["sigma"] ** 2)(0, 0)
    ranking.sort(key=lambda r: -r["mu"])

    # Parit
    parit = {}
    for e in erat:
        for joukkue, vast, pv, ph, voitti in ((e["a"], e["b"], e["pa"], e["pb"], 1),
                                               (e["b"], e["a"], e["pb"], e["pa"], 0)):
            k = " & ".join(joukkue)
            d = parit.setdefault(k, {"pari": joukkue, "erat": 0, "voitot": 0,
                                      "pallot_v": 0, "pallot_h": 0, "odotettu": 0.0})
            d["erat"] += 1; d["voitot"] += voitti
            d["pallot_v"] += pv; d["pallot_h"] += ph
            tila_w = {p: kayra_viikolla[p][e["viikko"]] for p in joukkue + vast}
            d["odotettu"] += pallo_p(joukkue, vast, tila_w) * (pv + ph)
    for d in parit.values():
        n = d["pallot_v"] + d["pallot_h"]
        d["pallo_osuus"] = d["pallot_v"] / n
        d["odotettu_osuus"] = d["odotettu"] / n
        del d["odotettu"]

    # Kaikki 2 vs 2 -asetelmat nykyisillä ratingeilla
    asetelmat = {}
    for pari_a in itertools.combinations(pelaajat, 2):
        for pari_b in itertools.combinations(pelaajat, 2):
            if set(pari_a) & set(pari_b) or pari_a > pari_b:
                continue
            asetelmat[" & ".join(pari_a) + " | " + " & ".join(pari_b)] = \
                ennuste(list(pari_a), list(pari_b), tila, gamma, nyt + 1)

    return {
        "luotu": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "pelaajat": pelaajat,
        "viikot": [{"viikko": w, "maanantai": viikko_pvm[w].isoformat(),
                    "pelipaivat": sorted(d.isoformat() for d in viikko_pelipvm[w])} for w in viikot],
        "gamma": gamma, "log_evidenssi": le,
        "evidenssit": [{"gamma": g, "log_evidenssi": v} for g, v in evidenssit],
        "beta": BETA, "mu0": MU0, "sigma0": SIGMA0,
        "ranking": ranking,
        "kayrat": {p: [{"viikko": w, "mu": jalkeen[w][p][1], "sigma": jalkeen[w][p][2]}
                       for w in viikot if p in jalkeen[w]] for p in pelaajat},
        "erat": eralista,
        "parit": sorted(parit.values(), key=lambda d: (-d["erat"], -d["pallo_osuus"])),
        "asetelmat": asetelmat,
        "logit": {k: v for k, v in logit.items() if k not in ("theta", "se")},
        "kalman": kalm,
        "yhteenveto": {"erat": len(erat), "pallot": sum(e["pa"] + e["pb"] for e in erat),
                       "viikkoja": len(viikot)},
    }


def tulosta(tulos, varoitukset):
    y = tulos["yhteenveto"]
    print(f"{y['erat']} erää, {y['pallot']} palloa, {y['viikkoja']} viikkoa")
    for v in varoitukset:
        print("  varoitus:", v)
    print("\nγ-valinta (log-evidenssi):")
    for d in tulos["evidenssit"]:
        merkki = " <" if d["gamma"] == tulos["gamma"] else ""
        print(f"  γ={d['gamma']:<6} {d['log_evidenssi']:.3f}{merkki}")
    print(f"\n{'Pelaaja':<12}{'μ':>7}{'σ':>7}{'μ−3σ':>8}{'erät':>6}{'V':>4}{'pallo%':>8}{'θ logit':>9}{'SE':>7}")
    for r in tulos["ranking"]:
        pall = r["pallot_v"] / (r["pallot_v"] + r["pallot_h"]) * 100
        print(f"{r['pelaaja']:<12}{r['mu']:>7.3f}{r['sigma']:>7.3f}{r['konservatiivinen']:>8.2f}"
              f"{r['erat']:>6}{r['voitot']:>4}{pall:>7.1f}%{r['theta']:>9.3f}{r['theta_se']:>7.3f}")
    l = tulos["logit"]
    k = tulos["kalman"]
    pp = k["pp_per_yksikko"]
    print(f"\nKalman: γ = {k['gamma']}, η = {k['eta']}, τ = {k['tau']}  (taito TTT-asteikolla, jaksaminen pp/erä)")
    for r in tulos["ranking"]:
        d = k["kutistamaton"][r["pelaaja"]][-1]
        print(f"  {r['pelaaja']:<12}taito {d['taito']:+.3f} ± {d['taito_sd']:.3f} (TTT {r['mu']:+.3f})   "
              f"jaksaminen {d['jaksaminen'] * pp:+.1f} ± {1.96 * d['jaksaminen_sd'] * pp:.1f}")
    print(f"\nBinomi-logit: Pearson χ²/df = {l['hajonta']:.2f} (df {l['df']})"
          + (" → quasi-binomi-SE:t" if l["quasi"] else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="paikallinen CSV (oletus: hae Google Sheetsistä)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--json", help="kirjoita myös laskennan tulokset JSONina")
    args = ap.parse_args()

    erat, varoitukset = jasenna(lue_csv(args.csv))
    if not erat:
        sys.exit("Ei eriä datassa.")
    try:
        valimuisti = json.loads(VALIMUISTI.read_text())
    except (OSError, ValueError):
        valimuisti = {}
    tulos = laske(erat, valimuisti)
    VALIMUISTI.parent.mkdir(parents=True, exist_ok=True)
    VALIMUISTI.write_text(json.dumps(valimuisti))
    tulosta(tulos, varoitukset)

    data = json.dumps(tulos, ensure_ascii=False, default=str).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", data)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"\nRaportti: {out}")
    if args.json:
        Path(args.json).write_text(json.dumps(tulos, ensure_ascii=False, indent=1, default=str),
                                   encoding="utf-8")


if __name__ == "__main__":
    main()
