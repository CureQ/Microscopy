# Protein Aggregate Analyzer

Een desktopapplicatie voor de detectie en kwantificering van eiwitaggregaten in confocale microscopiebeelden. Gebouwd met Python en PyQt5, met ondersteuning voor klassieke beeldverwerking én deep-learning segmentatie.

---

## Inhoudsopgave

1. [Overzicht](#overzicht)
2. [Systeemvereisten](#systeemvereisten)
3. [Installatie](#installatie)
4. [Opstarten](#opstarten)
5. [Functionaliteit per tabblad](#functionaliteit-per-tabblad)
6. [Ondersteunde bestandsformaten](#ondersteunde-bestandsformaten)
7. [Deep Learning — modelbestanden](#deep-learning--modelbestanden)
8. [Veelvoorkomende problemen](#veelvoorkomende-problemen)
9. [Projectstructuur](#projectstructuur)
10. [Licentie](#licentie)

---

## Overzicht

De Protein Aggregate Analyzer biedt een complete workflow voor microscopische beeldanalyse:

- **Inladen** van confocale beeldstacks (.lif, .tif/.tiff)
- **Pre-processing**: ruisonderdrukking, contrastverbetering en achtergrondcorrectie
- **Cellichamen detectie** via Cellpose (optioneel)
- **Segmentatie** van eiwitaggregaten via een Deep Learning Ensemble-methode
- **Correctie** van detectieresultaten aan de hand van handmatige ground-truth maskers
- **Validatie** met kwantitatieve maten: F1/Dice, IoU, precisie en recall

De interface heeft een professioneel donker thema en is volledig Nederlandstalig.

---

## Systeemvereisten

| Onderdeel        | Minimaal                  | Aanbevolen                    |
|-----------------|---------------------------|-------------------------------|
| **Besturingssysteem** | Windows 10, macOS 12, Ubuntu 20.04 | Windows 11 / Ubuntu 22.04 |
| **Python**       | 3.8                       | 3.10 of 3.11                  |
| **RAM**          | 8 GB                      | 16 GB of meer                 |
| **GPU**          | Niet vereist              | NVIDIA GPU met CUDA (voor DL) |
| **Schermresolutie** | 1280 × 800             | 1920 × 1080 of hoger          |

---

## Installatie

### Stap 1 — Python omgeving aanmaken (aanbevolen)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Stap 2 — Verplichte pakketten installeren

```bash
pip install -r requirements.txt
```

Dit installeert automatisch:

- `PyQt5` — GUI-framework
- `numpy` — numerieke berekeningen
- `scipy` — wetenschappelijke berekeningen en beeldfilters
- `scikit-image` — beeldverwerkingsalgoritmen
- `matplotlib` — visualisatie en plotweergave in de interface
- `tifffile` — lezen en schrijven van TIFF-beeldstacks

### Stap 3 — Optionele pakketten (indien gewenst)

**Leica .LIF bestanden:**

```bash
pip install readlif
```

**Deep Learning segmentatie (Tab 4 — Ensemble):**

```bash
# CPU-versie (eenvoudig, trager)
pip install torch torchvision segmentation-models-pytorch

# GPU-versie met CUDA (sneller, vereist NVIDIA GPU)
# Ga naar https://pytorch.org/get-started/locally/ voor de juiste opdracht.
# Voorbeeld voor CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install segmentation-models-pytorch
```

**Cellpose celdetectie (Tab 3):**

```bash
pip install cellpose
```

---

## Opstarten

```bash
python Gecorrigeerd_tabje_werkend.py
```

Bij het starten verschijnt de melding in de statusbalk:

> *Welkom! — Open een .LIF of .TIF bestand via 📂 Open.*

---

## Functionaliteit per tabblad

### Tab 1 — Viewer

Laad en bekijk confocale beeldstacks.

- Ondersteuning voor meerdere kanalen en Z-lagen
- Weergave van maximale projectie of individuele Z-slices
- Metadatapanel: pixelgrootte, bitdiepte, kanaalnamen
- Exporteer huidige weergave als afbeelding

### Tab 2 — Pre-processing

Verbeter de beeldkwaliteit voordat segmentatie plaatsvindt.

- **Ruisonderdrukking**: Gaussiaans filter, bilateraal filter
- **Contrastverbetering**: histogram-equalisatie, CLAHE
- **Achtergrondcorrectie**: white top-hat transformatie
- **Drempelwaarden**: Otsu, multi-Otsu, lokale drempelwaarden
- Voorbeeldweergave van het resultaat in real-time

### Tab 3 — Cellichamen

Detecteer cellichamen als regio's van interesse (ROI).

- Segmentatie via **Cellpose** (vereist optionele installatie)
- Morfologische nabewerking: verwijdering van kleine objecten, watershed
- Resultaten worden doorgegeven aan Tab 4 en Tab 5

### Tab 4 — Deep Learning (Ensemble)

Detecteer eiwitaggregaten met een ensemble van deep-learning modellen.

- Laad meerdere getrainde modelgewichten (`model_fold0.pth`, `model_fold1.pth`, …)
- **Test-Time Augmentation (TTA)** voor robuustere voorspellingen
- Instelbare drempelwaarde en minimaal objectoppervlak (px²)
- Keuze van rekeneenheid: `cpu` of `cuda`
- Voortgangsbalk en statusmeldingen tijdens inferentie

> **Let op:** Tab 4 vereist `torch` en `segmentation-models-pytorch`.  
> Zonder deze pakketten is de tab uitgeschakeld.

### Tab 5 — Correctie

Vergelijk DL-detecties met handmatige ground-truth maskers en corrigeer fouten.

- Laad een ground-truth masker (binair TIF)
- Detecties worden automatisch ingekleurd:
  - **Groen** — overlapt met ground-truth (correct)
  - **Rood** — overlapt niet (fout-positief)
- Klik op een rode regio om deze handmatig goed te keuren
- Stuur het gecorrigeerde masker door naar Tab 6

### Tab 6 — Validatie

Kwantificeer de segmentatieprestaties.

- Vergelijkt het DL-resultaat (of gecorrigeerd masker vanuit Tab 5) met de ground-truth
- Berekende maten:
  - **F1-score / Dice**
  - **IoU** (Intersection over Union)
  - **Precisie** en **Recall**
- Exporteer resultaten als CSV

---

## Ondersteunde bestandsformaten

| Formaat     | Extensie          | Vereist pakket |
|------------|-------------------|----------------|
| TIFF-stack  | `.tif`, `.tiff`   | `tifffile` (verplicht) |
| Leica LIF   | `.lif`            | `readlif` (optioneel)  |

---

## Deep Learning — modelbestanden

De Ensemble-methode verwacht **meerdere PyTorch-modelbestanden** in één map, benoemd volgens het patroon:

```
model_fold0.pth
model_fold1.pth
model_fold2.pth
...
```

Selecteer de map met de modelbestanden via de mapkiezer in Tab 4. De applicatie laadt automatisch alle bestanden die voldoen aan het naampatroon `model_fold*.pth`.

> Heb je zelf modellen getraind? Sla ze op met `torch.save(model.state_dict(), "model_foldN.pth")`.

---

## Veelvoorkomende problemen

| Probleem | Oorzaak | Oplossing |
|----------|---------|-----------|
| **Te veel detecties** | Drempelwaarde te laag | Verhoog de drempelwaarde of vergroot het minimale objectoppervlak (px²) in Tab 4 |
| **Trage segmentatie** | TTA ingeschakeld of CPU-modus | Schakel TTA uit, of stel Device in op `cuda` als je een NVIDIA GPU hebt |
| **CUDA-fout bij opstarten** | GPU niet beschikbaar / fout driver | Stel Device in op `cpu` in Tab 4 |
| **Geen modellen gevonden** | Onjuiste bestandsnamen of map | Controleer of de bestanden de naam `model_fold0.pth` e.v. hebben en de juiste map is geselecteerd |
| **`.lif` bestand niet herkend** | `readlif` niet geïnstalleerd | Voer `pip install readlif` uit en herstart de applicatie |
| **Tab 4 uitgeschakeld** | `torch` niet geïnstalleerd | Installeer PyTorch via `pip install torch torchvision` |
| **Cellpose-tab werkt niet** | `cellpose` niet geïnstalleerd | Voer `pip install cellpose` uit en herstart |

---

## Projectstructuur

```
project/
│
├── Gecorrigeerd_tabje_werkend.py   # Hoofdscript — start de applicatie
├── requirements.txt                # Pakketvereisten
├── README.md                       # Deze handleiding
│
└── models/                         # (optioneel) map voor DL-modelgewichten
    ├── model_fold0.pth
    ├── model_fold1.pth
    └── ...
```

---

## Licentie

Dit project is ontwikkeld voor intern gebruik binnen het confocale microscopieteam. Neem contact op met de ontwikkelaar voor vragen over hergebruik of distributie.
