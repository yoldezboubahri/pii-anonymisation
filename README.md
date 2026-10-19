# Détection de PII et anonymisation de textes

Détection automatique des **informations personnelles identifiantes (PII)** dans du texte libre, puis **anonymisation** en remplaçant chaque PII par son type.

> « Bonjour, je suis **Flen Fouleni**, joignable au **28 999 547** »
> → « Bonjour, je suis `[GIVENNAME]` `[LASTNAME]`, joignable au `[TEL]` »

**Dataset :** [`ai4privacy/pii-masking-300k`](https://huggingface.co/datasets/ai4privacy/pii-masking-300k)

---

## Contexte et problématique

Les textes produits par les organisations (e-mails, tickets de support, dossiers patients, formulaires…) contiennent très souvent des PII. Le **RGPD** impose de les protéger dès qu'un texte est partagé, archivé ou utilisé pour entraîner un modèle. L'anonymisation manuelle ne passe pas à l'échelle : il faut un outil automatique.

> **Peut-on détecter et masquer automatiquement les PII dans des textes, et quel gain un Transformer fine-tuné apporte-t-il par rapport à une approche classique ?**

## Approches comparées

| # | Approche | Rôle |
|---|---|---|
| 1 | **TF-IDF + SVM linéaire** | approche classique |
| 2 | **Transformer fine-tuné** (encodeur pré-entraîné) | approche Deep Learning |

## Plan du notebook

| Partie | Contenu |
|---|---|
| I | Introduction et problématique |
| II | Bref état de l'art |
| III | Données et préparation (source, exploration, preprocessing) |
| IV | Méthodologie (formulation, métriques) |
| V | Implémentation (TF-IDF + SVM, Transformer) |
| VI | Protocole expérimental |
| VII | Résultats |
| VIII | Interprétation et analyse critique |
| IX | Limites et perspectives |
| X | Conclusion |
| XI | Références |

## Structure du dépôt

```
<notebook>.ipynb   notebook principal (toutes les parties ci-dessus)
requirements.txt   dépendances Python
.gitignore
README.md
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # PowerShell : .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Lance ensuite Jupyter et exécute le notebook dans l'ordre :

```bash
jupyter notebook
```

Le dataset est téléchargé automatiquement depuis Hugging Face au premier lancement. Un GPU est recommandé pour le fine-tuning du Transformer (Google Colab convient).

## Résultats

_À compléter avec les métriques du notebook (précision, rappel, F1 par type de PII, comparaison TF-IDF + SVM / Transformer)._

## Limites

_À compléter à partir de la partie IX du notebook._

## Références

- Dataset : [ai4privacy/pii-masking-300k](https://huggingface.co/datasets/ai4privacy/pii-masking-300k)
- RGPD : Règlement (UE) 2016/679

**Auteure :** Yoldez Boubahri
