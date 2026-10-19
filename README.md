# Détection de PII et anonymisation de textes français

Détection automatique des **informations personnelles identifiantes (PII)** dans du texte libre français, puis **anonymisation** en remplaçant chaque PII par son type. Le projet compare une approche classique (TF-IDF + SVM linéaire) à un Transformer fine-tuné (XLM-RoBERTa).

> « Bonjour, je m'appelle Sonia Trabelsi et j'habite au 22 rue de la République, 13001 Marseille. Mon numéro est le 07 12 34 56 78. »
> → « Bonjour, je m'appelle `[GIVENNAME]` `[LASTNAME]` et j'habite au `[BUILDING]` `[STREET]`, `[POSTCODE]` `[CITY]`. Mon numéro est le `[TEL]`. »

**Dataset :** [`ai4privacy/pii-masking-300k`](https://huggingface.co/datasets/ai4privacy/pii-masking-300k) (textes synthétiques multilingues, sous-ensemble français)

---

## Contexte et problématique

Les e-mails, tickets de support, dossiers patients ou formulaires contiennent très souvent des PII. Le **RGPD** impose de les protéger dès qu'un texte est partagé, archivé ou utilisé pour entraîner un modèle. L'anonymisation manuelle ne passe pas à l'échelle.

> **Peut-on détecter et masquer automatiquement les PII dans des textes français avec un taux de fuite suffisamment faible, et quel gain un Transformer fine-tuné apporte-t-il par rapport à une approche classique TF-IDF + SVM ?**

## Méthode

La tâche est formulée comme de la **classification de tokens (NER)** avec un schéma **BIO** : 25 types de PII, soit 51 classes (1 × `O` + 25 × `B-` + 25 × `I-`).

```
Texte brut → Tokenisation → Classification → BIO → Reconstruction des spans → Masquage [TYPE]
```

| | Approche 1 : classique | Approche 2 : Transformer |
|---|---|---|
| Représentation | TF-IDF sur caractères et mots du contexte + traits simples (forme, majuscules, chiffres) | Représentations de `xlm-roberta-base` |
| Modèle | SVM linéaire (`SGDClassifier`), schéma IO | XLM-RoBERTa avec tête de classification de tokens |
| Réglage | Hyper-paramètre `alpha` | Nombre d'époques + seuil de confiance τ |

**Évaluation au niveau caractère** (offsets), ce qui rend les deux modèles comparables quelle que soit leur tokenisation. Métriques : précision, rappel, F1, F1 entité exact, F1 macro, et **taux de fuite** (part des données sensibles laissées visibles). L'accuracy seule serait trompeuse, car les PII sont minoritaires dans un texte.

## Données

- **Français uniquement :** 31 447 documents de train et 8 413 documents de test (le split `validation` du dataset sert de test). Le dev représente 10 % du train.
- **Augmentation (train uniquement) :** +4 000 phrases en prose générées à partir de 12 modèles de phrases, avec des valeurs tirées du train, pour mieux représenter des textes naturels et sans fuite vers le dev ou le test.
- **Contrôle qualité :** environ 11 % des documents sans PII, doublons exacts très rares (0,07 % du train), offsets incohérents et entités chevauchantes inférieurs à 1 %.
- **Tokenisation :** une tokenisation simple et une tokenisation « fine » ont été comparées sur la reconstruction des entités (98,75 % contre 98,70 %) ; la tokenisation simple est retenue.

## Hyper-paramètres principaux

| | Valeur |
|---|---|
| Graine | 42 |
| TF-IDF + SVM | `alpha` testé parmi 1e-7, 1e-6, 1e-5 → **1e-6** retenu (F1 entité dev 0,647) |
| Transformer | `xlm-roberta-base`, longueur max 256 (stride 64), 4 époques, lr 3e-5, batch 16 |
| Seuil de confiance τ | testé parmi 0,5 / 0,7 / 0,8 / 0,9 / 0,95 sur le dev (τ = 0,9 utilisé dans la démo) |

## Résultats

| Métrique | TF-IDF + SVM (test) | TF-IDF + SVM (OOD) | Transformer (test) | Transformer (OOD) |
|---|---|---|---|---|
| Précision caractère | 0,952 | 0,988 | 0,965 | 0,957 |
| Rappel caractère | 0,960 | 0,816 | **0,994** | 1,000 |
| **Taux de fuite** | 0,093 | 0,289 | **0,011** | 0,000 |
| Précision entité | 0,626 | 0,455 | 0,875 | 0,895 |
| Rappel entité | 0,758 | 0,526 | 0,956 | 0,895 |
| **F1 entité** | 0,686 | 0,488 | **0,914** | 0,895 |
| F1 macro | 0,648 | 0,478 | 0,878 | 0,876 |
| F1 grossier strict (PER/LOC/DATE) | 0,833 | 0,647 | 0,951 | 0,788 |
| Temps d'inférence | 3,0 ms/doc | n/a | 11,1 ms/doc | n/a |

**Ce qu'on en retient :**

- Le Transformer détecte et délimite mieux les PII : F1 entité de 0,914 contre 0,686, et seulement 1,1 % de fuite sur le test contre 9,3 %.
- Hors distribution, l'approche classique se dégrade nettement (F1 0,488, fuite de 28,9 %), alors que le Transformer reste proche de ses scores sur le test.
- Les écarts sont les plus marqués sur les types difficiles (dates, coordonnées GPS, adresses IP, numéros de sécurité sociale).
- Le TF-IDF + SVM reste environ 3,7 fois plus rapide et beaucoup plus simple à entraîner (158 s pour l'entraînement final).

## Démo

| Source | TF-IDF + SVM | Transformer (τ = 0,9) |
|---|---|---|
| …adresse IP 172.16.254.3. | …adresse IP `172[IP]3`. | …adresse IP `[IP]`. |
| …joignable au 07 12 34 56 78 | …le `[SOCIALNUMBER]` `[IP]` | …le `[TEL]` |
| …à ahmed.benyoussef@gmail.com avant le 30/11/2026 | `[EMAIL]` … `[DATE]` | `[EMAIL]` … `[DATE]` |

L'approche classique commet des erreurs de bornage (IP, morceaux d'adresse), que le Transformer évite.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # PowerShell : .venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter notebook NLP_project.ipynb
```

Le dataset est téléchargé automatiquement depuis Hugging Face au premier lancement. Un **GPU est recommandé** pour le fine-tuning (le notebook active `fp16` si CUDA est disponible) ; Google Colab convient. Pour PyTorch avec GPU en local, suis les instructions de [pytorch.org](https://pytorch.org/get-started/locally/) avant `pip install -r requirements.txt`.

Le modèle fine-tuné est enregistré dans `modele-pii-fr/` (ignoré par Git).

## Structure du dépôt

```
NLP_project.ipynb   notebook unique : données, EDA, preprocessing, deux approches, évaluation, démo
requirements.txt    dépendances Python
.gitignore
README.md
```

Plan du notebook : I Introduction · II État de l'art · III Données et préparation · IV Méthodologie · V Implémentation · VII Résultats · IX Analyse critique, limites et perspectives · X Conclusion · XI Références.

## Limites

- **Données synthétiques**, générées et fortement concentrées en PII, donc moins représentatives de textes réels. Les résultats doivent être validés sur des données authentiques.
- Annotations parfois ambiguës aux frontières des entités.
- Le Transformer demande plus de ressources et de temps d'entraînement que l'approche classique.
- Les tests manuels (démo) restent limités.

## Perspectives

Enrichir le corpus avec des textes réels et multilingues (notamment arabe et tunisien), améliorer la qualité des annotations, mieux détecter les numéros et les adresses, et analyser l'impact du seuil de confiance τ sur la précision et le rappel.

## Références

1. Ai4Privacy (2024). *pii-masking-300k*. Hugging Face.
2. Règlement (UE) 2016/679 (RGPD).
3. Devlin et al. (2019). *BERT*. NAACL.
4. Conneau et al. (2020). *Unsupervised Cross-lingual Representation Learning at Scale* (XLM-RoBERTa). ACL.
5. Lison et al. (2021). *Anonymisation Models for Text Data: State of the Art, Challenges and Future Directions*. ACL.
6. Pilán et al. (2022). *The Text Anonymization Benchmark (TAB)*. Computational Linguistics, 48(4).

La liste complète figure dans la partie XI du notebook.

**Auteure :** Yoldez Boubahri
