# Clausia — Assistant réglementaire LLM + RAG pour Solvabilité II et IFRS 17

Assistant de recherche réglementaire qui répond en langage naturel (en anglais) à des questions sur **Solvabilité II** (régime prudentiel européen des assureurs) et **IFRS 17** (norme comptable des contrats d'assurance). Chaque réponse s'appuie sur des passages réglementaires récupérés dans les textes officiels et cite des **références vérifiables** : article ou paragraphe, et version consolidée du texte.

> Outil d'aide à la recherche uniquement. Il ne remplace ni la lecture des textes officiels, ni un avis juridique, actuariel ou comptable.

**Auteures :** Yoldez Boubahri, Chayma Bdiri

<!-- Capture de l'interface : ajoute docs/clausia_interface.png puis décommente la ligne suivante
![Interface Clausia](docs/clausia_interface.png)
-->

---

## Sommaire

1. [Pourquoi ce projet](#pourquoi-ce-projet)
2. [Architecture](#architecture)
3. [Corpus](#corpus)
4. [Choix techniques](#choix-techniques)
5. [Structure du dépôt](#structure-du-dépôt)
6. [Installation et utilisation](#installation-et-utilisation)
7. [Évaluation et résultats](#évaluation-et-résultats)
8. [Fiabilité et sécurité](#fiabilité-et-sécurité)
9. [Limites connues](#limites-connues)
10. [Perspectives](#perspectives)
11. [Licences et sources](#licences-et-sources)

---

## Pourquoi ce projet

Solvabilité II et IFRS 17 reposent sur des textes techniques, structurés et qui évoluent (la réforme de Solvabilité II s'applique à partir du 30/01/2027). Les professionnels doivent retrouver vite la bonne disposition, vérifier son contexte et justifier leurs conclusions par des références précises.

Un LLM seul invente facilement des numéros d'articles ou mélange les versions. Clausia combine donc une recherche documentaire sur un corpus identifié avec un LLM contraint de ne répondre qu'à partir des sources fournies, puis vérifie ses citations après coup.

**Utilisateurs visés :** actuaires juniors, analystes reporting, auditeurs.

---

## Architecture

```
HORS LIGNE   sources.yaml → ingestion (HTML EUR-Lex, manifeste SHA-256) → parsing structurel
             → unités traçables (paragraphe d'article / paragraphe IFRS 17 + métadonnées)
             → segmentation (chunking) → index FAISS + BM25

EN LIGNE     question → routeur (cadre + version du texte) → recherche hybride (dense + BM25, RRF)
             → reranking cross-encoder → seuil d'abstention → passages parents [S1..Sn]
             → LLM avec prompt strict → vérificateur de citations → interface Streamlit
```

Les composants clés :

- **Unités traçables** : le plus petit passage citable (un paragraphe numéroté d'un article, un paragraphe IFRS 17, une définition). Chaque unité garde sa hiérarchie, sa version et son statut (`in_force` ou `upcoming`).
- **Recherche hiérarchique** : on cherche sur de petits segments (paragraphes) et on envoie au LLM le passage parent (article ou section) pour garder le contexte.
- **Routeur** : détecte le cadre (Solvabilité II, IFRS 17 ou comparaison) et la version visée (en vigueur ou réforme 2027) à partir de la question.
- **Abstention** : si le meilleur score de reranking est sous un seuil calibré sur le jeu de développement, le système refuse de répondre au lieu d'inventer.
- **Vérificateur de citations** : signale les sources `[Sx]` inexistantes et les articles mentionnés mais jamais récupérés.

---

## Corpus

| Document | Identifiant EUR-Lex | Versions |
|---|---|---|
| Directive 2009/138/CE (Solvabilité II) | `32009L0138` | en vigueur (consolidation du 17/01/2025) + version applicable le 30/01/2027 |
| Règlement délégué (UE) 2015/35 | `32015R0035` | configuré dans `sources.yaml` (voir [Limites connues](#limites-connues)) |
| Règlement (UE) 2023/1803 (IFRS 17 tel qu'adopté dans l'UE) | `32023R1803` | en vigueur |

**Corpus parsé lors de la dernière exécution :** 2 400 unités, 739 dispositions distinctes, 2 381 segments, 794 passages parents. L'index final pèse 13,9 Mo (2 381 vecteurs de dimension 768).

**Choix du HTML plutôt que du PDF :** le HTML d'EUR-Lex conserve la structure logique (articles, paragraphes, annexes) et n'a ni en-têtes de page ni césures. Contrepartie : pas de numéros de page, on cite donc au niveau article ou paragraphe, ce qui est plus précis pour un texte juridique.

---

## Choix techniques

| Composant | Choix | Alternatives comparées |
|---|---|---|
| Embeddings | `BAAI/bge-base-en-v1.5` (défaut) | `intfloat/e5-base-v2`, `all-MiniLM-L6-v2` |
| Index vectoriel | FAISS `IndexFlatIP` (recherche exacte) | — |
| Recherche lexicale | BM25 | — |
| Fusion | Reciprocal Rank Fusion (RRF) | dense seule, BM25 seule |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `bge-reranker-base` (perspective) |
| LLM | Gemini (`gemini-flash-lite-latest`), température 0 | Mistral, Groq, modèle local (`Qwen2.5-1.5B-Instruct`) |
| Juge d'évaluation | Un modèle Gemini différent du générateur | — |
| Interface | Streamlit | — |

**Stratégies de segmentation comparées :** `fixed` (fenêtre glissante de 220 mots, référence de base), `paragraph_noheader`, `paragraph` (avec en-tête contextuel), `article` et `hierarchical` (enfants pour la recherche, parents pour le LLM).

Les appels au LLM sont mis en cache sur disque (clé = hash du fournisseur, du modèle, des messages et des paramètres) : relancer le notebook ne consomme pas de quota et reste reproductible.

---

## Structure du dépôt

```
<notebook>.ipynb   notebook unique : toutes les phases, explications et expériences
config/            sources.yaml (documents, date de référence), settings.yaml (modèles, chunking, LLM)
src/regrag/        ingestion/ parsing/ chunking/ retrieval/ generation/ evaluation/ pipeline.py
app/               streamlit_app.py
eval/              questions.jsonl (56 questions, découpe dev/test), manual_review.csv
reports/           résultats CSV et figures
tests/             tests du parseur, de traçabilité du corpus, du routeur et du vérificateur de citations
Dockerfile         image de l'application (l'index est monté, pas embarqué)
```

Le code réutilisable est écrit dans `src/regrag/` par le notebook (cellules `%%writefile`). L'application et les tests importent ainsi exactement le même code que le notebook.

---

## Installation et utilisation

### Option 1 — Google Colab (recommandée)

1. Ouvre le notebook dans Colab et choisis un runtime GPU (T4).
2. Ajoute la clé API de ton fournisseur dans les *Secrets* Colab, par exemple `GEMINI_API_KEY` (ou `MISTRAL_API_KEY`, `GROQ_API_KEY`).
3. Exécute les sections dans l'ordre. Les fichiers sont stockés dans `MyDrive/regulatory-rag/`.

Si EUR-Lex bloque les téléchargements automatiques (page anti-robot), le notebook essaie l'accès machine *Cellar* de l'Office des publications. En dernier recours, il explique comment enregistrer la page avec ton navigateur puis la téléverser.

### Option 2 — Application en local

Une fois l'index `data/index/final` construit par le notebook :

```bash
pip install -r requirements-app.txt
export GEMINI_API_KEY=...        # PowerShell : $env:GEMINI_API_KEY="..."
streamlit run app/streamlit_app.py
```

L'interface affiche la réponse structurée, le cadre et les versions détectés, les sources avec lien EUR-Lex, les extraits dépliables (avec avertissement si la version n'est pas encore applicable) et les alertes du vérificateur de citations.

### Option 3 — Docker

```bash
docker build -t clausia .
docker run -p 8501:8501 -e GEMINI_API_KEY=... -v "$(pwd)/data:/app/data" clausia
```

### Tests

```bash
python -m pytest -q tests/
```

---

## Évaluation et résultats

**Jeu de questions :** 56 questions réparties en 7 catégories (factuelles, multi-passages, comparatives, termes techniques, hors corpus, ambiguës, dépendantes de la version), découpées en développement (14) et test (42). Les références attendues sont annotées au niveau article (Solvabilité II) ou paragraphe (IFRS 17), indépendamment de la stratégie de segmentation. Les configurations sont choisies sur le jeu de développement et rapportées sur le jeu de test.

**Systèmes comparés :** `no_rag` (LLM seul), `simple_rag` (segments fixes, recherche dense, top-5) et `improved_rag` (hybride + reranker + routeur + contexte hiérarchique + garde-fous).

### Recherche (jeu de test)

| Configuration | Recall@5 | Recall@10 | MRR | nDCG@10 | Latence p50 |
|---|---|---|---|---|---|
| `paragraph` · dense | 0,680 | 0,776 | 0,691 | 0,659 | 1,3 ms |
| `paragraph` · hybride + reranker | 0,675 | 0,789 | 0,672 | 0,674 | 201,7 ms |
| `paragraph` · hybride + reranker + routeur | 0,669 | 0,792 | 0,678 | 0,672 | 202,4 ms |
| `article` · dense | 0,654 | 0,728 | 0,635 | 0,606 | 1,2 ms |
| `fixed` · dense (référence) | 0,533 | 0,625 | 0,476 | 0,483 | 1,2 ms |
| `fixed` · BM25 | 0,566 | 0,664 | 0,408 | 0,431 | 2,0 ms |

Ce qu'on en retient : la segmentation par paragraphe avec en-tête contextuel bat nettement la fenêtre fixe (MRR 0,69 contre 0,48 en recherche dense). Le reranker et le routeur améliorent un peu le rappel à 10 mais coûtent environ 200 ms par requête. Les résultats complets sont dans `reports/`.

### Génération (indicatif)

| Système | Correction (juge) | Fidélité (juge) | Mentions non supportées / réponse | Abstention correcte | Fausse abstention |
|---|---|---|---|---|---|
| `no_rag` | 0,750 | n/a | 2,67 | 0,0 | 0,0 |
| `simple_rag` | 1,000 | 1,000 | 0,10 | 0,0 | 0,05 |
| `improved_rag` | 0,867 | 0,917 | 0,02 | 1,0 | 0,11 |

**À lire avec prudence :** seules 32 des 126 réponses ont été notées par le juge (quotas de l'API gratuite), aucune revue humaine n'a encore calibré le juge, et les questions du jeu de référence ne sont pas toutes vérifiées manuellement. Ces chiffres montrent surtout que `improved_rag` cite plus précisément (précision des citations 0,36 contre 0,18 pour `simple_rag`), invente beaucoup moins de dispositions que `no_rag` et sait s'abstenir sur les questions hors corpus. Ils ne permettent pas de conclure sur la correction globale.

### Performance opérationnelle

Latence de bout en bout médiane d'environ 7 s par requête (p95 d'environ 28 s, dominé par le LLM), recherche d'environ 284 ms pour `improved_rag`, mémoire du processus d'environ 2,3 Go. Coût nul avec le plan gratuit, sous réserve de quotas.

---

## Fiabilité et sécurité

| Risque | Contrôle mis en place | Limite résiduelle |
|---|---|---|
| Hallucination | Prompt strict (sources uniquement), température 0, juge de fidélité | Une paraphrase infidèle reste possible |
| Citation incorrecte | Vérificateur post-génération (source inexistante, article jamais récupéré) | Ne vérifie pas que la source soutient la phrase |
| Version obsolète | Métadonnées de statut, filtre « en vigueur » par défaut, avertissement dans l'interface | Le routeur de version repose sur des mots-clés |
| Corpus incomplet | Abstention sous le seuil | Seuil calibré sur peu d'exemples |
| Injection d'instructions | Sources balisées comme données, règle dédiée du prompt, test ciblé (injection ignorée) | Aucune défense n'est garantie à 100 % |
| Licence | `data/` exclu de Git, usage personnel, démo non publique | Ne pas déployer publiquement avec le texte IFRS 17 |

---

## Limites connues

- **Règlement délégué 2015/35 :** il est configuré dans `sources.yaml`, mais lors de la dernière exécution il n'apparaît pas dans le corpus parsé. Le test `test_expected_provisions_present` échoue (articles 1, 37 et 39 absents) et 9 références attendues du jeu de questions n'ont pas été retrouvées, la plupart pointant vers ce règlement. Les résultats ci-dessus portent donc surtout sur la directive et sur IFRS 17.
- Corpus limité aux niveaux 1 et 2 : pas d'orientations ni de Q&R de l'EIOPA, pas de droit national.
- Parseur HTML heuristique.
- Jeu de test réduit et annoté par une seule personne ; juge automatique imparfait et partiellement exécuté.
- Quotas des API gratuites.

---

## Perspectives

Ajout des orientations et Q&R de l'EIOPA, corpus en français, reranker plus puissant (`bge-reranker-base`), routeur de version appris, vérification phrase par phrase (NLI), jeu de test élargi avec double annotation, correction du parsing du règlement délégué.

---

## Licences et sources

- Les contenus d'EUR-Lex sont réutilisables avec mention de la source.
- Les normes IFRS sont © IFRS Foundation : reproduction autorisée dans l'EEE, usage personnel ailleurs. **Les textes bruts ne sont donc pas dans ce dépôt** (`data/` est ignoré par Git) : le notebook les télécharge.
- Sources officielles : [Directive 2009/138/CE](https://eur-lex.europa.eu/eli/dir/2009/138/oj), [Règlement délégué (UE) 2015/35](https://eur-lex.europa.eu/eli/reg_del/2015/35/oj), Règlement (UE) 2023/1803.
- Projet académique, non commercial.
