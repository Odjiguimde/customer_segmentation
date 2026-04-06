# ◈ Customer Segmentation — Clustering Lab

> Application interactive de segmentation clients par clustering non-supervisé, avec réduction de dimensions et visualisations avancées.

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)]([https://rtr2eednbjtntbyv6uzxze.streamlit.app/])
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=white)](https://plotly.com)

---

## 📌 Contexte

Ce projet s'inscrit dans le cadre d'une problématique marketing :

> **Comment segmenter les clients en fonction de leurs comportements d'achat ?**

La segmentation client permet d'identifier des groupes homogènes dans une base de données, d'adapter les stratégies marketing, et de mieux comprendre les profils d'acheteurs. L'application combine des algorithmes de clustering classiques avec des techniques modernes de réduction de dimensions pour rendre les résultats interprétables et visuellement riches.

---

## 🗂 Dataset

**Source :** [Customer Segmentation — Kaggle](https://www.kaggle.com/datasets/kaushiksuresh147/customer-segmentation)

| Variable | Type | Description |
|---|---|---|
| `Gender` | Catégoriel | Genre du client |
| `Ever_Married` | Catégoriel | Marié ou non |
| `Age` | Numérique | Âge du client |
| `Graduated` | Catégoriel | Diplômé ou non |
| `Profession` | Catégoriel | Secteur d'activité |
| `Work_Experience` | Numérique | Années d'expérience |
| `Spending_Score` | Catégoriel | Score de dépense (Low / Average / High) |
| `Family_Size` | Numérique | Taille du foyer |
| `Segmentation` | Catégoriel | Segment cible (A / B / C / D) |

> Si vous n'avez pas accès au dataset Kaggle, l'application inclut un **dataset synthétique de 2 000 clients** prêt à l'emploi.

---

## 🚀 Installation & Lancement local

### Prérequis

- Python 3.10 ou supérieur
- pip

### Étapes

```bash
# 1. Cloner le dépôt
git clone https://github.com/TON_USER/customer-segmentation.git
cd customer-segmentation

# 2. (Optionnel) Créer un environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer l'application
streamlit run app.py
```

L'application s'ouvre automatiquement sur [http://localhost:8501](http://localhost:8501).

---

## ☁️ Déploiement sur Streamlit Cloud

1. **Pousser le projet sur GitHub** (repo public ou privé)

```bash
git init
git add .
git commit -m "feat: customer segmentation app"
git remote add origin https://github.com/TON_USER/customer-segmentation.git
git push -u origin main
```

2. **Aller sur [share.streamlit.io](https://share.streamlit.io)**

3. Cliquer sur **"New app"** et remplir :
   - Repository : `TON_USER/customer-segmentation`
   - Branch : `main`
   - Main file path : `app.py`

4. Cliquer **"Deploy"** — l'app sera en ligne en ~2 minutes ✅

---

## 🧠 Algorithmes implémentés

### Clustering

| Algorithme | Description | Paramètres |
|---|---|---|
| **KMeans** | Partitionnement en k groupes par minimisation de l'inertie | `k`, initialisation (`k-means++` / `random`) |
| **CAH** | Clustering Agglomératif Hiérarchique (bottom-up) | `k`, méthode de linkage (`ward`, `complete`, `average`, `single`) |
| **DBSCAN** | Clustering par densité, détecte les outliers automatiquement | `eps` (rayon), `min_samples` |

### Réduction de dimensions

| Méthode | Description | Usage |
|---|---|---|
| **PCA** | Analyse en Composantes Principales — linéaire | Visualisation rapide, analyse des loadings |
| **t-SNE** | t-distributed Stochastic Neighbor Embedding — non-linéaire | Visualisation de structures complexes |

Toutes les visualisations sont disponibles en **2D et 3D**.

---

## 📊 Fonctionnalités de l'application

```
Sidebar
├── Chargement des données (CSV ou dataset démo)
├── Sélection des variables (features)
├── Choix de l'algorithme + hyperparamètres
└── Choix de la réduction de dimensions

Onglet 1 — Visualisation
├── Scatter plot 2D/3D des clusters (interactif)
├── Distribution / taille des clusters (bar chart)
└── Métriques de qualité (Silhouette, Calinski-Harabasz, Davies-Bouldin)

Onglet 2 — Profils clusters
├── Radar chart (profil normalisé par cluster)
├── Heatmap des moyennes normalisées
└── Box plots par variable

Onglet 3 — Méthode du coude
├── Courbe d'inertie (k=2..10)
└── Score de Silhouette par k + recommandation automatique

Onglet 4 — Dendrogramme / PCA
├── Dendrogramme CAH (échantillon 200 pts)
├── Variance expliquée cumulée (PCA)
└── Loadings PC1 & PC2

Onglet 5 — Données
├── Table résultat filtrable par cluster
└── Export CSV
```

---

## 📐 Métriques de qualité

| Métrique | Interprétation |
|---|---|
| **Silhouette Score** | [-1, 1] — proche de 1 = clusters bien séparés |
| **Calinski-Harabasz** | Plus élevé = meilleure séparation inter/intra clusters |
| **Davies-Bouldin** | Plus faible = clusters plus compacts et distincts |

---

## 🏗 Structure du projet

```
customer_segmentation/
├── app.py                  # Application Streamlit principale
├── requirements.txt        # Dépendances Python
├── .streamlit/
│   └── config.toml         # Thème et configuration Streamlit
└── README.md               # Ce fichier
```

---

## 📦 Dépendances principales

```
streamlit >= 1.32
pandas >= 2.0
numpy >= 1.24
scikit-learn >= 1.3
plotly >= 5.18
scipy >= 1.11
matplotlib >= 3.7
```

---

## 🎓 Compétences mises en œuvre

- **Clustering non supervisé** : KMeans, CAH, DBSCAN
- **Réduction de dimensions** : PCA, t-SNE
- **Évaluation de modèles** : indices de Silhouette, Calinski-Harabasz, Davies-Bouldin
- **Prétraitement** : encodage des variables catégorielles, normalisation (StandardScaler)
- **Visualisation interactive** : Plotly (scatter, radar, heatmap, box plots)
- **Déploiement** : Streamlit Cloud

---

## 📄 Licence

MIT — libre d'utilisation, de modification et de distribution.
