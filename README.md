# Netflix Global Content Analysis with Focus on Indian OTT Market Trends

### MSc Data Science Ireland 2027 Application Project
**Author:** Drijil Pandit  
**Location:** CBD Belapur, Navi Mumbai, India  
**Target Universities:** University College Dublin (UCD) / Trinity College Dublin (TCD)

---

### 🎯 Project Objective

This project analyzes 8,808 global Netflix titles from 2008-2021 to identify content trends, with specialized focus on 1,046 Indian productions. The study demonstrates data science skills in regional market segmentation, cross-cultural content analysis, and predictive modeling.

**Relevance to Ireland**: The techniques developed for Indian market analysis will be extended at UCD/TCD to compare Irish and Indian OTT consumption patterns, supporting research on multicultural recommendation systems for European streaming platforms.

---

### 📊 Dataset Overview

| Metric | Value | Details |
| --- | --- | --- |
| **Total Titles** | 8,808 | Movies + TV Shows across 200+ countries |
| **Indian Content** | 1,046 titles | 11.9% of global catalog, primary focus of analysis |
| **Time Period** | 2008 - 2021 | Covers Netflix India launch and OTT boom |
| **Features** | 12 columns | type, title, director, cast, country, date_added, release_year, rating, duration, listed_in, description |

**Data Source**: [Kaggle - Netflix Movies and TV Shows Dataset](https://www.kaggle.com/datasets/shivamb/netflix-shows)  
**License**: CC0 Public Domain

---

### 🔍 Key Findings – Indian Market Analysis

1. **Content Boom**: India produced 203 titles in 2019, representing 3.2x growth from 2015, aligned with Jio digital expansion
2. **Format Preference**: Indian catalog shows 68% Movies vs 32% TV Shows, compared to 50-50 global average
3. **Talent Analysis**: Anupam Kher appears in 41 titles, highest for any Indian actor on Netflix
4. **Genre Concentration**: Drama and International Movies constitute 45% of Indian content
5. **Rating Distribution**: 72% of Indian content is rated TV-MA or TV-14, indicating mature audience targeting

---

### 🛠️ Technical Implementation

**Technologies**: Python 3.10, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

**Methodology**:
1. **Data Cleaning**: Handled 10.3% missing director values, 1.1% missing country data, standardized date formats
2. **Exploratory Data Analysis**: Country-wise distribution, temporal trends, content rating analysis
3. **Indian Market Segmentation**: Isolated and analyzed 1,046 Indian titles using `.str.contains('India')` filtering
4. **Predictive Modeling**: Random Forest classifier to predict content type (Movie vs TV Show) based on duration, rating, and genre – Achieved 82.3% accuracy

```python
# Core Analysis: Indian Content Segmentation
import pandas as pd

df = pd.read_csv('data/netflix_titles.csv')
indian_content = df[df['country'].str.contains('India', na=False)]

print(f"Total Indian Titles: {len(indian_content)}")
print(f"Indian Movies: {len(indian_content[indian_content['type']=='Movie'])}")
print(f"Indian TV Shows: {len(indian_content[indian_content['type']=='TV Show'])}")
