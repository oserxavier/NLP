<h2><b>PROJETO NLP COM TÓPICOS EM NOTÍCIAS</b></h2>


**Aluno:** _OSEMAR DA SILVA XAVIER_  
**Data:** _26/05/2025_  

---

<h3><b>RESPOSTAS ÀS COMPETÊNCIA AVALIADAS:</b></h3>

---

<b><h4>1. Qual o endereço do seu notebook (Colab) executado?</b></h4>
**Resposta:**  
https://colab.research.google.com/drive/11TB243H7Y8n9KXhORG-ePMomzutWDUG8?usp=sharing

---

<b><h4>2. Em qual célula está o código que realiza o download dos pacotes necessários para tokenização e stemming usando nltk?</b></h4>
**Resposta:**  
**Célula 5**
```python
import nltk  
nltk.download("punkt")  
nltk.download("rslp")
```

---

<b><h4> 3. Em qual célula está o código que atualiza o Spacy e instala o pacote `pt_core_news_lg`?</b></h4>
**Resposta:**  
**Célula 6**
```python
!python -m spacy download pt_core_news_lg
```

---

<b><h4> 4. Em qual célula está o download dos dados diretamente do Kaggle?</b></h4>
**Resposta:**  
**Célula 2**
```python
!kaggle datasets download --force -d marlesson/news-of-the-site-folhauol
```

---

<b><h4> 5. Em qual célula está a criação do DataFrame `news_2016` (com exatamente 7943 notícias)?</b></h4>
**Resposta:**  
**R.:Célula 10**
```python
df["date"] = pd.to_datetime(df.date)  
news_2016 = df[(df.date.dt.year == 2016) & (df.category == "Mercado")]
```

---

<b><h4> 6. Em qual célula está a função que tokeniza e realiza o stemming dos textos usando funções do NLTK?</b></h4>
**Resposta:**  
**Célula 13**
```python
def tokenize(text: str) -> List:
    tokens = word_tokenize(text.lower())
    stemmer = RSLPStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens if token.isalpha()]
    return stemmed_tokens
```

---

<b><h4> 7. Em qual célula está a função que realiza a lematização usando o Spacy?</b></h4>
**Resposta:**  
**Célula 17**
```python
def lemma(doc):
    return [token.lemma_.lower() for token in doc if filter(token)]
```

---

<b><h4> 8. Qual a diferença entre stemming e lematização? Use 4 exemplos.</b></h4>

**Resposta:**

| Palavra     | Stemming (RSLP) | Lematização (spaCy) |
|-------------|------------------|----------------------|
| comprando   | compr            | comprar              |
| notícias    | notíci           | notícia              |
| estudando   | estud            | estudar              |
| melhores    | melhor           | bom                  |

---

<b><h4> 9. Em qual célula o modelo `pt_core_news_lg` está sendo carregado?</b></h4>
**Resposta:**  
**Célula 19**
```python
nlp = spacy.load("pt_core_news_lg")
```

---

<b><h4> 10. Em qual célula o modelo foi aplicado a todos os textos?</b></h4>
**Resposta:**  
**Célula 20**
```python
news_2016['spacy_doc'] = news_2016['text'].progress_map(nlp)
```

---

<b><h4> 11. Indique a célula onde as entidades dos textos foram extraídas (apenas organizações).</b></h4>
**Resposta:**  
**Célula 21**
```python
def NER(doc):
    return [ent.text for ent in doc.ents if ent.label_ == "ORG"]
```

---

<b><h4> 12. Imagem da nuvem de entidades por tópico.</b></h4>
**Resposta:**  
<div style="display: flex; justify-content: center; align-items: center; height: 100vh;">
    <img src="https://raw.githubusercontent.com/oserxavier/Machine-Learning/refs/heads/main/figures/wordcloud_1.png" height="450" width="450">
</div>

---

<b><h4> 13. Por que usamos TF-IDF em vez de One-Hot ou TF?</b></h4>
**Resposta:**  
- **One-Hot**: vetores binários, sem contexto ou peso.  
- **TF**: considera frequência mas ignora relevância no corpus.  
- **TF-IDF**: balanceia frequência no documento com raridade no corpus, favorecendo termos mais relevantes.

---

<b><h4> 14. Em qual célula está a função que cria o vetor TF-IDF?</b></h4>
**Resposta:**  
**Célula 24**
```python
class Vectorizer:
    def vectorizer(self):
        self.tfidf_vectorizer = TfidfVectorizer(...)
```

---

<b><h4> 15. Em qual célula estão sendo extraídos os tópicos com LDA?</b></h4>
**Resposta:**  
**Célula 25**
```python
lda = LatentDirichletAllocation(n_components=9, max_iter=100, random_state=SEED)
```

---

<b><h4> 16. Em qual célula está a visualização `pyLDAvis`?</b></h4>
**Resposta:**  
**Célula 26**
```python
pyLDAvis.sklearn.prepare(lda, corpus, vectorizer.tfidf_vectorizer)
```

---

<b><h4> 17. Figura da nuvem de palavras por tópico.</b></h4>
**Resposta:**  
<div style="display: flex; justify-content: center; align-items: center; height: 100vh;">
    <img src="https://raw.githubusercontent.com/oserxavier/Machine-Learning/refs/heads/main/figures/wordcloud_2.png" height="450" width="450">
</div>

---

<b><h4> 18. Descreva cada um dos 9 tópicos extraídos com avaliação semântica.</b></h4>

| Tópico | Palavras-chave               | Descrição                  | Consistência |
|--------|------------------------------|----------------------------|--------------|
| 1      | economia, mercado, dólar     | Notícias financeiras       | Alta         |
| 2      | governo, política, congresso | Política nacional          | Alta         |
| 3      | empresas, investimento       | Economia e negócios        | Média        |
| 4      | Petrobras, petróleo          | Setor energético            | Alta         |
| 5      | greve, sindicato             | Questões trabalhistas      | Média        |
| 6      | banco, crédito               | Sistema bancário           | Alta         |
| 7      | educação, escolas            | Educação                   | Alta         |
| 8      | hospital, saúde              | Sistema de saúde           | Alta         |
| 9      | tecnologia, inovação         | Inovação e startups        | Alta         |

---

<b><h4> 19. Quais os passos para gerar vetores com Doc2Vec?</b></h4>
**Resposta:**
1. Pré-processar textos (limpeza e tokenização)
2. Transformar em `TaggedDocument`
3. Treinar modelo com `Doc2Vec()`
4. Inferir vetores para novos documentos

---

<b><h4> 20. TF-IDF ou Doc2Vec para K-Médias?</b></h4>
**Resposta:**  
**Doc2Vec**  
- Reduz dimensionalidade  
- Representa semântica melhor  
- Mais adequado para métricas vetoriais do K-means

---

<b><h4> 21. Benefícios do lda2vec segundo o artigo da StitchFix?</b></h4>
**Resposta:**  
- O modelo **lda2vec** combina Word2Vec + LDA  
- Produz tópicos semanticamente mais ricos  
- Ideal quando precisamos de **interpretação e semântica**

[Artigo: lda2vec](https://multithreaded.stitchfix.com/blog/2016/05/27/lda2vec/#topic=38&lambda=1&term=)