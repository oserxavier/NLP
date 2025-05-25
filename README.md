# Análise de Tópicos em Notícias com NLP (Português)

Este projeto utiliza técnicas de **Processamento de Linguagem Natural (PLN)** para analisar e agrupar automaticamente notícias em português em diferentes **tópicos latentes**. O estudo tem como base dados da seção *Mercado* da Folha de São Paulo (ano de 2016), extraídos da plataforma [Kaggle](https://www.kaggle.com/).

---

## Objetivo

O principal objetivo foi aplicar uma pipeline completa de NLP para:
- Pré-processar textos jornalísticos
- Aplicar técnicas de lematização e reconhecimento de entidades (NER)
- Gerar vetores TF-IDF a partir dos textos
- Extrair tópicos com **Latent Dirichlet Allocation (LDA)**
- Visualizar os principais temas por meio de nuvens de palavras e entidades

---

## Tecnologias e Bibliotecas

- Python 3.x
- Pandas & NumPy
- NLTK
- spaCy (`pt_core_news_lg`)
- scikit-learn (TF-IDF & LDA)
- Matplotlib & Seaborn
- WordCloud
- pyLDAvis
- Google Colab

---

## Principais Aprendizados

### 1. Pré-processamento de Texto
Aprendemos a aplicar **tokenização, remoção de stopwords, stemming e lematização**, adaptados para o idioma português, com NLTK e spaCy.

### 2. Vetorização
Exploramos o uso de **TF-IDF** como alternativa eficaz às codificações frequenciais tradicionais (TF e One-hot), melhorando a representação dos documentos.

### 3. Extração de Tópicos
Aplicamos o modelo **Latent Dirichlet Allocation (LDA)** para identificar padrões temáticos nos textos, agrupando documentos de forma não supervisionada.

### 4. Entidades Nomeadas (NER)
Utilizamos o modelo de spaCy para extrair **organizações** de cada notícia e visualizá-las por tópico, permitindo uma análise mais interpretável dos resultados.

### 5. Visualização
Geramos **nuvens de palavras** e **nuvens de entidades** para cada um dos 9 tópicos, facilitando a interpretação dos temas dominantes.

---

## Importância do Estudo

O domínio de técnicas de NLP é cada vez mais essencial para lidar com o volume massivo de textos disponíveis hoje. Neste projeto, foi possível:

- Praticar uma **pipeline realista de NLP**, do pré-processamento à modelagem de tópicos
- Trabalhar com **textos em língua portuguesa**, ainda pouco priorizados em datasets globais
- Interpretar grandes volumes de dados textuais sem rotulagem manual
- Utilizar a **representação vetorial e semântica** para agrupar automaticamente conteúdos com características semelhantes

---

## Estrutura do Projeto

nlp-topicos-noticias<br>
├── data/ # Arquivo de notícias extraído do Kaggle<br>
├── notebooks/ # Notebooks em Colab<br>
│ └── analise_topicos.ipynb<br>
├── outputs/ # Nuvens de palavras e entidades<br>
├── projeto_nlp_respostas.md # Arquivo de respostas às competências<br>
└── README.md # Este arquivo<br>


---

## Como Executar

1. Faça o fork ou clone deste repositório
2. Execute o notebook principal no Google Colab
3. Instale os requisitos abaixo no ambiente:
```bash
pip install -U spacy nltk wordcloud pyldavis
python -m spacy download pt_core_news_lg


