import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics.pairwise import cosine_similarity
import os
from PIL import Image
import matplotlib.pyplot as plt
from joblib import dump, load

# Veri yükleme
articles = pd.read_csv("datasets/articles.csv")

# Gerekli sütunları seçme ve veriyi hazırlama
articles_sub = articles[['article_id','prod_name','product_type_name','product_group_name','graphical_appearance_name','colour_group_name'
                         ,'perceived_colour_value_name','perceived_colour_master_name','department_name','index_name','index_group_name'
                         ,'section_name','garment_group_name','detail_desc']]

# İstenmeyen kategorileri ve ürün gruplarını filtreleme
articles_sub = articles_sub[~articles_sub['index_group_name'].isin(['Baby/Children'])]
articles_sub = articles_sub[~articles_sub['index_name'].isin(['Lingeries/Tights'])]
articles_sub = articles_sub[~articles_sub['product_group_name'].isin(['Furniture','Items','Swimwear','Underwear','Unknown','Stationery','Nightwear',"Socks & Tights","Garment and Shoe care"])]
articles_sub = articles_sub[~articles_sub['product_type_name'].isin(["Dog Wear", "Giftbox", "Ring", "Wallet", "Waterbottle", "Costumes", "Underwear Tights"])]
articles_sub = articles_sub[~articles_sub['perceived_colour_value_name'].isin(["Undefined", "Unknown"])]


for col in articles.columns:
    if not 'no' in col and not 'code' in col and not 'id' in col:
        un_n = articles[col].nunique()
        print(f'n of unique {col}: {un_n}')

# Cilt alt tonlarına göre renk seçimi
articles_sub["warm_untone"] = articles_sub['colour_group_name'].apply(lambda x: x if x in ["White", "Light Beige", "Beige", "Light Pink", "Off White", "Red", "Dark Green", "Gold", "Yellowish Brown", "Pink",
                                                      "Light Grey", "Yellow", "Light Orange", "Dark Beige", "Dark Orange", "Dark Pink", "Orange", "Light Yellow", "Green",
                                                      "Dark Yellow", "Light Green", "Other Pink", "Greyish Pink", "Light Red", "Other Yellow", "Other Orange", "Other Red",
                                                      "Other Green", "Bronze"] else None)
articles_sub["cold_untone"] = articles_sub['colour_group_name'].apply(lambda x: x if x in ["Black", "Dark Blue", "Grey", "Greenish Khaki", "Red", "Dark Grey", "Light Blue", "Dark Red",
                                                                           "Dark Orange", "Silver", "Light Purple", "Dark Turquoise", "Light Turquoise", "Dark Purple",
                                                                           "Turquoise", "Purple", "Other Blue", "Other Purple", "Other Turquoise"] else None)
articles_sub["neutral_untone"] = articles_sub['colour_group_name']
articles_sub['cold_untone'] = articles_sub['cold_untone'].replace({None: "Not",})
articles_sub['warm_untone'] = articles_sub['warm_untone'].replace({None: "Not",})
articles_sub['neutral_untone'] = articles_sub['neutral_untone'].replace({None: "Not",})

# Verileri birleştirme
cols = ["prod_name", "product_type_name", "product_group_name", "graphical_appearance_name", "colour_group_name", "perceived_colour_value_name", "perceived_colour_master_name",
        "department_name", "index_name", "index_group_name", "section_name", "garment_group_name", "detail_desc", "warm_untone", "cold_untone", "neutral_untone"]
articles_sub['combined'] = articles_sub[cols].apply(lambda row: ', '.join(row.values.astype(str)), axis=1)

# Sadece gerekli sütunları saklama
df = articles_sub[["article_id", "combined"]]

# TF-IDF ve Incremental PCA tanımlama
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined'])

# Incremental PCA tanımlama
ipca = IncrementalPCA(n_components=300)

# Veriyi parçalara bölme ve Incremental PCA uygulama
chunk_size = 10000
reduced_tfidf_matrix = np.zeros((len(df), 300))

# İlk olarak PCA modelini eğitmek için partial_fit kullan
for i in range(0, len(df), chunk_size):
    tfidf_matrix_chunk = tfidf_matrix[i:i + chunk_size].toarray()
    ipca.partial_fit(tfidf_matrix_chunk)
    print(f"Chunk {i // chunk_size + 1} PCA eğitimi tamamlandı.")

# Eğitimden sonra transform işlemini uygulayarak veriyi dönüştür
for i in range(0, len(df), chunk_size):
    tfidf_matrix_chunk = tfidf_matrix[i:i + chunk_size].toarray()
    reduced_chunk = ipca.transform(tfidf_matrix_chunk)
    reduced_tfidf_matrix[i:i + chunk_size] = reduced_chunk
    print(f"Chunk {i // chunk_size + 1} PCA dönüşümü tamamlandı.")

# Kosinüs benzerliğini hesaplama
cosine_sim_products = cosine_similarity(reduced_tfidf_matrix, reduced_tfidf_matrix)

# İndeksleri oluşturma
indices = pd.Series(df.index, index=df['article_id']).drop_duplicates()

# Öneri fonksiyonu
def get_recommendations(article_id, cosine_sim=cosine_sim_products):
    idx = indices[article_id]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[:5]
    article_indices = [i[0] for i in sim_scores]
    return df['article_id'].iloc[article_indices]

# Görselleştirme fonksiyonu
def display_articles(article_ids):
    rows = 4
    cols = 3
    image_path = "datasets/images"
    plt.figure(figsize=(2 + 3 * cols, 2 + 4 * rows))
    for i in range(len(article_ids)):
        article_id = ("0" + str(article_ids[i]))[-10:]
        plt.subplot(rows, cols, i + 1)
        plt.axis('off')
        try:
            image = Image.open(f"{image_path}/{article_id[:3]}/{article_id}.jpg")
            plt.imshow(image)
        except Exception as e:
            print(f"{article_id} kimliğindeki görüntü yüklenirken hata oluştu: {e}")
    plt.show()

# Örnek öneri ve görselleştirme
recom = list(get_recommendations(176209025))
display_articles(recom)
