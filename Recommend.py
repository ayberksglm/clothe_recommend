import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import os
import matplotlib.image as mpimg
from PIL import Image
##################  EDA & PRE-PROCCESSING ####################

articles = pd.read_csv("datasets/articles.csv")

df = articles.copy()
articles.describe().T
articles["product_type_name"].nunique

articles_sub = articles[['article_id','prod_name','product_type_name','product_group_name','graphical_appearance_name','colour_group_name'
                         ,'perceived_colour_value_name','perceived_colour_master_name','department_name','index_name','index_group_name'
                         ,'section_name','garment_group_name','detail_desc']]
articles_sub.head()


f, ax = plt.subplots(figsize=(15, 7))
ax = sns.histplot(data=articles, y='garment_group_name', color='orange', hue='index_group_name', multiple="stack")
ax.set_xlabel('count by garment group')
ax.set_ylabel('garment group')
plt.show()

articles_sub.groupby(['index_group_name', 'index_name']).count()['article_id']
dff = articles_sub[articles_sub['index_group_name'] != 'Baby/Children']
dff.shape
dff.head()

dff = dff[~dff['index_name'].isin(['Lingeries/Tights'])]

dff.groupby(['index_group_name', 'index_name']).count()['article_id']

for col in articles.columns:
    if not 'no' in col and not 'code' in col and not 'id' in col:
        un_n = articles[col].nunique()
        print(f'n of unique {col}: {un_n}')


pd.options.display.max_rows = None


dff = dff[~dff['product_group_name'].isin(['Furniture','Items','Swimwear','Underwear','Unknown','Stationery','Nightwear',"Socks & Tights","Garment and Shoe care"])]

dff = dff[~dff['product_type_name'].isin(["Dog Wear", "Giftbox", "Ring", "Wallet", "Waterbottle", "Costumes", "Underwear Tights"])]

dff.head()
dff.shape

dff["colour_group_name"].value_counts()
dff[dff["colour_group_name"].isin(["Green"])]

green_rows = dff[dff['colour_group_name'].str.contains('Green')]


green_rows.value_counts().head(20)
dff = dff[~dff['perceived_colour_value_name'].isin(["Undefined", "Unknown"])]
perceived_colour_value_name = dff["perceived_colour_value_name"].value_counts()

kucuk_veri = dff.to_csv("kucuk_veri.csv")


##################   FEATURE ENGINEERING ##################
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 800)

articles = pd.read_csv("datasets/articles.csv")
articles["colour_group_name"].value_counts()
df = pd.read_csv("kucuk_veri.csv")
df.head(20)

df["colour_group_name"].value_counts().shape


# CİLT ALT TONLARINA GÖRE RENK SEÇİMİ YAPILDI
df["warm_untone"] = df['colour_group_name'].apply(lambda x: x if x in ["White", "Light Beige", "Beige", "Light Pink", "Off White", "Red", "Dark Green", "Gold", "Yellowish Brown", "Pink",
                                                      "Light Grey", "Yellow", "Light Orange", "Dark Beige", "Dark Orange", "Dark Pink", "Orange", "Light Yellow", "Green",
                                                      "Dark Yellow", "Light Green", "Other Pink", "Greyish Pink", "Light Red", "Other Yellow", "Other Orange", "Other Red",
                                                      "Other Green", "Bronze"] else None)


df["cold_untone"] = df['colour_group_name'].apply(lambda x: x if x in ["Black", "Dark Blue", "Grey", "Greenish Khaki", "Red", "Dark Grey", "Light Blue", "Dark Red",
                                                                           "Dark Orange", "Silver", "Light Purple", "Dark Turquoise", "Light Turquoise", "Dark Purple",
                                                                           "Turquoise", "Purple", "Other Blue", "Other Purple", "Other Turquoise"] else None)

df["neutral_untone"] = df['colour_group_name']

# KUMAŞ TÜRÜNE GÖRE --- RAHATLIK  MATLIK   PARLAKLIK  BELİRLENDİ
# comfy = ["cotton canvas", "cotton weave", "elastic"]
# matte = ["cotton canvas", "cotton weave", "soft wool", "suede", "woven"]
# bright = ["cashmere", "leather", "viscose"]


# # Kumaş Özellikleri
# df['comfy'] = df['detail_desc'].apply(lambda x: any(kelime in x for kelime in comfy) if isinstance(x, str) else False)
# df['matte'] = df['detail_desc'].apply(lambda x: any(kelime in x for kelime in matte) if isinstance(x, str) else False)
# df['bright'] = df['detail_desc'].apply(lambda x: any(kelime in x for kelime in bright) if isinstance(x, str) else False)

# df['comfy'] = df['comfy'].replace({False: "Not", True: "Comfy"})
# df['matte'] = df['matte'].replace({False: "Not", True: "Matte"})
# df['bright'] = df['bright'].replace({False: "Not", True: "Bright"})

# df = df.drop("dewy_skin", axis=1)
# df = df.drop("matte_skin", axis=1)
# df = df.drop("normal_skin", axis=1)



df['cold_untone'] = df['cold_untone'].replace({None: "Not",})
df['warm_untone'] = df['warm_untone'].replace({None: "Not",})
df['neutral_untone'] = df['neutral_untone'].replace({None: "Not",})

cols = ["prod_name", "product_type_name", "product_group_name", "graphical_appearance_name", "colour_group_name", "perceived_colour_value_name", "perceived_colour_master_name",
        "department_name", "index_name", "index_group_name", "section_name", "garment_group_name", "detail_desc", "warm_untone", "cold_untone", "neutral_untone"]


df['combined'] = df[cols].apply(lambda row: ', '.join(row.values.astype(str)), axis=1)

# Sadece gerekli sütunları saklama
df = df[["article_id", "combined"]]

df.head()


articles_final = df.to_csv("articles_final.csv")

################### MATRIX FACTORIZATION #####################
articles_final = pd.read_csv("articles_final.csv")

articles_final = articles_final.loc[:15000]
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(articles_final['combined'])
tfidf_matrix.shape

#her bir ürünü temsil eden vektörler arasındaki benzerliği ölçer.
cosine_sim_products = linear_kernel(tfidf_matrix, tfidf_matrix)

#her article değerine ait bir index değeri atamış
indices = pd.Series(articles_final.index, index=articles_final['article_id']).drop_duplicates()

#idx diye bir parametre oluşturup bu parametrelerin benzerlik oranları üzerinden benzer 5 ürünün ıd sini getirmiş

articles = pd.read_csv("datasets/articles.csv")
image_dir = "datasets/images"
# img = mpimg.imread(f'datasets/images/0{str(data.article_id)

def get_recommendations(title, cosine_sim=cosine_sim_products):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[:5]
    article_indices = [i[0] for i in sim_scores]
    return articles_final['article_id'].iloc[article_indices]

recom = list(get_recommendations(319906002))
recom

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
            img_path = f"{image_path}/{article_id[:3]}/{article_id}.jpg"
            print(f"Fotoğraf yükleniyor: {img_path}") 
            image = Image.open(img_path)
            plt.imshow(image)
        except Exception as e:
            print(f"{article_id} kimliğindeki görüntü yüklenirken hata oluştu: {e}")

    plt.show()  


recom = list(get_recommendations(316441036))
display_articles(recom)
