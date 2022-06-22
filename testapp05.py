### 使用ライブラリの説明 ###
#
# requirements.txt の内容
#   streamlit               ← 毎度、おなじみ
#   typing_extensions       ← これがないとエラーが出ることがあるので一応
#   numpy                   ← 毎度、おなじみ
#   pandas                  ← 毎度、おなじみ
#   scikit-learn            ← 超有名、機械学習ライブラリ 
#   imblearn                ← データの水増し用
#   matplotlib              ← 超有名、グラフ描画ライブラリ
#   japanize-matplotlib     ← （の日本語対応版）
#   seaborn                 ← こちらもメジャーなグラフ描画ライブラリ
#
#
### 今回のミッション ###
#
# いままでの演習を振り返りながら…
#
#   ・仮想環境の構築（venvの利用）
#   ・pipによるライブラリのインストール（requirements.txtの利用）
#   ・Streamlit runでアプリの起動
#   ・不具合修正（今回配布するコードには、わざと！エラーが埋め込まれています）
#   ・GitHubへのアップロード（新規リポジトリの作成 → ファイルのプッシュ）
#   ・Streamlit Cloudでのデプロイ
#
# ……を行ってください（まさに、総集編！）
#
# 完了後、デプロイされたアプリの「URL」をレポートに記載して提出してください。
# 
# 締め切り：2022年6月24日(金) 17:30
#
# ※環境構築はハマると泥沼なので「おかしい？」と思ったら、すぐ相談しましょう。
#

from itertools import chain
import numpy as np
import pandas as pd 
import streamlit as st
import matplotlib.pyplot as plt 
import japanize_matplotlib
import seaborn as sns 

# 決定木
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

# ランダムフォレスト
from sklearn.ensemble import RandomForestClassifier

# 精度評価用
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

# データを分割するライブラリを読み込む
from sklearn.model_selection import train_test_split

# データを水増しするライブラリを読み込む
from imblearn.over_sampling import SMOTE

# ロゴの表示用
from PIL import Image

# ディープコピー
import copy

sns.set()
japanize_matplotlib.japanize()  # 日本語フォントの設定

# matplotlib / seaborn の日本語の文字化けを直す、汎用的かつ一番簡単な設定方法 | BOUL
# https://boul.tech/mplsns-ja/


def st_display_table(df: pd.DataFrame):

    # データフレームを表示
    st.subheader('データの確認')
    st.table(df)

    # Streamlitでdataframeを表示させる | ITブログ
    # https://kajiblo.com/streamlit-dataframe/


def main():
    """ メインモジュール
    """

    # stのタイトル表示
    st.title("退職予測AI\n（Maschine Learning)")

    # ファイルのアップローダー
    uploaded_file = st.sidebar.file_uploader("訓練用データのアップロード", type='csv') 

    # サイドメニューの設定
    activities = ["データ確認", "要約統計量", "グラフ表示", "学習と検証", "About"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    if choice == 'データ確認':
        # アップロードの有無を確認
        if uploaded_file is not None:

            # 一度、read_csvをするとインスタンスが消えるので、コピーしておく
            ufile = copy.deepcopy(uploaded_file)

            try:
                # 文字列の判定
                pd.read_csv(ufile, encoding="utf_8_sig")
                enc = "utf_8_sig"
            except:
                enc = "shift-jis"

            finally:
                # データフレームの読み込み
                df = pd.read_csv(uploaded_file, encoding=enc) 

                # ary_cnt = ["10", "50", "100", ]
                # cnt = st.sidebar.selectbox("Select Max mm", ary_cnt)
                cnt = st.sidebar.slider('表示する件数', 1, len(df), 10)

                # テーブルの表示
                st_display_table(df.head(int(cnt)))
        else:
            st.subheader('訓練用データをアップロードしてください')

    elif choice == '要約統計量':

        if uploaded_file is not None:

            ufile = copy.deepcopy(uploaded_file)

            try:
                pd.read_csv(ufile, encoding="utf_8_sig")
                enc = "utf_8_sig"
            except:
                enc = "shift-jis"
            finally:
                # columns:列    index:行
                df = pd.read_csv(uploaded_file, encoding=enc)
                # st.write(df)
                columns1 = []
                # count = 0
                # count1 = 0
                df = df.drop(columns='退職')
                dataset = [[],[],[],[],[],[],[],[]]
                for j in df.count():
                    dataset[0].append(j)
                for j in df.mean():
                    dataset[1].append(j)
                for j in df.std():
                    dataset[2].append(j)
                for j in df.min():
                    dataset[3].append(j)
                for j in df.quantile(0.25):
                    dataset[4].append(j)
                for j in df.quantile(0.50):
                    dataset[5].append(j)
                for j in df.quantile(0.75):
                    dataset[6].append(j)
                for j in df.max():
                    dataset[7].append(j)
                index1 = [[df.count()],[df.mean()],[df.std()],[df.min()],[df.quantile(0.25)],[df.quantile(0.50)],[df.quantile(0.75)],[df.max()]]
                # for k in index1:
                #     count1 = 0
                #     for l in k:
                #         # dataset[count][count1].append(l)
                #         st.write(l)
                #         count1 += 1
                #     count += 1
                index_title = ['count','mean','std','min','25%','50%','75','max']
                for i in df.columns.values:
                    columns1.append(i)
                new_df = pd.DataFrame(data=dataset,index=index_title,columns=columns1)
                st.write(new_df)

            
        else:
            st.subheader('訓練用データをアップロードしてください')

    elif choice == 'グラフ表示':
        if uploaded_file is not None:
            ufile = copy.deepcopy(uploaded_file)
            try:
                pd.read_csv(ufile,encoding="utf_8_sig")
                enc = "utf_8_sig"
            except:
                enc = "shift-jis"
            finally:
                df = pd.read_csv(uploaded_file, encoding=enc)
                st.write(df)
                side = []
                for i in df.columns.values:
                    side.append(i)
                set_side = st.sidebar.selectbox('グラフのx軸',side)
                for j in set_side:
                    st.write(j)
                st.write(len(df.columns.values))
                st.write(df[:1].plot.bar())


if __name__ == "__main__":
    main()

