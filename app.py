import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

st.title("🍅協力型おいしさ判定AIアプリ（試作版）")

st.write("写真をアップロードして、おいしさを評価しましょう！")

# --- 画像アップロード ---
uploaded_file = st.file_uploader("画像を選択", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="アップロードした画像", use_column_width=True)

    # --- スライダー入力 ---
    st.subheader("おいしさの評価")
    sweetness = st.slider("甘さ", 0, 10, 5)
    juiciness = st.slider("ジューシーさ", 0, 10, 5)
    color = st.slider("色の良さ", 0, 10, 5)

    # 総合スコア
    score = round((sweetness + juiciness + color) / 3, 1)
    st.metric("総合スコア", f"{score}/10")

    # --- AI予測部分（ダミー or 簡易モデル）---
    st.subheader("AIによるおいしさ予測（デモ）")

    # 画像をリサイズ＆正規化（あなたのAIの入力サイズに合わせて変更OK）
    img_resized = image.resize((128, 128))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # 簡易AIモデル（あなたの既存モデルを使う場合はここを差し替え）
    model = tf.keras.models.load_model("mini_tomato_model.h5")


    # 仮の予測値（学習済モデルに差し替えればOK）
    prediction = float(model.predict(img_array)[0][0])
    predicted_score = round(prediction * 10, 1)

    st.write(f"🤖 AI予測スコア: **{predicted_score}/10**")

    # --- 保存ボタン ---
    if st.button("データを保存（擬似）"):
        st.success("データを保存しました！（実際の保存は未実装）")
