import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title("株価予測 AIアシスタント")
st.write("ボリンジャーバンドやマクロ動向から明日の動きを予測します。")

# ==========================================
# 1. データの準備（AIの学習）※先ほどと同じ
# ==========================================
# （※本来はここで実際の株価データを読み込みます）
np.random.seed(42)
df = pd.DataFrame({
    "return_1d": np.random.randn(500),
    "bb_position": np.random.uniform(-2, 2, 500),
    "macro_yield": np.random.randn(500),
})
df["target"] = np.where((df["bb_position"] < -1.0) & (df["macro_yield"] > 0), 1, 0)

ai_agent = RandomForestClassifier(n_estimators=100, random_state=42)
ai_agent.fit(df[["return_1d", "bb_position", "macro_yield"]], df["target"])

# ==========================================
# 2. Streamlitの画面操作でAIに予測させる
# ==========================================
st.subheader("今日の相場データを入力して予測")

# ユーザーが画面上で数値をスライダーで入力できるUIを作成
current_bb = st.slider("現在のボリンジャーバンド位置", -3.0, 3.0, -1.5)
current_yield = st.slider("現在のマクロ金利動向", -2.0, 2.0, 0.5)
current_return = st.slider("前日騰落率", -5.0, 5.0, 0.0)

# ボタンを押したらAIが予測を実行
if st.button("AIに予測させる"):
    # 入力されたデータをAIに渡す
    today_data = pd.DataFrame([[current_return, current_bb, current_yield]], 
                              columns=["return_1d", "bb_position", "macro_yield"])
    
    # 予測の実行
    prediction = ai_agent.predict(today_data)
    
    st.divider()
    if prediction[0] == 1:
        st.success("🤖 AIの予測: **「上昇する可能性が高いです（買いサイン）」**")
    else:
        st.error("🤖 AIの予測: **「下落、または様子見です」**")
