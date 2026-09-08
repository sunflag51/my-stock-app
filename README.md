# BB Bottom Dashboard
ボリンジャーバンド下限付近で、底打ち確認条件を学習するための
Streamlitダッシュボードです。
## 主な機能
- Yahoo Financeから日足データを取得
- ローソク足とボリンジャーバンドを表示
- BB下限への接触・接近を黄色い三角で表示
- RSIと出来高を同時表示
- 6つの底打ち確認条件を自動判定
- 底打ち確認度を6点満点で表示
- バンドウォーク警告
- 初心者向けの確認手順を表示
- 分析データのCSVダウンロード
## 判定する6条件
1. 終値がBB下限内へ復帰
2. 陽線かつ前日終値超え
3. RSIが売られ過ぎから反転
4. 前日高値を終値で上抜け
5. 出来高増加
6. BB下限の下落停止
## GitHubへの配置
以下の4ファイルを同じ階層へ置いてください。
```text
bb_bottom_dashboard/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```
## Streamlit Community Cloudでの公開方法
1. GitHubで新しいリポジトリを作成します。
2. `app.py`、`requirements.txt`、`README.md`、`.gitignore`をアップロードします。
3. Streamlit Community Cloudへログインします。
4. `Create app`を選択します。
5. 作成したGitHubリポジトリを指定します。
6. Main file pathに次を入力します。
```text
app.py
```
7. `Deploy`を押します。
## 推奨Pythonバージョン
Python 3.11またはPython 3.12を推奨します。
Streamlit Community CloudでPythonバージョンを指定できる場合は、
Python 3.12を使用してください。
## 銘柄コードの入力例
このアプリの入力欄ではYahoo Finance用のコードを使用します。
### 日本株
```text
7974.T
7203.T
6758.T
```
### 米国株
```text
AAPL
MSFT
NVDA
```
入力コードはデータ取得サービス用の表記です。
実際の取引画面で使用される銘柄コード表記と異なる場合があります。
## ローカル環境での起動
ターミナルでアプリのフォルダへ移動します。
```bash
pip install -r requirements.txt
streamlit run app.py
```
## データ更新について
株価データは15分間キャッシュします。
更新したい場合は、サイドバーの「分析を更新」を押してください。
取得サービス側の状況によって、データの遅延、欠損、取得失敗が
発生する場合があります。
## 注意事項
このアプリの判定は、一般的なテクニカル指標を使った学習用判定です。
「底打ち確認候補」と表示されても、将来の上昇や利益を保証するものでは
ありません。最新の株価、出来高、企業情報、取引可能状況は、利用している
取引プラットフォームで確認してください。
