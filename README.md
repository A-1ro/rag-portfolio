# RAG Portfolio API

LangChain + Qdrant + Groq で構築した RAG（Retrieval-Augmented Generation）API のポートフォリオです。

**デモ URL**: https://rag-application-theta.vercel.app

---

## 技術スタック

| 役割 | 技術 |
|------|------|
| オーケストレーション | LangChain >= 0.3（LCEL） |
| Embedding | OpenAI text-embedding-3-small |
| Vector DB | Qdrant Cloud Free Tier |
| LLM | Groq + Llama 3.3 70B |
| API | FastAPI (Python) |
| ホスト | Vercel Hobby |

**月額コスト目安: $0〜3**（デモ用途であれば実質 $0 に近い）

---

## アーキテクチャ

```
ユーザー
  │
  ▼
Vercel (FastAPI)
  │
  ├─► LangChain Chain（LCEL）
  │       │
  │       ├─► Qdrant Cloud（ベクター検索）
  │       │       └─ text-embedding-3-small でクエリを Embedding
  │       │
  │       └─► Groq API（Llama 3.3 70B）
  │               └─ 検索結果 + プロンプトで回答生成
  │
  ▼
レスポンス（JSON / SSE）
```

---

## API エンドポイント

### `GET /health`

ヘルスチェック。

```bash
curl https://rag-application-theta.vercel.app/health
# {"status":"ok"}
```

### `POST /query`

質問を送信し、回答と参照ソースを JSON で受け取る。

```bash
curl -X POST https://rag-application-theta.vercel.app/query \
  -H "Content-Type: application/json" \
  -d '{"question": "あなたのスキルセットを教えてください"}'
```

```json
{
  "answer": "Azure・LLM を軸に...",
  "sources": ["data/documents/スキルシート_入口英一郎.md"]
}
```

### `POST /query/stream`

同じ質問を SSE（Server-Sent Events）でストリーミング受信する。

```bash
curl -N -X POST https://rag-application-theta.vercel.app/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "あなたの強みを教えてください"}'
```

```
data: {"type": "token", "content": "Azure"}
data: {"type": "token", "content": "・LLM"}
...
data: {"type": "sources", "content": ["data/documents/スキルシート_入口英一郎.md"]}
```

---

## ローカルセットアップ

```bash
# 1. 依存インストール（uv 推奨）
uv venv
uv pip install -r requirements.txt

# 2. 環境変数を設定
cp .env.example .env
# .env を編集して各 API キーを記入

# 3. ドキュメントを取り込む
python -m app.ingest

# 4. サーバー起動
uvicorn app.main:app --reload
```

### 必要なサービス

| サービス | 用途 | 料金 |
|----------|------|------|
| [OpenAI](https://platform.openai.com) | Embedding | 従量課金（月 $1 以下） |
| [Qdrant Cloud](https://cloud.qdrant.io) | Vector DB | Free Tier |
| [Groq](https://console.groq.com) | LLM 推論 | Free Tier あり |

---

## ディレクトリ構成

```
rag-portfolio/
├── app/
│   ├── main.py     # FastAPI エントリーポイント
│   ├── chain.py    # LangChain LCEL チェーン定義
│   ├── ingest.py   # ドキュメント取り込み・Qdrant へのupsert
│   └── config.py   # 環境変数管理
├── data/
│   └── documents/  # 取り込むドキュメント（.txt, .md）
├── requirements.txt
├── vercel.json
└── .env.example
```

---

## 業務経験との接続

業務では **Azure AI Search + Azure OpenAI Service** を使った RAG 構成を本番運用まで担当。本プロジェクトはその同等構成を OSS スタック（Qdrant + Groq + LangChain）で再現したものです。

- Azure 側の経験: Azure Blob をドキュメントストアとする RAG 実装、プロンプト戦略の設計、レスポンス評価パイプライン
- 本プロジェクトでの差別化: LCEL による宣言的チェーン設計、SSE ストリーミング対応、コスト最適化（月 $0〜3）
