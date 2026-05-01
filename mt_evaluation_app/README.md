# 机器翻译对比与评测系统

一个基于 Streamlit 的交互式机器翻译对比与质量评测平台。

## 快速部署

### 1. 安装依赖

```bash
cd mt_evaluation_app
pip install -r requirements.txt
```

### 2. 下载 NLTK 数据（首次运行自动下载）

### 3. 启动应用

```bash
streamlit run app.py
```

应用将在 `http://localhost:8501` 启动。

## 功能模块

| 模块 | 功能 |
|------|------|
| 🧠 神经机器翻译 | 使用 Helsinki-NLP/opus-mt-en-zh 模型进行英译中 |
| ⚔️ 规则直译 vs 神经意译 | 对比词典逐词替换与 NMT 的翻译效果差异 |
| 📊 BLEU 质量评测 | 计算候选译文与参考译文的 BLEU 分数 |

## 技术栈

- **Streamlit** — Web 框架
- **Hugging Face Transformers** — NMT 模型推理
- **NLTK** — BLEU 评分计算
- **Jieba** — 中文分词（用于 BLEU 的 token 化）

## 部署到 Streamlit Cloud

1. 将本文件夹推送到 GitHub 仓库
2. 前往 [share.streamlit.io](https://share.streamlit.io)
3. 选择仓库并指定 `app.py` 作为入口
4. 点击 Deploy

> 注意：首次启动时模型下载约需 1-2 分钟，之后会被缓存。
