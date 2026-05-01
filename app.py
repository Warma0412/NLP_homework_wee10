import streamlit as st
import jieba
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

# Ensure nltk punkt is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

# ─── Page Config ───
st.set_page_config(
    page_title="机器翻译对比与评测系统",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Custom CSS ───
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }

    .main-header h1 {
        background: linear-gradient(120deg, #a8edea 0%, #fed6e3 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .main-header p {
        color: #b8b8d4;
        font-size: 1.1rem;
        font-weight: 300;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 6px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #b8b8d4;
        font-weight: 500;
        padding: 10px 24px;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
    }

    .result-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }

    .result-card h3 {
        color: #a8edea;
        margin-bottom: 0.8rem;
        font-weight: 600;
    }

    .result-card p, .result-card .translation-text {
        color: #e8e8f0;
        font-size: 1.1rem;
        line-height: 1.8;
    }

    .metric-card {
        background: linear-gradient(135deg, rgba(102,126,234,0.2) 0%, rgba(118,75,162,0.2) 100%);
        border: 1px solid rgba(168,237,234,0.3);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }

    .metric-card .score {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(120deg, #a8edea 0%, #fed6e3 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-card .label {
        color: #b8b8d4;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }

    .compare-col {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.5rem;
        min-height: 200px;
    }

    .tag-rule {
        display: inline-block;
        background: rgba(255,107,107,0.2);
        border: 1px solid rgba(255,107,107,0.4);
        color: #ff6b6b;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
    }

    .tag-nmt {
        display: inline-block;
        background: rgba(72,219,251,0.2);
        border: 1px solid rgba(72,219,251,0.4);
        color: #48dbfb;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
    }

    .info-box {
        background: rgba(168,237,234,0.08);
        border-left: 4px solid #a8edea;
        border-radius: 0 12px 12px 0;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        color: #d0d0e8;
    }

    div[data-testid="stTextInput"] label,
    div[data-testid="stTextArea"] label {
        color: #d0d0e8 !important;
        font-weight: 500 !important;
    }

    .stTextInput input, .stTextArea textarea {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 10px !important;
        color: #e8e8f0 !important;
    }

    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 2px rgba(102,126,234,0.3) !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102,126,234,0.4) !important;
    }

    .stSpinner > div {
        border-color: #667eea !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ───
st.markdown("""
<div class="main-header">
    <h1>🌐 机器翻译对比与评测系统</h1>
    <p>Neural Machine Translation Comparison & Evaluation Platform</p>
</div>
""", unsafe_allow_html=True)


# ─── Model Loading (Cached) ───
@st.cache_resource(show_spinner=False)
def load_nmt_model():
    from transformers import pipeline
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-zh")
    return translator


# ─── Rule-Based Translation Engine ───
ENGLISH_CHINESE_DICT = {
    # Common words
    "i": "我", "you": "你", "he": "他", "she": "她", "it": "它",
    "we": "我们", "they": "他们", "am": "是", "is": "是", "are": "是",
    "was": "是", "were": "是", "the": "那个", "a": "一个", "an": "一个",
    "this": "这个", "that": "那个", "these": "这些", "those": "那些",
    "have": "有", "has": "有", "had": "有", "do": "做", "does": "做",
    "did": "做", "will": "将", "would": "会", "can": "能", "could": "能够",
    "should": "应该", "may": "可能", "might": "可能", "must": "必须",
    "not": "不", "no": "不", "yes": "是的",
    # Nouns
    "cat": "猫", "cats": "猫", "dog": "狗", "dogs": "狗",
    "book": "书", "books": "书", "water": "水", "food": "食物",
    "man": "男人", "woman": "女人", "child": "孩子", "children": "孩子们",
    "time": "时间", "day": "天", "year": "年", "world": "世界",
    "life": "生活", "hand": "手", "house": "房子", "home": "家",
    "school": "学校", "student": "学生", "teacher": "老师",
    "computer": "计算机", "machine": "机器", "translation": "翻译",
    "language": "语言", "word": "词", "sentence": "句子",
    "rain": "雨", "rains": "下雨", "sun": "太阳", "moon": "月亮",
    "tree": "树", "flower": "花", "river": "河", "mountain": "山",
    # Verbs
    "go": "去", "goes": "去", "come": "来", "comes": "来",
    "see": "看", "look": "看", "read": "读", "write": "写",
    "eat": "吃", "drink": "喝", "sleep": "睡觉", "run": "跑",
    "walk": "走", "talk": "说话", "speak": "说", "say": "说",
    "think": "想", "know": "知道", "want": "想要", "need": "需要",
    "like": "喜欢", "love": "爱", "hate": "讨厌", "make": "制作",
    "give": "给", "take": "拿", "get": "得到", "put": "放",
    "learn": "学习", "study": "学习", "work": "工作", "play": "玩",
    "live": "住", "die": "死", "kill": "杀", "help": "帮助",
    "translate": "翻译", "understand": "理解",
    # Adjectives
    "good": "好的", "bad": "坏的", "big": "大的", "small": "小的",
    "new": "新的", "old": "旧的", "young": "年轻的",
    "long": "长的", "short": "短的", "high": "高的", "low": "低的",
    "happy": "快乐的", "sad": "悲伤的", "beautiful": "美丽的",
    "important": "重要的", "different": "不同的",
    # Prepositions & Conjunctions
    "in": "在...里", "on": "在...上", "at": "在", "to": "到",
    "from": "从", "with": "和", "without": "没有", "about": "关于",
    "for": "为了", "of": "的", "and": "和", "or": "或",
    "but": "但是", "because": "因为", "if": "如果", "when": "当",
    "very": "非常", "also": "也", "just": "只是", "then": "然后",
    # Weather & Nature
    "weather": "天气", "hot": "热的", "cold": "冷的", "warm": "温暖的",
    "wind": "风", "snow": "雪", "cloud": "云",
    # Other common
    "today": "今天", "tomorrow": "明天", "yesterday": "昨天",
    "morning": "早上", "evening": "晚上", "night": "夜晚",
    "here": "这里", "there": "那里", "where": "哪里",
    "what": "什么", "who": "谁", "how": "怎样", "why": "为什么",
    "all": "所有", "every": "每个", "many": "许多", "much": "很多",
    "some": "一些", "any": "任何", "other": "其他",
}


def rule_based_translate(text):
    """Simulate early rule-based machine translation via word-by-word dictionary lookup."""
    import re
    # Tokenize by splitting on spaces and punctuation boundaries
    tokens = re.findall(r"[a-zA-Z']+|[^\s\w]", text)
    translated_tokens = []
    for token in tokens:
        lower = token.lower().strip("'\"")
        if lower in ENGLISH_CHINESE_DICT:
            translated_tokens.append(ENGLISH_CHINESE_DICT[lower])
        else:
            translated_tokens.append(token)
    return " ".join(translated_tokens)


def nmt_translate(text, translator):
    """Translate using NMT model."""
    result = translator(text, max_length=512)
    return result[0]['translation_text']


def compute_bleu(reference, candidate):
    """Compute BLEU score between reference and candidate (Chinese text, char-level tokenization via jieba)."""
    ref_tokens = list(jieba.cut(reference))
    cand_tokens = list(jieba.cut(candidate))
    smoothie = SmoothingFunction().method1
    score = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothie)
    return score


# ─── Tabs ───
tab1, tab2, tab3 = st.tabs(["🧠 神经机器翻译", "⚔️ 规则直译 vs 神经意译", "📊 BLEU 质量评测"])

# ═══════════════════════════════════════════
# TAB 1: Neural Machine Translation
# ═══════════════════════════════════════════
with tab1:
    st.markdown("""
    <div class="info-box">
        <strong>神经机器翻译 (NMT)</strong> 使用深度学习的 Encoder-Decoder 架构，能够理解上下文语境，
        自动学习语言之间的对应关系，处理多义词、语序差异等问题。本模块使用 Helsinki-NLP/opus-mt-en-zh 模型。
    </div>
    """, unsafe_allow_html=True)

    col_input, col_output = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown("#### ✍️ 输入英文文本")
        nmt_input = st.text_area(
            "请输入英文句子",
            value="It rains cats and dogs.",
            height=150,
            key="nmt_input",
            label_visibility="collapsed"
        )
        nmt_btn = st.button("🚀 翻译", key="nmt_btn", use_container_width=True)

    with col_output:
        st.markdown("#### 🎯 翻译结果")
        if nmt_btn and nmt_input.strip():
            with st.spinner("🔄 神经网络推理中..."):
                translator = load_nmt_model()
                nmt_result = nmt_translate(nmt_input, translator)
            st.markdown(f"""
            <div class="result-card">
                <h3>中文译文</h3>
                <p class="translation-text">{nmt_result}</p>
            </div>
            """, unsafe_allow_html=True)
            # Store in session for reuse
            st.session_state['last_nmt_input'] = nmt_input
            st.session_state['last_nmt_output'] = nmt_result
        elif nmt_btn:
            st.warning("请输入文本后再翻译。")

    st.markdown("---")
    st.markdown("""
    <div class="info-box">
        💡 <strong>观察建议：</strong>尝试输入含有俚语的句子（如 "It rains cats and dogs." — 倾盆大雨）、
        含有定语从句的复杂长句，观察 NMT 是否能正确理解语境和隐含含义。
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════
# TAB 2: Rule-based vs NMT Comparison
# ═══════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class="info-box">
        <strong>基于规则的机器翻译</strong> 依赖预定义词典进行逐词替换，无法理解上下文、语序、一词多义等语言现象。
        在此对比中，你可以直观地看到"直译"与"意译"之间的巨大鸿沟。
    </div>
    """, unsafe_allow_html=True)

    compare_input = st.text_area(
        "输入英文句子进行对比翻译",
        value="The student who loves reading books goes to school every day.",
        height=120,
        key="compare_input"
    )

    compare_btn = st.button("⚡ 对比翻译", key="compare_btn", use_container_width=True)

    if compare_btn and compare_input.strip():
        col_rule, col_nmt = st.columns(2, gap="large")

        with col_rule:
            rule_result = rule_based_translate(compare_input)
            st.markdown(f"""
            <div class="compare-col">
                <span class="tag-rule">📖 基于规则的直译</span>
                <h3 style="color:#ff6b6b; margin-top:1rem;">逐词替换结果</h3>
                <p style="color:#e8e8f0; font-size:1.1rem; line-height:2;">{rule_result}</p>
            </div>
            """, unsafe_allow_html=True)

        with col_nmt:
            with st.spinner("🧠 NMT 翻译中..."):
                translator = load_nmt_model()
                nmt_result = nmt_translate(compare_input, translator)
            st.markdown(f"""
            <div class="compare-col">
                <span class="tag-nmt">🧠 神经机器翻译</span>
                <h3 style="color:#48dbfb; margin-top:1rem;">NMT 意译结果</h3>
                <p style="color:#e8e8f0; font-size:1.1rem; line-height:2;">{nmt_result}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div class="info-box">
            🔍 <strong>对比分析：</strong><br>
            • <strong>语序问题：</strong>规则翻译保留了英文语序，而中文的定语通常前置于被修饰词<br>
            • <strong>一词多义：</strong>规则翻译对每个词只有一个固定译法，无法根据上下文选择<br>
            • <strong>整体流畅性：</strong>NMT 生成的译文更符合中文的表达习惯
        </div>
        """, unsafe_allow_html=True)
    elif compare_btn:
        st.warning("请输入文本后再进行对比。")

# ═══════════════════════════════════════════
# TAB 3: BLEU Score Evaluation
# ═══════════════════════════════════════════
with tab3:
    st.markdown("""
    <div class="info-box">
        <strong>BLEU (Bilingual Evaluation Understudy)</strong> 是机器翻译自动评测的经典指标。
        它通过计算候选译文与参考译文之间的 n-gram 重叠率来衡量翻译质量。分数范围 0~1，越接近 1 表示越接近参考译文。
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1.2, 0.8], gap="large")

    with col_left:
        st.markdown("#### 📝 评测输入")

        bleu_source = st.text_area(
            "英文原文 (Source)",
            value="The weather is very nice today.",
            height=80,
            key="bleu_source"
        )

        bleu_reference = st.text_area(
            "中文参考译文 (Reference)",
            value="今天天气非常好。",
            height=80,
            key="bleu_ref"
        )

        bleu_candidate = st.text_area(
            "机器候选译文 (Candidate)",
            value="",
            height=80,
            key="bleu_cand",
            placeholder="留空可自动使用 NMT 模型生成..."
        )

        bleu_btn = st.button("📊 计算 BLEU 分数", key="bleu_btn", use_container_width=True)

    with col_right:
        st.markdown("#### 🎯 评测结果")

        if bleu_btn:
            candidate_text = bleu_candidate.strip()

            if not bleu_reference.strip():
                st.warning("请提供参考译文。")
            else:
                # If candidate is empty, auto-generate with NMT
                if not candidate_text:
                    if bleu_source.strip():
                        with st.spinner("🤖 自动生成候选译文..."):
                            translator = load_nmt_model()
                            candidate_text = nmt_translate(bleu_source, translator)
                        st.markdown(f"""
                        <div class="result-card">
                            <h3>自动生成的候选译文</h3>
                            <p class="translation-text">{candidate_text}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning("请输入英文原文或手动填写候选译文。")
                        st.stop()

                # Compute BLEU
                score = compute_bleu(bleu_reference.strip(), candidate_text)

                # Display score
                score_pct = score * 100
                if score_pct >= 60:
                    quality = "优秀 — 译文与参考高度吻合"
                    color = "#00d2d3"
                elif score_pct >= 40:
                    quality = "良好 — 译文较为接近参考"
                    color = "#a8edea"
                elif score_pct >= 20:
                    quality = "一般 — 有部分 n-gram 匹配"
                    color = "#feca57"
                else:
                    quality = "较差 — 与参考差异较大"
                    color = "#ff6b6b"

                st.markdown(f"""
                <div class="metric-card">
                    <div class="score">{score_pct:.2f}</div>
                    <div class="label">BLEU Score (0-100)</div>
                    <div style="color:{color}; margin-top:0.8rem; font-weight:500;">{quality}</div>
                </div>
                """, unsafe_allow_html=True)

                # Explanation
                st.markdown(f"""
                <div class="info-box">
                    <strong>📖 分数解读：</strong><br><br>
                    BLEU = <strong>{score_pct:.2f}</strong> 分（满分 100）<br><br>
                    该分数反映候选译文与参考译文在 1-gram 到 4-gram 上的匹配程度。<br><br>
                    <strong>BLEU 的优点：</strong>计算快速、可重复、与人工评分有一定相关性<br>
                    <strong>BLEU 的局限：</strong>无法识别同义词替换、不考虑语义相似性、对语序敏感度有限
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="info-box">
        💡 <strong>实验建议：</strong><br>
        • 尝试提供"词汇相同但语序全错"的参考译文 → 观察 BLEU 是否仍然给出较高分（体会 n-gram 的局限）<br>
        • 尝试提供"语义相同但用同义词替换"的参考译文 → 观察 BLEU 分数下降（体会 BLEU 对词汇重叠的依赖）
    </div>
    """, unsafe_allow_html=True)

# ─── Footer ───
st.markdown("---")
st.markdown("""
<div style="text-align:center; padding:1rem 0; color:#7f7f9f; font-size:0.85rem;">
    Machine Translation Comparison & Evaluation Platform &nbsp;|&nbsp;
    Powered by Helsinki-NLP/opus-mt-en-zh & NLTK BLEU &nbsp;|&nbsp;
    Built with Streamlit
</div>
""", unsafe_allow_html=True)
