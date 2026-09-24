import streamlit as st
import fitz
import re
import os
from collections import Counter
from io import BytesIO

from dotenv import load_dotenv
from openai import OpenAI

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH


# =========================================================
# 1. 基础配置
# =========================================================

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

if DEEPSEEK_API_KEY:
    deepseek_client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com"
    )
else:
    deepseek_client = None


st.set_page_config(
    page_title="PDF智能摘要与文档问答系统",
    page_icon="📚",
    layout="wide"
)


# =========================================================
# 2. Session State 初始化
# =========================================================

if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "file_name" not in st.session_state:
    st.session_state.file_name = ""

if "file_size" not in st.session_state:
    st.session_state.file_size = 0

if "page_count" not in st.session_state:
    st.session_state.page_count = 0

if "ai_summary" not in st.session_state:
    st.session_state.ai_summary = ""

if "ai_keywords" not in st.session_state:
    st.session_state.ai_keywords = []

if "ai_structure" not in st.session_state:
    st.session_state.ai_structure = ""

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "api_status" not in st.session_state:
    st.session_state.api_status = "等待调用"


# =========================================================
# 3. 页面标题
# =========================================================

st.title("📚 PDF智能摘要与文档问答系统")

st.markdown(
    """
基于 **Python + Streamlit + PyMuPDF + DeepSeek API** 构建，
实现 PDF 文档解析、智能摘要、关键词提取、结构分析以及多轮文档问答。
"""
)


# =========================================================
# 4. DeepSeek API 状态
# =========================================================

if DEEPSEEK_API_KEY:
    st.success("🟢 DeepSeek API 已连接")
else:
    st.error("🔴 DeepSeek API 未连接，请检查 .env 文件中的 DEEPSEEK_API_KEY")


# =========================================================
# 5. 工具函数
# =========================================================

def clean_text(text):
    """清理 PDF 提取出来的文本"""

    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def split_text(text, chunk_size=1500):
    """将长文本切分成多个文本块"""

    chunks = []

    if not text:
        return chunks

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        start = end

    return chunks


def extract_pdf(file):
    """读取 PDF"""

    doc = fitz.open(stream=file.read(), filetype="pdf")

    pages = []

    for page in doc:
        pages.append(page.get_text())

    text = "\n".join(pages)

    return text, len(doc)


def call_deepseek(
    prompt,
    system_prompt="你是一个专业的中文文档分析助手。",
    temperature=0.3
):
    """
    调用 DeepSeek API
    """

    if not deepseek_client:
        return None, "DeepSeek API 未连接，请检查 .env 文件。"

    try:

        st.session_state.api_status = "正在调用"

        response = (
            deepseek_client
            .chat
            .completions
            .create(
                model="deepseek-flash",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature,
                stream=False
            )
        )

        content = response.choices[0].message.content

        if not content:
            st.session_state.api_status = "调用失败"

            return None, "DeepSeek 没有返回有效内容。"

        st.session_state.api_status = "调用成功"

        return content.strip(), None

    except Exception as e:

        st.session_state.api_status = "调用失败"

        return None, str(e)


def generate_ai_summary(text):
    """生成 AI 摘要"""

    prompt = f"""
请分析下面的 PDF 文档内容。

请输出：

1. 一段 150～250 字左右的中文摘要。
2. 文档最重要的 5 个核心内容。

要求：
- 使用中文。
- 不要编造文档中没有的信息。
- 不要输出“### 💡 回答”等标题。
- 直接输出正文。

文档内容：

{text[:20000]}
"""

    return call_deepseek(
        prompt,
        system_prompt="你是一名专业的中文文档摘要助手。"
    )


def generate_ai_keywords(text):
    """生成 AI 关键词"""

    prompt = f"""
请从下面的文档中提取最重要的关键词。

要求：
- 最多 15 个。
- 使用中文。
- 每个关键词尽量简洁。
- 使用顿号“、”分隔。
- 只能根据文档内容提取。
- 不要添加解释。

文档：

{text[:20000]}
"""

    return call_deepseek(
        prompt,
        system_prompt="你是一名专业的中文关键词提取助手。"
    )


def generate_ai_structure(text):
    """分析文档结构"""

    prompt = f"""
请分析下面 PDF 文档的结构。

请说明：

1. 文档主要由哪些部分组成。
2. 每个部分主要讲什么。
3. 文档整体的组织逻辑。

要求：
- 使用中文。
- 可以使用编号。
- 不要编造文档不存在的章节。
- 不要输出额外的标题。

文档：

{text[:25000]}
"""

    return call_deepseek(
        prompt,
        system_prompt="你是一名专业的中文文档结构分析助手。"
    )


def question_keywords(question):
    """提取问题关键词"""

    words = re.findall(r"[\u4e00-\u9fff]{2,}", question)

    return words


def search_relevant_chunks(question, chunks, top_k=5):
    """简单本地检索相关文本"""

    if not chunks:
        return []

    keywords = question_keywords(question)

    if not keywords:
        return chunks[:top_k]

    results = []

    for index, chunk in enumerate(chunks):

        score = 0

        for keyword in keywords:

            if keyword in chunk:
                score += chunk.count(keyword)

            # 2 字、3 字短语匹配
            if len(keyword) >= 2:

                for i in range(len(keyword) - 1):

                    sub = keyword[i:i + 2]

                    if sub in chunk:
                        score += 0.3

        results.append(
            (
                score,
                index,
                chunk
            )
        )

    results.sort(
        key=lambda x: x[0],
        reverse=True
    )

    selected = [
        item[2]
        for item in results[:top_k]
    ]

    return selected


def generate_ai_answer(question, chunks):
    """
    DeepSeek 多轮文档问答
    """

    relevant_chunks = search_relevant_chunks(
        question,
        chunks,
        top_k=5
    )

    context = "\n\n".join(relevant_chunks)

    # 限制上下文长度
    context = context[:16000]

    # =====================================================
    # 构造历史对话
    # =====================================================

    history_text = ""

    if st.session_state.chat_history:

        history_text = "\n\n之前的对话：\n"

        # 最近 6 轮
        recent_history = st.session_state.chat_history[-6:]

        for item in recent_history:

            history_text += (
                f"用户：{item['question']}\n"
                f"助手：{item['answer']}\n\n"
            )

    prompt = f"""
你现在需要回答用户关于 PDF 文档的问题。

请严格依据提供的文档内容进行回答。

如果文档中能够找到答案：
- 直接回答问题。
- 可以结合之前的对话理解用户的追问。
- 回答要清晰、准确。
- 必要时可以使用编号或项目符号。

如果文档中没有足够信息：
- 明确告诉用户“根据当前文档内容无法确定”。
- 不要凭空编造。

当前用户问题：

{question}

{history_text}

相关文档内容：

{context}
"""

    return call_deepseek(
        prompt,
        system_prompt="""
你是一名专业的 PDF 文档智能问答助手。

你的任务是根据用户上传的文档回答问题。

必须遵守：
1. 优先使用文档内容。
2. 不要编造文档中不存在的信息。
3. 可以理解多轮上下文。
4. 用户追问时，要结合之前的问题和回答。
5. 使用自然、清晰的中文回答。
6. 不要自己添加“### 💡 回答”这样的标题。
""",
        temperature=0.2
    )


# =========================================================
# 6. 侧边栏
# =========================================================

with st.sidebar:

    st.header("⚙️ 系统状态")

    if DEEPSEEK_API_KEY:
        st.success("DeepSeek API：已连接")
    else:
        st.error("DeepSeek API：未连接")

    st.divider()

    st.subheader("📄 当前文档")

    if st.session_state.file_name:

        st.write(
            f"**文件：** {st.session_state.file_name}"
        )

        st.write(
            f"**页数：** {st.session_state.page_count}"
        )

        st.write(
            f"**大小：** "
            f"{st.session_state.file_size / 1024:.2f} KB"
        )

        st.write(
            f"**文本长度：** "
            f"{len(st.session_state.pdf_text)} 字符"
        )

        st.write(
            f"**文本块：** "
            f"{len(st.session_state.chunks)} 个"
        )

    else:

        st.info("暂未上传 PDF")


# =========================================================
# 7. PDF 上传
# =========================================================

st.header("📤 上传 PDF 文档")

uploaded_file = st.file_uploader(
    "请选择一个 PDF 文件",
    type=["pdf"]
)


if uploaded_file:

    # 判断是否是新文件
    if (
        st.session_state.file_name
        != uploaded_file.name
    ):

        with st.spinner("正在解析 PDF……"):

            raw_text, page_count = extract_pdf(
                uploaded_file
            )

            cleaned_text = clean_text(
                raw_text
            )

            chunks = split_text(
                cleaned_text,
                chunk_size=1500
            )

            st.session_state.pdf_text = cleaned_text

            st.session_state.chunks = chunks

            st.session_state.file_name = uploaded_file.name

            st.session_state.file_size = uploaded_file.size

            st.session_state.page_count = page_count

            # 新文件清空之前的分析结果
            st.session_state.ai_summary = ""

            st.session_state.ai_keywords = []

            st.session_state.ai_structure = ""

            st.session_state.chat_history = []

        st.success("✅ PDF 解析完成")


# =========================================================
# 8. 没有文件时停止
# =========================================================

if not st.session_state.pdf_text:

    st.info("👆 请先上传 PDF 文档")

    st.stop()

    # =========================================================
# 9. 首页系统状态
# =========================================================

st.divider()

st.subheader("📊 系统运行状态")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.session_state.pdf_text:
        st.metric(
            "📄 PDF状态",
            "已加载"
        )
    else:
        st.metric(
            "📄 PDF状态",
            "未加载"
        )

with col2:
    st.metric(
        "📑 文档页数",
        st.session_state.page_count
    )

with col3:
    st.metric(
        "🧠 文本块",
        len(st.session_state.chunks)
    )

with col4:
    if DEEPSEEK_API_KEY:
        st.metric(
            "🤖 DeepSeek",
            "已连接"
        )
    else:
        st.metric(
            "🤖 DeepSeek",
            "未连接"
        )

st.divider()


# =========================================================
# 10. Tabs
# =========================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📊 文档概览",
        "🧠 DeepSeek智能分析",
        "💬 DeepSeek文档问答",
        "📖 原文与报告",
        "🧪 系统测试"
    ]
)


# =========================================================
# TAB 1
# =========================================================

with tab1:

    st.header("📊 文档概览")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "页数",
            st.session_state.page_count
        )

    with col2:
        st.metric(
            "文本字符",
            len(st.session_state.pdf_text)
        )

    with col3:
        st.metric(
            "文本块",
            len(st.session_state.chunks)
        )

    with col4:
        st.metric(
            "文件大小",
            f"{st.session_state.file_size / 1024:.1f} KB"
        )

    st.divider()

    st.subheader("📄 文档基本信息")

    st.write(
        f"**文件名：** {st.session_state.file_name}"
    )

    st.write(
        f"**页数：** {st.session_state.page_count}"
    )

    st.write(
        f"**文本长度：** "
        f"{len(st.session_state.pdf_text)} 字符"
    )

    st.write(
        f"**文本块数量：** "
        f"{len(st.session_state.chunks)}"
    )

    st.divider()

    st.subheader("📝 文本预览")

    preview = st.session_state.pdf_text[:3000]

    st.text_area(
        "前 3000 字",
        preview,
        height=300
    )


# =========================================================
# TAB 2
# =========================================================

with tab2:

    st.header("🧠 DeepSeek智能分析")

    # -----------------------------------------------------
    # 摘要
    # -----------------------------------------------------

    st.subheader("📌 AI智能摘要")

    if st.button(
        "生成 DeepSeek 智能摘要",
        use_container_width=True
    ):

        with st.spinner(
            "DeepSeek 正在生成摘要……"
        ):

            result, error = generate_ai_summary(
                st.session_state.pdf_text
            )

        if error:

            st.error(
                f"🔴 API 调用失败：{error}"
            )

        else:

            st.session_state.ai_summary = result

            st.success(
                "🟢 摘要生成成功"
            )

    if st.session_state.ai_summary:

        st.markdown(
            st.session_state.ai_summary
        )

    # -----------------------------------------------------
    # 关键词
    # -----------------------------------------------------

    st.divider()

    st.subheader("🔑 AI关键词")

    if st.button(
        "生成 DeepSeek 关键词",
        use_container_width=True
    ):

        with st.spinner(
            "DeepSeek 正在提取关键词……"
        ):

            result, error = generate_ai_keywords(
                st.session_state.pdf_text
            )

        if error:

            st.error(
                f"🔴 API 调用失败：{error}"
            )

        else:

            keywords = [
                x.strip()
                for x in result.replace(
                    "，",
                    "、"
                ).split("、")
                if x.strip()
            ]

            st.session_state.ai_keywords = keywords

            st.success(
                "🟢 关键词提取成功"
            )

    if st.session_state.ai_keywords:

        cols = st.columns(3)

        for i, keyword in enumerate(
            st.session_state.ai_keywords
        ):

            with cols[i % 3]:

                st.info(
                    f"🔹 {keyword}"
                )

    # -----------------------------------------------------
    # 文档结构
    # -----------------------------------------------------

    st.divider()

    st.subheader("📑 文档结构分析")

    if st.button(
        "分析文档结构",
        use_container_width=True
    ):

        with st.spinner(
            "DeepSeek 正在分析文档结构……"
        ):

            result, error = generate_ai_structure(
                st.session_state.pdf_text
            )

        if error:

            st.error(
                f"🔴 API 调用失败：{error}"
            )

        else:

            st.session_state.ai_structure = result

            st.success(
                "🟢 文档结构分析完成"
            )

    if st.session_state.ai_structure:

        st.markdown(
            st.session_state.ai_structure
        )


# =========================================================
# TAB 3：多轮问答
# =========================================================

with tab3:

    st.header("💬 DeepSeek文档问答")

    st.caption(
        "可以连续提问，DeepSeek 会结合当前文档和之前的对话理解你的问题。"
    )

    # -----------------------------------------------------
    # API 状态
    # -----------------------------------------------------

    if st.session_state.api_status == "调用成功":

        st.success(
            "🟢 最近一次 DeepSeek API 调用成功"
        )

    elif st.session_state.api_status == "正在调用":

        st.info(
            "🟡 DeepSeek API 正在调用……"
        )

    elif st.session_state.api_status == "调用失败":

        st.error(
            "🔴 最近一次 DeepSeek API 调用失败"
        )

    else:

        st.info(
            "⚪ 等待 DeepSeek API 调用"
        )

    # -----------------------------------------------------
    # 清空聊天
    # -----------------------------------------------------

    col1, col2 = st.columns(
        [5, 1]
    )

    with col2:

        if st.button(
            "🗑️ 清空",
            use_container_width=True
        ):

            st.session_state.chat_history = []

            st.rerun()

    # -----------------------------------------------------
    # 显示历史聊天
    # -----------------------------------------------------

    if st.session_state.chat_history:

        for item in st.session_state.chat_history:

            with st.chat_message(
                "user"
            ):

                st.markdown(
                    item["question"]
                )

            with st.chat_message(
                "assistant"
            ):

                st.markdown(
                    item["answer"]
                )

    else:

        st.info(
            "还没有聊天记录。你可以先问：\n\n"
            "“这个文件主要讲了什么？”"
        )

    # -----------------------------------------------------
    # 聊天输入框
    # -----------------------------------------------------

    question = st.chat_input(
        "请输入关于当前 PDF 的问题……"
    )

    if question:

        # 显示用户问题
        with st.chat_message(
            "user"
        ):

            st.markdown(question)

        # 调用 DeepSeek
        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "DeepSeek 正在思考……"
            ):

                answer, error = generate_ai_answer(
                    question,
                    st.session_state.chunks
                )

            if error:

                st.error(
                    f"🔴 DeepSeek API 调用失败：\n\n{error}"
                )

            else:

                # 清除 DeepSeek 可能生成的标题
                answer = re.sub(
                    r"^\s*#+\s*(💡\s*)?(回答|答案)\s*:?\s*",
                    "",
                    answer,
                    flags=re.IGNORECASE
                ).strip()

                st.markdown(answer)

                # 保存聊天记录
                st.session_state.chat_history.append(
                    {
                        "question": question,
                        "answer": answer
                    }
                )

                st.success(
                    "🟢 DeepSeek API 调用成功"
                )


# =========================================================
# TAB 4：原文与报告
# =========================================================

with tab4:

    st.header("📖 原文与报告")

    # -----------------------------------------------------
    # 原文
    # -----------------------------------------------------

    st.subheader("📄 PDF文本")

    st.text_area(
        "提取后的文本",
        st.session_state.pdf_text,
        height=500
    )

    st.divider()

    # -----------------------------------------------------
    # Word 报告
    # -----------------------------------------------------

    st.subheader("📝 生成 Word 分析报告")

    if st.button(
        "📥 下载 Word 分析报告",
        use_container_width=True
    ):

        doc = Document()

        # 标题
        title = doc.add_heading(
            "PDF智能摘要与文档问答系统分析报告",
            level=0
        )

        title.alignment = WD_ALIGN_PARAGRAPH.CENTER

        # -------------------------------------------------
        # 一、基本信息
        # -------------------------------------------------

        doc.add_heading(
            "一、文档基本信息",
            level=1
        )

        doc.add_paragraph(
            f"文件名称：{st.session_state.file_name}"
        )

        doc.add_paragraph(
            f"PDF页数：{st.session_state.page_count}"
        )

        doc.add_paragraph(
            f"文件大小："
            f"{st.session_state.file_size / 1024:.2f} KB"
        )

        doc.add_paragraph(
            f"文本长度："
            f"{len(st.session_state.pdf_text)} 字符"
        )

        doc.add_paragraph(
            f"文本块数量："
            f"{len(st.session_state.chunks)}"
        )

        # -------------------------------------------------
        # 二、AI摘要
        # -------------------------------------------------

        doc.add_heading(
            "二、DeepSeek智能摘要",
            level=1
        )

        if st.session_state.ai_summary:

            doc.add_paragraph(
                st.session_state.ai_summary
            )

        else:

            doc.add_paragraph(
                "尚未生成AI摘要。"
            )

        # -------------------------------------------------
        # 三、关键词
        # -------------------------------------------------

        doc.add_heading(
            "三、核心关键词",
            level=1
        )

        if st.session_state.ai_keywords:

            doc.add_paragraph(
                "、".join(
                    st.session_state.ai_keywords
                )
            )

        else:

            doc.add_paragraph(
                "尚未生成关键词。"
            )

        # -------------------------------------------------
        # 四、结构分析
        # -------------------------------------------------

        doc.add_heading(
            "四、文档结构分析",
            level=1
        )

        if st.session_state.ai_structure:

            doc.add_paragraph(
                st.session_state.ai_structure
            )

        else:

            doc.add_paragraph(
                "尚未进行文档结构分析。"
            )

        # -------------------------------------------------
        # 五、多轮问答
        # -------------------------------------------------

        doc.add_heading(
            "五、DeepSeek文档问答记录",
            level=1
        )

        if st.session_state.chat_history:

            for index, item in enumerate(
                st.session_state.chat_history,
                start=1
            ):

                doc.add_paragraph(
                    f"问题 {index}："
                    f"{item['question']}"
                )

                doc.add_paragraph(
                    f"回答："
                    f"{item['answer']}"
                )

        else:

            doc.add_paragraph(
                "暂无问答记录。"
            )

        # -------------------------------------------------
        # 六、系统测试
        # -------------------------------------------------

        doc.add_heading(
            "六、系统测试情况",
            level=1
        )

        doc.add_paragraph(
            "1. PDF文件上传测试：正常"
        )

        doc.add_paragraph(
            "2. PDF文本提取测试：正常"
        )

        doc.add_paragraph(
            "3. 文本清洗与分块测试：正常"
        )

        doc.add_paragraph(
            "4. DeepSeek智能摘要测试：支持"
        )

        doc.add_paragraph(
            "5. DeepSeek关键词提取测试：支持"
        )

        doc.add_paragraph(
            "6. 文档结构分析测试：支持"
        )

        doc.add_paragraph(
            "7. 多轮文档问答测试：支持"
        )

        # -------------------------------------------------
        # 七、项目总结
        # -------------------------------------------------

        doc.add_heading(
            "七、项目总结",
            level=1
        )

        doc.add_paragraph(
            "本系统基于 Python、Streamlit、PyMuPDF "
            "和 DeepSeek API 构建，实现了 PDF 文档上传、"
            "文本提取、文本处理、智能摘要、关键词提取、"
            "文档结构分析以及多轮文档问答等功能。"
        )

        doc.add_paragraph(
            "通过多轮问答功能，用户可以围绕上传的 PDF "
            "持续提出问题，系统结合文档内容和历史对话"
            "生成回答，从而提高文档信息获取效率。"
        )

        # -------------------------------------------------
        # 输出
        # -------------------------------------------------

        buffer = BytesIO()

        doc.save(buffer)

        buffer.seek(0)

        st.download_button(
            label="⬇️ 下载分析报告.docx",
            data=buffer,
            file_name="PDF智能摘要与文档问答系统分析报告.docx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "wordprocessingml.document"
            )
        )
        # =========================================================
# TAB 5：系统测试
# =========================================================

with tab5:

    st.header("🧪 系统功能测试")

    st.markdown(
        """
        通过自动检查当前系统状态，对 PDF 解析、文本处理、
        DeepSeek API、AI 分析、多轮问答以及 Word 报告等功能
        进行测试。
        """
    )

    st.divider()

    # =====================================================
    # 测试结果函数
    # =====================================================

    test_results = []

    # -----------------------------------------------------
    # 1. PDF 上传测试
    # -----------------------------------------------------

    if st.session_state.file_name:
        test_results.append(
            ("PDF文件上传", "通过", "已成功加载 PDF 文件")
        )
    else:
        test_results.append(
            ("PDF文件上传", "未通过", "尚未上传 PDF 文件")
        )

    # -----------------------------------------------------
    # 2. PDF 文本提取
    # -----------------------------------------------------

    if len(st.session_state.pdf_text) > 0:
        test_results.append(
            (
                "PDF文本提取",
                "通过",
                f"成功提取 {len(st.session_state.pdf_text)} 个字符"
            )
        )
    else:
        test_results.append(
            (
                "PDF文本提取",
                "未通过",
                "没有检测到 PDF 文本"
            )
        )

    # -----------------------------------------------------
    # 3. 文本分块
    # -----------------------------------------------------

    if len(st.session_state.chunks) > 0:
        test_results.append(
            (
                "文本分块",
                "通过",
                f"当前共有 {len(st.session_state.chunks)} 个文本块"
            )
        )
    else:
        test_results.append(
            (
                "文本分块",
                "未通过",
                "没有生成文本块"
            )
        )

    # -----------------------------------------------------
    # 4. DeepSeek API
    # -----------------------------------------------------

    if DEEPSEEK_API_KEY:
        test_results.append(
            (
                "DeepSeek API配置",
                "通过",
                "已检测到 API Key"
            )
        )
    else:
        test_results.append(
            (
                "DeepSeek API配置",
                "未通过",
                "没有检测到 API Key"
            )
        )

    # -----------------------------------------------------
    # 5. AI 摘要
    # -----------------------------------------------------

    if st.session_state.ai_summary:
        test_results.append(
            (
                "AI智能摘要",
                "通过",
                "已经生成摘要"
            )
        )
    else:
        test_results.append(
            (
                "AI智能摘要",
                "待测试",
                "点击“生成 DeepSeek 智能摘要”后再次检查"
            )
        )

    # -----------------------------------------------------
    # 6. 关键词
    # -----------------------------------------------------

    if st.session_state.ai_keywords:
        test_results.append(
            (
                "AI关键词提取",
                "通过",
                f"已经提取 {len(st.session_state.ai_keywords)} 个关键词"
            )
        )
    else:
        test_results.append(
            (
                "AI关键词提取",
                "待测试",
                "点击“生成 DeepSeek 关键词”后再次检查"
            )
        )

    # -----------------------------------------------------
    # 7. 文档结构
    # -----------------------------------------------------

    if st.session_state.ai_structure:
        test_results.append(
            (
                "文档结构分析",
                "通过",
                "已经生成结构分析"
            )
        )
    else:
        test_results.append(
            (
                "文档结构分析",
                "待测试",
                "点击“分析文档结构”后再次检查"
            )
        )

    # -----------------------------------------------------
    # 8. 多轮问答
    # -----------------------------------------------------

    if st.session_state.chat_history:
        test_results.append(
            (
                "多轮文档问答",
                "通过",
                f"当前已有 {len(st.session_state.chat_history)} 轮对话"
            )
        )
    else:
        test_results.append(
            (
                "多轮文档问答",
                "待测试",
                "进入问答页面进行提问后再次检查"
            )
        )

    # -----------------------------------------------------
    # 显示测试结果
    # -----------------------------------------------------

    st.subheader("📋 测试结果")

    for name, status, message in test_results:

        if status == "通过":

            st.success(
                f"✅ {name}：{status}\n\n{message}"
            )

        elif status == "未通过":

            st.error(
                f"❌ {name}：{status}\n\n{message}"
            )

        else:

            st.warning(
                f"⚠️ {name}：{status}\n\n{message}"
            )

    # =====================================================
    # 测试统计
    # =====================================================

    passed = sum(
        1 for item in test_results
        if item[1] == "通过"
    )

    total = len(test_results)

    st.divider()

    st.subheader("📊 测试统计")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "测试项目",
            total
        )

    with col2:
        st.metric(
            "已通过",
            passed
        )

    with col3:

        rate = (
            passed / total * 100
            if total > 0
            else 0
        )

        st.metric(
            "当前通过率",
            f"{rate:.0f}%"
        )

    # =====================================================
    # 测试说明
    # =====================================================

    st.divider()

    st.subheader("💡 测试流程")

    st.markdown(
        """
        **建议按照以下顺序进行完整测试：**

        1. 上传一个 PDF 文件。
        2. 查看 PDF 文本是否成功提取。
        3. 点击“生成 DeepSeek 智能摘要”。
        4. 点击“生成 DeepSeek 关键词”。
        5. 点击“分析文档结构”。
        6. 进入“DeepSeek文档问答”。
        7. 连续提出 2～3 个相关问题。
        8. 返回本页面查看测试结果。
        9. 进入“原文与报告”。
        10. 下载 Word 分析报告。
        """
    )