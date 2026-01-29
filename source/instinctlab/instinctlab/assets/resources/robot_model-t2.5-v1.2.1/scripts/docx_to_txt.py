from docx import Document

def docx_to_txt(docx_path, txt_path):
    """
    将docx文件转换为txt文件
    """
    try:
        # 读取Word文档
        doc = Document(docx_path)
        
        # 提取所有段落文本
        text_content = []
        for paragraph in doc.paragraphs:
            text_content.append(paragraph.text)
        
        # 将文本写入TXT文件
        with open(txt_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write('\n'.join(text_content))
        
        print(f"成功将 {docx_path} 转换为 {txt_path}")
        
    except Exception as e:
        print(f"转换失败: {e}")

# 使用示例
docx_to_txt('./txt.docx', './output.txt')