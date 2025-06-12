import re
from scholarly import scholarly
import time
import os
import random

def extract_paper_titles(md_content):
    # 匹配arXiv链接后面的标题，包括已有的引用量信息
    pattern = r'\[arXiv.*?\]\((.*?)\).*?,\s*(.*?)(?:\[Citations: \d+\]|$)'
    matches = re.finditer(pattern, md_content)
    papers = []
    skipped_count = 0
    for match in matches:
        url = match.group(1)
        title = match.group(2).strip()
        # 检查是否已经有引用量信息
        citation_pattern = fr'\[arXiv.*?\]\({re.escape(url)}\).*?,\s*{re.escape(title)}\s*\[Citations: (\d+)\]'
        citation_match = re.search(citation_pattern, md_content)
        if citation_match:
            citations = int(citation_match.group(1))
            print(f"论文 {title} 已有引用量信息 [Citations: {citations}]，跳过")
            skipped_count += 1
            continue
        papers.append((url, title))
    print(f"找到 {len(papers)} 篇需要更新的论文，跳过 {skipped_count} 篇已有引用量的论文")
    return papers

def get_citation_count(paper_url, max_retries=3):
    for attempt in range(max_retries):
        try:
            # 从arXiv URL中提取论文ID
            paper_id = paper_url.split('/')[-1]
            print(f"正在查询论文 {paper_id} 的引用量... (尝试 {attempt + 1}/{max_retries})")
            
            # 使用scholarly搜索论文
            search_query = scholarly.search_pubs(paper_id)
            paper = next(search_query, None)
            
            if paper:
                citations = paper.get('num_citations', 0)
                print(f"论文 {paper_id} 的引用量为: {citations}")
                return citations
                
            print(f"未找到论文 {paper_id} 的引用信息")
            return 0
            
        except Exception as e:
            print(f"获取论文 {paper_url} 引用量时出错: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = random.uniform(5, 10)  # 随机等待5-10秒
                print(f"等待 {wait_time:.1f} 秒后重试...")
                time.sleep(wait_time)
            else:
                print(f"达到最大重试次数，跳过此论文")
                return 0

def update_readme():
    print("开始更新README.md文件...")
    
    # 读取README.md文件
    readme_path = 'Teleoperation/README.md'
    if not os.path.exists(readme_path):
        print(f"错误：找不到文件 {readme_path}")
        return
        
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
    print(f"成功读取 {readme_path}")

    # 提取所有论文标题
    paper_titles = extract_paper_titles(content)
    
    # 更新内容
    updated_count = 0
    for i, (url, title) in enumerate(paper_titles, 1):
        print(f"\n处理第 {i}/{len(paper_titles)} 篇论文: {title}")
        
        # 添加随机延迟以避免被Google Scholar限制
        wait_time = random.uniform(3, 5)
        print(f"等待 {wait_time:.1f} 秒...")
        time.sleep(wait_time)
        
        citations = get_citation_count(url)
        # 无论引用量是否为0，都更新
        pattern = fr'\[arXiv.*?\]\({re.escape(url)}\).*?,\s*{re.escape(title)}'
        replacement = f'[arXiv]({url}), {title} [Citations: {citations}]'
        new_content = re.sub(pattern, replacement, content)
        if new_content != content:
            content = new_content
            updated_count += 1
            print(f"已更新论文 {title} 的引用量")
            
            # 每更新5篇论文就保存一次
            if updated_count % 5 == 0:
                with open(readme_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"已保存当前进度，已更新 {updated_count} 篇论文")

    # 写回文件
    if updated_count > 0:
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"\n成功更新了 {updated_count} 篇论文的引用量")
    else:
        print("\n没有找到需要更新的论文")

if __name__ == "__main__":
    try:
        update_readme()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n程序发生错误: {str(e)}")
    finally:
        print("\n程序结束") 