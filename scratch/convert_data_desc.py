# -*- coding: utf-8 -*-
"""
Script to fetch https://topaneu-26.grand-challenge.org/data/ and convert
the exact page content into topaneu_release/data_description.md with full fidelity.
"""
import urllib.request
import pathlib
import sys
from bs4 import BeautifulSoup, NavigableString, Tag

OUT_PATH = pathlib.Path(r"d:\NLP_Project\topaneu_release\data_description.md")
URL = "https://topaneu-26.grand-challenge.org/data/"

def inline_to_md(tag):
    """Convert an inline element and its children to Markdown text."""
    if isinstance(tag, NavigableString):
        return str(tag)
    
    if tag.name == 'a':
        href = tag.get('href', '')
        text = ''.join(inline_to_md(c) for c in tag.children).strip()
        if not text:
            return ''
        if not href:
            return text
        if href.startswith('/'):
            href = 'https://topaneu-26.grand-challenge.org' + href
        return f"[{text}]({href})"
    
    elif tag.name == 'code':
        text = ''.join(inline_to_md(c) for c in tag.children)
        return f"`{text}`"
    
    elif tag.name in ('strong', 'b'):
        text = ''.join(inline_to_md(c) for c in tag.children)
        return f"**{text}**"
    
    elif tag.name in ('em', 'i'):
        text = ''.join(inline_to_md(c) for c in tag.children)
        return f"*{text}*"
    
    elif tag.name == 'br':
        return "\n"
    
    else:
        return ''.join(inline_to_md(c) for c in tag.children)

def table_to_md(table):
    """Convert HTML table to Markdown table."""
    lines = []
    thead = table.find('thead')
    if thead:
        headers = []
        for th in thead.find_all(['th', 'td']):
            headers.append(inline_to_md(th).strip().replace('\n', ' '))
        lines.append('| ' + ' | '.join(headers) + ' |')
        lines.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')
    
    tbody = table.find('tbody') or table
    for tr in tbody.find_all('tr'):
        cells = []
        for td in tr.find_all(['td', 'th']):
            cells.append(inline_to_md(td).strip().replace('\n', ' '))
        if cells:
            lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)

def convert_page():
    req = urllib.request.Request(URL, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as resp:
        raw = resp.read()
        encoding = resp.headers.get_content_charset() or 'utf-8'
        html = raw.decode(encoding, errors='replace')

    soup = BeautifulSoup(html, 'html.parser')
    pc = soup.find('div', id='pageContainer')
    if not pc:
        raise ValueError("Could not find div with id='pageContainer'")

    # Remove anchor links with class 'headerlink'
    for a in pc.find_all('a', class_='headerlink'):
        a.decompose()

    md_blocks = []

    for child in pc.children:
        if isinstance(child, NavigableString):
            text = str(child).strip()
            if text:
                md_blocks.append(text)
            continue

        if not isinstance(child, Tag):
            continue

        name = child.name

        if name == 'h1':
            text = ''.join(inline_to_md(c) for c in child.children).strip()
            md_blocks.append(f"# {text}")
        elif name == 'h2':
            text = ''.join(inline_to_md(c) for c in child.children).strip()
            md_blocks.append(f"## {text}")
        elif name == 'h3':
            text = ''.join(inline_to_md(c) for c in child.children).strip()
            md_blocks.append(f"### {text}")
        elif name == 'h4':
            text = ''.join(inline_to_md(c) for c in child.children).strip()
            md_blocks.append(f"#### {text}")
        elif name == 'p':
            text = ''.join(inline_to_md(c) for c in child.children).strip()
            if text:
                md_blocks.append(text)
        elif name == 'table':
            md_blocks.append(table_to_md(child))
        elif name in ('ul', 'ol'):
            list_items = []
            for i, li in enumerate(child.find_all('li', recursive=False)):
                prefix = f"{i+1}." if name == 'ol' else "-"
                item_text = ''.join(inline_to_md(c) for c in li.children).strip()
                list_items.append(f"{prefix} {item_text}")
            if list_items:
                md_blocks.append('\n'.join(list_items))
        elif name in ('pre', 'div'):
            # Check if it has a pre or code inside
            pre = child if name == 'pre' else child.find('pre')
            if pre:
                code_text = pre.get_text()
                md_blocks.append(f"```\n{code_text.strip()}\n```")
            else:
                text = ''.join(inline_to_md(c) for c in child.children).strip()
                if text:
                    md_blocks.append(text)
        elif name == 'hr':
            md_blocks.append("---")
        else:
            text = ''.join(inline_to_md(c) for c in child.children).strip()
            if text:
                md_blocks.append(text)

    final_content = '\n\n'.join(md_blocks) + '\n'

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        f.write(final_content)

    print(f"Successfully wrote {OUT_PATH} ({len(final_content)} characters, {len(final_content.splitlines())} lines)")

if __name__ == '__main__':
    convert_page()
