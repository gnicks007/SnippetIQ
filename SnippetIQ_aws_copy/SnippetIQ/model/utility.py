import requests
from bs4 import BeautifulSoup
from flask import Markup
from SnippetIQ.model.snippetheader import SNIPPET_HEADER, SNIPPET_BODY
import os

highlight_css = """
    .highlightme { background-color:#FFFF00; }
    """
highlight_start = '<span class="highlightme">'
highlight_end = '</span>'


def highlight(url):
    r = requests.get(url)
    html_text = r.text
    soup = BeautifulSoup(html_text, "lxml")
    headSnippetSoup = BeautifulSoup(SNIPPET_HEADER, "lxml")
    bodySnippetSoup = BeautifulSoup(SNIPPET_BODY, "lxml")

    head_snippet = removeTags(headSnippetSoup)
    body_snippet = removeTags(bodySnippetSoup)

    head = soup.head
    head.insert(1, soup.new_tag('style', type='text/css'))
    head.style.append(highlight_css)
    head.insert(0, head_snippet)
    soup.head = head


    # soup.body = add_text(soup, " I didn't find any helpful answers here")
    body = soup.body
    body.insert(0, body_snippet)
    soup.body = body

    newsoup = Markup(soup)
    html = soup.prettify("utf-8")

    templates = "/SnippetIQ/templates/output_template.html"
    pwd = os.getcwd()
    filename = pwd+templates

    with open(filename, "wb") as file:
        file.write(html)

    return newsoup

def add_text(soup, input_txt):
    body = soup.body
    new_tag = soup.new_tag('h3')
    new_tag.insert(0, soup.new_tag('br'))
    new_tag.insert(0, soup.new_tag('br'))
    new_tag.insert(0, soup.new_tag('br'))
    new_tag.insert(0, soup.new_tag('br'))
    new_tag2 = soup.new_tag('span')
    new_tag2['class'] = 'highlightme'
    new_tag2.string = input_txt
    new_tag.append(new_tag2)

    body.insert(0, new_tag)

    return body

# remove invalid tags from a soup object
def removeTags(soup):
    invalid_tags = ['head', 'html', 'body']
    for tag in invalid_tags:
        for match in soup.findAll(tag):
            match.replaceWithChildren()
    return soup
