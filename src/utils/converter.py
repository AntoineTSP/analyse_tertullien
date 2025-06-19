import xml.etree.ElementTree as ET
from io import StringIO
from pathlib import Path
from xml.etree import cElementTree as ET

import lxml.etree as etree
from lxml import etree, html
from tensorflow.keras import Model

NS = {"tei": "http://www.tei-c.org/ns/1.0"}

ANNOTATIONS = {
    "metaphore": {},
    "comparaison": {},
    "personnification": {},
    "enargeia": {},
    "dialogisme": {},
    "deductio_ad_absurdum": {},
    "symbolisme": {},
    "exemple": {},
    "militaire": {},
    "retorsion_inversée": {},
    "irreel": {},
    "epiphoneme": {},
    "ab_absurdo": {},
    "ab_origine": {},
    "calembour": {},
    "dicton": {},
    "prosopopeia": {},
    "scene": {},
    "corps": {},
}


def get_model_summary(model: Model) -> str:
    """
    Generates a summary of the Keras model.
    Args:
        model (Model): A Keras model instance.
    Returns:
        str: A string representation of the model summary.
    """

    stream = StringIO()
    model.summary(print_fn=lambda x: stream.write(x + "\n"))
    summary_str = stream.getvalue()
    stream.close()
    return summary_str


def parse_tei(xml_path: str) -> dict:
    """
    Parses a TEI XML file and extracts the title and body as HTML.
    Args:
        xml_path (str): The file path to the TEI XML file.
    Returns:
        dict: A dictionary containing the title and body as HTML.
    """

    tree = etree.parse(xml_path)
    root = tree.getroot()

    # Extract the body or whole text as HTML string
    body = root.find(".//{http://www.tei-c.org/ns/1.0}body")

    # Convert the body to HTML (with tags like <span>, <p>, etc.)
    body_html = etree.tostring(body, encoding="unicode", method="html")

    return {"title": root.findtext(".//{http://www.tei-c.org/ns/1.0}title"), "body_html": body_html}


def parse_full_tei(xml_path: str) -> dict:
    """
    Parses a TEI XML file and extracts metadata, witness information, and text body.

    Args:
        xml_path (str): The file path to the TEI XML file.

    Returns:
        dict: A dictionary containing the parsed metadata, witness information, and text body.
    """

    tree = etree.parse(Path(xml_path))
    root = tree.getroot()

    # === METADATA ===
    title = root.find(".//tei:titleStmt/tei:title", NS).text
    # author_el = root.find(".//tei:titleStmt/tei:author//tei:persName", NS)
    # author = " ".join(author_el.xpath(".//text()", namespaces=NS)) if author_el is not None else ""
    author_el = root.find(".//tei:titleStmt/tei:author", NS)
    author_fix = " ".join(author_el.xpath(".//text()", namespaces=NS)) if author_el is not None else ""

    # Extract respStmt
    resp_stmts = []
    for stmt in root.findall(".//tei:respStmt", NS):
        name = stmt.find("tei:persName", NS)
        resp = stmt.find("tei:resp", NS)
        resp_date = resp.find("tei:date", NS) if resp is not None else None
        resp_text = (resp.text or "").strip() if resp is not None else ""
        resp_stmts.append(
            {
                "name": name.text if name is not None else "",
                "id": name.get("{http://www.w3.org/XML/1998/namespace}id") if name is not None else None,
                "date": resp_date.text if resp_date is not None else "",
                "responsibility": resp_text,
            }
        )

    # Extract listWit → witness → bibl
    witnesses = []
    for wit_var in root.findall(".//tei:sourceDesc//tei:listWit/tei:witness", NS):
        wit_id = wit_var.get("{http://www.w3.org/XML/1998/namespace}id")
        bibl = wit_var.find("tei:bibl", NS)
        author = bibl.find("tei:author", NS)
        forename = author.find("tei:forename", NS)
        surname = author.find("tei:surname", NS)
        title_el = bibl.find("tei:title/tei:title", NS)
        publisher_names = bibl.findall("tei:publisher", NS)
        pub_place = bibl.find("tei:pubPlace", NS)
        date = bibl.find("tei:date", NS)
        pages = bibl.find("tei:biblScope", NS)
        link = bibl.find("tei:ptr", NS)

        witnesses.append(
            {
                "id": wit_id,
                "author": {
                    "forename": forename.text if forename is not None else "",
                    "surname": surname.text if surname is not None else "",
                },
                "title": title_el.text if title_el is not None else "",
                "publishers": [p.text for p in publisher_names if p.text],
                "pub_place": pub_place.text if pub_place is not None else "",
                "date": date.text if date is not None else "",
                "pages": pages.text if pages is not None else "",
                "link": link.get("target") if link is not None else "",
            }
        )

    # Editorial and language
    project_desc = root.find(".//tei:encodingDesc/tei:projectDesc/tei:p", NS)
    normalization = root.find(".//tei:editorialDecl/tei:normalization/tei:p", NS)
    punctuation = root.find(".//tei:editorialDecl/tei:punctuation/tei:p", NS)
    languages = root.findall(".//tei:profileDesc/tei:langUsage/tei:language", NS)
    langs = [f"{lang.get('ident')}: {lang.text}" for lang in languages]
    licence = root.find(".//tei:licence", NS)

    # === TEXT BODY ===
    text_body = root.find(".//tei:text", NS)
    body_html = etree.tostring(text_body, encoding="unicode", method="html")

    return {
        "title": title,
        "author": author_fix,
        "resp_stmts": resp_stmts,
        "witnesses": witnesses,
        "project_desc": project_desc.text if project_desc is not None else "",
        "normalization": normalization.text if normalization is not None else "",
        "punctuation": punctuation.text if punctuation is not None else "",
        "languages": langs,
        "licence": licence.text.strip() if licence is not None else "",
        "body_html": body_html,
    }


def add_title_to_spans(html_string: str) -> str:
    """
    Adds a title attribute to all <span> elements with an 'ana' attribute in the provided HTML string.

    Args:
        html_string (str): The HTML string to process.

    Returns:
        str: The modified HTML string with title attributes added to <span> elements.
    """

    # Parse the HTML string into an lxml tree
    tree = html.fromstring(html_string)

    # Find all <span> elements with the 'ana' attribute
    for span in tree.xpath("//span[@ana]"):
        ana_value = span.get("ana")
        if ana_value:
            # Add a title attribute with the same value
            span.set("title", ana_value)

    # Serialize back to string, preserving the inner HTML structure
    return html.tostring(tree, encoding="unicode", method="html")


def get_all_annotations(xml_path: str) -> dict:
    """
    Extracts annotations from an XML file and organizes them into a structured dictionary.

    Args:
        xml_path (str): The file path to the XML file containing annotations.

    Returns:
        dict: A dictionary where keys are annotation types and values are dictionaries
              containing chapter headers as keys and lists of annotation contents as values.
    """

    xmlstr = ET.parse(xml_path)
    root = xmlstr.getroot()
    new_root = root[1][0]

    annotations_copy = ANNOTATIONS.copy()

    for i in range(len(new_root[2])):
        chapter_header = f"Chapitre {i + 1}"

        for key, value in annotations_copy.items():
            value[chapter_header] = []

        for deep6 in new_root[2][i][0]:
            ana = deep6.attrib.get("ana")
            if ana and ana in annotations_copy:
                content = " ".join((deep6.text or "").split())
                annotations_copy[ana][chapter_header].append(content)

    return annotations_copy


def get_all_annotations_categories() -> list:
    """
    Returns a list of all annotation categories defined in the ANNOTATIONS dictionary.

    Returns:
        list: A list of annotation category names.
    """
    return list(ANNOTATIONS.keys())
