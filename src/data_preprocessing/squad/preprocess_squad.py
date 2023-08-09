"""
Processing functions for SQuAD dataset
"""

import json
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import RegexpTokenizer
from subprocess import run
from typing import List, Dict, Optional
import os
from tqdm import tqdm
import time
import pandas as pd
import sys
from datetime import datetime
import logging

# Init logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s: %(message)s"
)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    logger.info("NLTK stopwords not found. Downloading the package...")
    nltk.download("stopwords")
    logger.info("NLTK stopwords downloaded.")

from nltk.corpus import stopwords

###################################################
# Constants
###################################################

SRC_DIRECTORY = os.path.split(os.path.split(os.path.dirname(__file__))[0])[0]
WKHTMLTOPDF_PATH = r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"
WIKI_BASE_URL = "https://en.wikipedia.org/w/api.php"
WIKI_API_BASE_URL = "https://en.wikipedia.org/api/rest_v1/page/html"

###################################################

sys.path.append(SRC_DIRECTORY)
sys.path.append(os.path.join(SRC_DIRECTORY, "data_processing"))
from data_processing import save_images_words_bboxes


###################################################
# Common helper functions
###################################################


def squad_load(data_path: str) -> List[Dict]:
    """
    Loads original SQuAD train/dev/test json file.

    :param data_path: path to data file to read json from
    :return: list of documents in the dataset
    """
    with open(data_path, "r") as f:
        docs = json.load(f)["data"]
    return docs


def remove_old_files(file_list: List[str]):
    """
    Deletes files in given list.

    :param file_list: list of filepaths
    """
    for file_path in file_list:
        if os.path.exists(file_path):
            os.remove(file_path)


def refactor_json_files(json_paths: List[str]):
    """
    Adds "[" and "]" at the start and end respectively to JSONs whose paths
    are given as a list.

    :param json_paths: list of JSON file paths
    """
    for json_path in json_paths:
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                data = f.read()
            with open(json_path, "w", encoding="utf-8") as f:
                edited_json_str = "[" + str(data[:-1]) + "]"
                f.write(edited_json_str)


###################################################
# Webpage validation
###################################################

# main


def collect_docs(
    data: List[Dict],
    output_dir: str,
    threshold_min: float = 1.0,
    threshold_max: float = 1.0,
    html_version: str = "2017",
    sleep_time: int = 5,
    include_references: bool = False,
):
    """
    Validates the relevancy of wikipages whose titles are found in data
    (dataset object) using validate_qas function, cleans and saves HTMLs
    of the all the wikipages, and finally saves the data found in the
    webpages into 2 JSON files.

    :param data: data from dataset as list of individual webpage data
    :param output_dir: directory in which the JSON files are to be saved
    :param threshold_min: match percentage threshold (0 to 1) above which
        question-answer pair is said to be valid
    :param threshold_max: match percentage threshold (0 to 1) below which
        question-answer pair is said to be valid
    :param html_version: "online" or "2017"
    :param sleep_time: sleep time between scarping each webpage
    :param include_references: whether to include citation tags like '[5]'
    """
    
    logs_dir = os.path.join(output_dir, "unsuccessful_logs")
    html_dir = os.path.join(output_dir, "htmls")

    for dir in [logs_dir, html_dir]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    # to log unsuccessful steps
    current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
    unsuccessful_qas_path = os.path.join(
        logs_dir, f"{current_datetime }_unsuccessful_qas.json"
    )
    unsuccessful_doc_path = os.path.join(
        logs_dir, f"{current_datetime }_unsuccessful_doc.json"
    )

    # file to save scraped data - paras, title
    doc_paras_path = os.path.join(output_dir, "doc_paras.json")
    # file to save qas
    qas_path = os.path.join(output_dir, "qas.json")

    remove_old_files(
        [doc_paras_path, qas_path, unsuccessful_qas_path, unsuccessful_doc_path]
    )

    count_answerable_answers = 0

    # for each document/webpage
    for doc_id, doc in tqdm(enumerate(data), total=len(data), desc="Progress"):
        try:
            # get HTML strings
            html_path = os.path.join(html_dir, f"{doc_id}.html")
            if html_version == "online":
                time.sleep(sleep_time)
                html = filter_sections_online_html(
                    get_online_html(
                        get_wikipedia_url(doc["title"]), include_ref=include_references
                    )
                )
            else:
                html = filter_sections_2017_html(
                    get_html_by_year(doc["title"]), include_ref=include_references
                )

            # save HTML
            create_html_file(html, html_path)

            # parse HTML
            soup = BeautifulSoup(html, "html.parser")
            html_paras = [
                para.text.replace("\n", " ")
                for para in soup.select("body")[0].find_all("p")
            ]

            with open(doc_paras_path, "a", encoding="utf-8") as f:
                json.dump(
                    {"title": doc["title"], "paras": html_paras, "doc_id": doc_id}, f
                )
                f.write(",")

            for ref_para in doc["paragraphs"]:
                ref_para_text = ref_para["context"].lower()
                for qas in ref_para["qas"]:

                    # for answerable question
                    if not qas["is_impossible"]:
                        for answer_obj in qas["answers"]:
                            try:
                                response = validate_qas(
                                    ref_para_text,
                                    answer_obj,
                                    html_paras,
                                    threshold_min=threshold_min,
                                    threshold_max=threshold_max,
                                )
                                # save question-answers
                                with open(qas_path, "a", encoding="utf-8") as f:
                                    json.dump(
                                        {
                                            "is_validated": response["is_validated"],
                                            "percent_match": response[
                                                "match_percentage"
                                            ],
                                            "idx_para_matched": response[
                                                "matched_para_id"
                                            ],
                                            "answer_start_char_idx": response[
                                                "answer_start_char_idx"
                                            ],
                                            "answer_end_char_idx": response[
                                                "answer_end_char_idx"
                                            ],
                                            "question": qas["question"],
                                            "answer_text": answer_obj["text"],
                                            "doc_id": doc_id,
                                            "doc_title": doc["title"],
                                            "qas_id": qas["id"],
                                            "is_impossible": False,
                                            "plausible_answers": list(),
                                        },
                                        f,
                                    )
                                    f.write(",")
                                count_answerable_answers += 1
                            except Exception as e:
                                # log failed - question-answer validation problem
                                with open(
                                    unsuccessful_qas_path, "a", encoding="utf-8"
                                ) as f:
                                    logger.warning(
                                        f'Scraping of <qas_id:{qas["id"]}> in "{doc["title"]}" wiki article failed'
                                    )
                                    json.dump(
                                        {
                                            "qas_id": qas["id"],
                                            "doc_id": doc_id,
                                            "doc_title": doc["title"],
                                            "error": str(e),
                                        },
                                        f,
                                    )
                                    f.write(",")

                    # for unanswerable questions
                    else:
                        para_id = get_impossible_qas_para(ref_para_text, html_paras)
                        # save qas
                        with open(qas_path, "a", encoding="utf-8") as f:
                            json.dump(
                                {
                                    "is_validated": True,
                                    "percent_match": None,
                                    "idx_para_matched": para_id,
                                    "answer_start_char_idx": None,
                                    "answer_end_char_idx": None,
                                    "question": qas["question"],
                                    "answer_text": None,
                                    "doc_id": doc_id,
                                    "doc_title": doc["title"],
                                    "qas_id": qas["id"],
                                    "is_impossible": True,
                                    "plausible_answers": qas["plausible_answers"],
                                },
                                f,
                            )
                            f.write(",")

        # problem in scraping
        except Exception as e:
            # log failed
            with open(unsuccessful_doc_path, "a", encoding="utf-8") as f:
                logger.warning(f'Scraping of "{doc["title"]}" wiki article failed')
                json.dump(
                    {"doc_id": doc_id, "doc_title": doc["title"], "error": str(e)},
                    f,
                )
                f.write(",")

    logger.info("Refactoring the generated json files...")

    # close opened files
    refactor_json_files(
        [doc_paras_path, qas_path, unsuccessful_qas_path, unsuccessful_doc_path]
    )

    logger.info(
        f"Number of answerable questions-answer pairs extracted: {count_answerable_answers}"
    )
    logger.info("Validation loop is completed!!!")


# helpers


def get_online_html(url: str) -> str:
    """
    Gets online HTML of given webpage by actually accessing the webpage.
    Here online means the version that you see on your browser.

    :param url: url of a webpage
    :return: html of the webpage
    """
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    r = session.get(url)
    return r.text


def get_wikipedia_url(title: str) -> str:
    """
    Gets url of given webpage by hitting wikipedia api.

    :param title: title of a wikipedia webpage
    :return: url of the webpage
    """
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "info",
        "inprop": "url",
    }
    response = requests.get(WIKI_BASE_URL, params=params).json()
    pages = response["query"]["pages"]
    page_id = list(pages.keys())[0]
    return pages[page_id]["canonicalurl"]


def get_html_by_year(title: str, year: str = "2017") -> str:
    """
    Gets HTML of given webpage by hitting wikipedia api.
    (Squad 2.0 derives paragraphs from Squad 1.1 thats released in 2016)

    :param title: title of a wikipedia webpage
    :return: html of the webpage as string
    """
    params = {
        "action": "query",
        "format": "json",
        "prop": "revisions",
        "titles": title,
        "rvprop": "ids",
        "rvlimit": "1",
        "rvdir": "newer",
        "rvstart": year + "0101000000",
    }
    response = requests.get(WIKI_BASE_URL, params=params)
    data = json.loads(response.text)
    revision_id = data["query"]["pages"][list(data["query"]["pages"].keys())[0]][
        "revisions"
    ][0]["revid"]
    r = requests.get(f"{WIKI_API_BASE_URL}/{title}/{revision_id}")
    return r.text


def filter_sections_online_html(
    html: str, remove_images: bool = False, include_ref: bool = False
) -> str:
    """
    For online HTML: filters sections that come after wiki the main article paragraphs.

    :param html: online wiki html
    :param remove_images: if images are to be excluded
    :param include_ref: whether to include citation tags like '[5]'
    :return: filtered html which doesn't contain unwanted content
    """
    soup = BeautifulSoup(html, "lxml")
    delete_these = []

    article = soup.find("div", attrs={"class": "mw-content-container"})
    contents = article.find("div", attrs={"id": "mw-content-text"})
    paras = contents.find("div", attrs={"class": "mw-parser-output"}).find_all("p")

    if remove_images:
        for img_tag in soup.find_all("img"):
            delete_these.append(img_tag)

    current_tag = contents.next_sibling
    while current_tag is not None:
        delete_these.append(current_tag)
        current_tag = current_tag.next_sibling

    current_tag = article.next_sibling
    while current_tag is not None:
        delete_these.append(current_tag)
        current_tag = current_tag.next_sibling
    current_tag = article.previous_sibling
    while current_tag is not None:
        delete_these.append(current_tag)
        current_tag = current_tag.previous_sibling

    for para in paras[::-1]:
        if "." in para.text:
            current_tag = para.next_sibling
            break
    while current_tag is not None:
        delete_these.append(current_tag)
        current_tag = current_tag.next_sibling

    for tag in delete_these:
        tag.extract()

    if include_ref:
        return str(soup)
    else:
        ref_tags = soup.find_all("sup", attrs={"class": "reference"})
        for tag in ref_tags:
            tag.extract()
        return str(soup)


def filter_sections_2017_html(
    html: str, remove_images: bool = False, include_ref: bool = False
) -> str:
    """
    For 2017 HTML: filters sections that come after wiki the main article paragraphs.

    :param html: online wiki html
    :param remove_images: if images are to be excluded
    :param include_ref: whether to include citation tags like '[5]'
    :return: filtered html which doesn't contain unwanted content
    """
    soup = BeautifulSoup(html, "lxml")
    delete_these = []

    article = soup.find("body")
    sections = article.findChildren("section", recursive=False)

    to_break = 0
    for section in sections[::-1]:
        paras = section.find_all("p")
        for para in paras:
            if "." in para.text:
                to_break = 1
                break
        if to_break:
            current_tag = section.next_sibling
            break

    while current_tag is not None:
        delete_these.append(current_tag)
        current_tag = current_tag.next_sibling

    if remove_images:
        for img_tag in soup.find_all("img"):
            delete_these.append(img_tag)

    for tag in delete_these:
        tag.extract()

    if include_ref:
        return str(soup)
    else:
        ref_tags = soup.find_all("sup", attrs={"class": "mw-ref reference"})
        for tag in ref_tags:
            tag.extract()
        return str(soup)


def get_answer_sentence(context: str, answer_start_char_idx: int) -> str:
    """
    Finds sentence in which answer is present.

    :param context: a paragraph
    :param answer_start_char_idx: start position of answer in context
    :return: sentence containing the answer
    """
    sentence_start = context.rfind(".", 0, answer_start_char_idx) + 1
    sentence_end = context.find(".", answer_start_char_idx)
    if sentence_end == -1:
        return context[sentence_start:].strip()
    else:
        return context[sentence_start : sentence_end + 1].strip()


def validate_qas(
    ref_para_text: str,
    answer_obj: Dict,
    html_paras: List,
    threshold_min: float = 1.0,
    threshold_max: float = 1.0,
) -> Dict:
    """
    Validates a question-answer pair from SQuAD dataset and also
    returns in which paragraph of the webpage true answer is found.
    This is done by looking at how many words in answer sentence
    of dataset are present in answer sentence of webpage excluding
    stopwords.

    :param ref_para_text: wiki paragraph from dataset
    :param answer_obj: dict contain an answer's start position and text
    :param html_paras: list of wiki paragraphs from html
    :param threshold_min: match percentage threshold (0 to 1) above which
        question-answer pair is said to be valid
    :param threshold_max: match percentage threshold (0 to 1) below which
        question-answer pair is said to be valid
    :returns: a dictionary with following keys
        - is_validated - boolean/None
        - match_percentage - percentage of words in answer sentence of
            dataset are present in answer sentence of webpage
        - matched_para_id - index of paragraph in the webpage
        - answer_start_char_idx - start position of answer in the para corresponding to para_index
        - answer_end_char_idx - end position of answer in the para corresponding to para_index
    """
    answer = answer_obj["text"].lower()
    tokenizer = RegexpTokenizer(r"\w+")
    stop_words = set(stopwords.words("english"))

    # get reference answer sentence
    ref_match_start = answer_obj["answer_start"]
    ref_sentence_words = tokenizer.tokenize(
        get_answer_sentence(ref_para_text, ref_match_start)
    )
    ref_filtered_words = [word for word in ref_sentence_words if word not in stop_words]

    # get matching webpage answer sentence
    webpage_matches = [
        {"start_position": pos, "para_id": para_id}
        for para_id, para in enumerate(html_paras)
        for pos in find_substring_positions(answer, para.lower())
    ]
    all_webpage_filtered_words = []

    output = {
        "is_validated": None,
        "match_percentage": None,
        "matched_para_id": None,
        "answer_start_char_idx": None,
        "answer_end_char_idx": None,
    }

    if webpage_matches:

        # for every web match
        for i, webpage_match in enumerate(webpage_matches):
            webpage_sent = get_answer_sentence(
                html_paras[webpage_match["para_id"]], webpage_match["start_position"]
            ).lower()
            webpage_sent_words = tokenizer.tokenize(webpage_sent)
            all_webpage_filtered_words.append(
                [word for word in webpage_sent_words if word not in stop_words]
            )

        # checking matches are in ref sentences
        num_matches = [
            sum([1 for word in ref_filtered_words if word in words])
            for words in all_webpage_filtered_words
        ]
        selected_match_idx = num_matches.index(max(num_matches))

        output["match_percentage"] = num_matches[selected_match_idx] / len(
            ref_filtered_words
        )
        output["matched_para_id"] = webpage_matches[selected_match_idx]["para_id"]
        output["answer_start_char_idx"] = webpage_matches[selected_match_idx][
            "start_position"
        ]
        output["answer_end_char_idx"] = (
            output["answer_start_char_idx"] + len(answer) - 1
        )

        # if the match is above threshold -> save it or else ignore
        if (
            output["match_percentage"] >= threshold_min
            and output["match_percentage"] <= threshold_max
        ):
            # successful match
            output["is_validated"] = True
            return output
        else:
            # match found but below threshold
            output["is_validated"] = False
            return output
    else:
        # no matches found in whole webpage
        return output


def find_substring_positions(substring: str, string: str) -> List[int]:
    """
    Returns positions at which a substring is matched in given string.

    :param substring: substring to be searched
    :param string: string to be searched in
    :return: list of start positions at which substring is present
    """
    start = 0
    matched_positions = []
    while start < len(string):
        i = string.find(substring, start)
        if i == -1:
            break
        matched_positions.append(i)
        start = i + 1
    return matched_positions


def create_html_file(html: str, html_path: str):
    """
    Given .html file path, it saves HTML at the location

    :param html: any html
    :param html_path: path of html file to be created
    """
    html = html.replace(
        "//upload.wikimedia.org", "https://upload.wikimedia.org"
    ).replace("//en.wikipedia.org", "https://en.wikipedia.org")
    html = html.replace(
        "/static/images", "https://en.wikipedia.org/static/images"
    ).replace("/w/load.php?", "https://en.wikipedia.org/w/load.php?")
    with open(html_path, "w", encoding="utf-8") as file:
        file.write(html)


def get_impossible_qas_para(ref_para_text: str, html_paras: List[str]) -> int:
    """
    Gets para_id of paragraph from html_paras that best matches with the ref_para_text

    :param ref_para_text: reference paragraph from original SQuAD 2.0 dataset
    :param html_paras: list of paragraphs from a webpage
    :return: paragraph id that is nearest match to the reference paragraph
    """
    tokenizer = RegexpTokenizer(r"\w+")
    stop_words = set(stopwords.words("english"))

    # get reference para words
    ref_para_words = tokenizer.tokenize(ref_para_text.lower())
    ref_filtered_words = [word for word in ref_para_words if word not in stop_words]

    all_webpage_filtered_words = []
    # for each webpage para
    for web_para in html_paras:
        webpage_para_words = tokenizer.tokenize(web_para.lower())
        # save words that match with reference para
        all_webpage_filtered_words.append(
            [word for word in webpage_para_words if word not in stop_words]
        )

    # count matches in ref sentences
    num_matches = [
        sum([1 for word in ref_filtered_words if word in words])
        for words in all_webpage_filtered_words
    ]

    # get the para with highest match
    selected_match_idx = num_matches.index(max(num_matches))

    return selected_match_idx


###################################################
# SQuAD-like json creation
###################################################

# main


def create_squad_like_json(output_dir: str):
    """
    Creates SQuAD-like JSON based on JSONs produced by collect_docs
    function.

    :param output_dir: directory in which the base JSON files are
        found and in which SQuAD-like JSON is to be saved
    """

    # read files
    doc_paras_path = os.path.join(output_dir, "doc_paras.json")
    with open(doc_paras_path, "r", encoding="utf-8") as f:
        df_doc_paras = pd.DataFrame.from_dict(json.load(f))
    qas_path = os.path.join(output_dir, "qas.json")
    with open(qas_path, "r", encoding="utf-8") as f:
        df_qas = pd.DataFrame.from_dict(json.load(f))

    # consider only the questions that are validated
    df_qas = df_qas[df_qas["is_validated"] == True]
    doc_id_grouped = df_qas.groupby(["doc_id"])

    data = []

    for doc_id, doc_id_group in doc_id_grouped:
        paras = df_doc_paras[df_doc_paras["doc_id"] == doc_id]["paras"].iloc[0]
        title = df_doc_paras[df_doc_paras["doc_id"] == doc_id]["title"].iloc[0]

        para_ids = doc_id_group["idx_para_matched"].unique()
        para_grouped = doc_id_group.groupby(["idx_para_matched"])

        paragraphs = []

        for para_id in para_ids:
            para_group = para_grouped.get_group(para_id)
            qas_grouped = para_group.groupby(["qas_id"])

            qas = []

            for qas_id, group in qas_grouped:
                # question is unanswerable
                if group["is_impossible"].iloc[0]:
                    # save question-answer information
                    qas.append(
                        {
                            "id": qas_id[0],
                            "is_impossible": True,
                            "question": group["question"].iloc[0],
                            "answers": [],
                            "plausible_answers": group["plausible_answers"].iloc[0],
                        }
                    )
                # question is answerable
                else:
                    answer_group = group.drop_duplicates(
                        ["answer_text", "answer_start_char_idx", "answer_end_char_idx"],
                        keep="last",
                    )
                    # save question-answer information
                    qas.append(
                        {
                            "id": qas_id[0],
                            "is_impossible": False,
                            "question": group["question"].iloc[0],
                            "answers": [
                                {
                                    "text": group["answer_text"].to_list()[i],
                                    "answer_start_char_idx": int(
                                        group["answer_start_char_idx"].to_list()[i]
                                    ),
                                    "answer_end_char_idx": int(
                                        group["answer_end_char_idx"].to_list()[i]
                                    ),
                                }
                                for i in range(len(answer_group["answer_text"]))
                            ],
                        }
                    )

            # save paragraph information
            paragraphs.append(
                {"context": paras[int(para_id)], "qas": qas, "para_id": int(para_id)}
            )

        # save document information
        data.append(
            {"paragraphs": paragraphs, "title": title, "doc_id": int(doc_id[0])}
        )  # new pandas version sets doc_id as tuple

    squad_like_json_path = os.path.join(output_dir, "squad_like_json.json")
    with open(squad_like_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f)


###################################################
# PDFs, page image assets generation
###################################################


def generate_pdfs_images_bboxes(
    output_dir: str, image_width: Optional[int], image_height: Optional[int]
):
    """
    Generates PDFs, images, words, bounding boxes of HTMLs in output_dir.

    :param output_dir: directory in which htmls directory is found
    :param size_factor: factor by which the HTMLs are to be zoomed while
        generating images of the webpages
    """
    doc_paras_path = os.path.join(output_dir, "doc_paras.json")
    with open(doc_paras_path, "r") as f:
        df_doc_paras = pd.DataFrame.from_dict(json.load(f))

    for doc_id in tqdm(df_doc_paras["doc_id"], total=len(df_doc_paras["doc_id"])):
        doc_id_dir = os.path.join(output_dir, "visual_squad", str(doc_id))
        if not os.path.exists(doc_id_dir):
            os.makedirs(doc_id_dir)

        pdf_path = os.path.join(doc_id_dir, "pdf.pdf")
        html_path = os.path.join(output_dir, "htmls", f"{doc_id}.html")
        html_to_pdf(html_path, pdf_path)

        pages_dir = os.path.join(doc_id_dir, "pages")
        save_images_words_bboxes(
            pdf_path, pages_dir, width=image_width, height=image_height
        )


# helpers


def html_to_pdf(html_path: str, pdf_path: str):
    """
    Create a PDF based on given HTML file

    :param html_path: html file based on which pdf is to be created
    :param pdf_path: path of pdf file to be created
    """
    command = f"""{WKHTMLTOPDF_PATH} --quiet --enable-local-file-access {html_path} {pdf_path}"""
    try:
        run(command)
    except FileNotFoundError:
        logger.error(
            "Binary files of 'wkhtmltopdf' are missing, make sure you have installed them and set \
                      'WKHTMLTOPDF_PATH' to the path where wkhtmltopdf.exe lives."
        )


###################################################
# Modelling data generation
###################################################


def create_modelling_data(output_dir: str):
    """
    Generates json files that are ingested as data to textual and multi-modal models

    :param output_dir: directory in which preprocessing outputs (doc_paras.json,
        qas.json) exist
    """

    # Reading generated data...

    doc_paras_path = os.path.join(output_dir, "doc_paras.json")
    with open(doc_paras_path, "r", encoding="utf-8") as f:
        df_doc_paras = pd.DataFrame.from_dict(json.load(f))

    qas_path = os.path.join(output_dir, "qas.json")
    with open(qas_path, "r", encoding="utf-8") as f:
        df_qas = pd.DataFrame.from_dict(json.load(f))

    # Performing checks...

    # Keep only validated questions
    df_qas = df_qas[df_qas["is_validated"] == True]

    # Keep only possible questions
    df_qas = df_qas[df_qas["is_impossible"] == False]

    df_qas = df_qas.astype(
        {
            "answer_start_char_idx": int,
            "answer_end_char_idx": int,
            "idx_para_matched": int,
        }
    )

    # original data contains answer duplicates - drop them
    df_qas.drop_duplicates(
        ["qas_id", "answer_start_char_idx", "answer_end_char_idx", "answer_text"],
        keep="last",
        inplace=True,
    )

    # drop unwanted columns
    df_qas.drop(
        columns=[
            "doc_title",
            "plausible_answers",
            "percent_match",
            "is_impossible",
            "is_validated",
        ],
        inplace=True,
    )

    # Updating data frame...

    # update start, end char positions + add image_path column
    df_qas = updated_df_qas(df_qas, df_doc_paras, output_dir)

    df_qas.drop(
        columns=[
            "idx_para_matched",
            "doc_id",
        ],
        inplace=True,
    )

    # Saving preliminary modelling data...

    modelling_data_json_path = os.path.join(output_dir, "modelling_data.json")
    with open(modelling_data_json_path, "w", encoding="utf-8") as f:
        f.write(df_qas.to_json(orient="records", lines=True).replace("}", "},")[:-1])

    refactor_json_files([modelling_data_json_path])


# helpers


def updated_df_qas(df_qas: pd.DataFrame, df_doc_paras: pd.DataFrame, output_dir: str):
    """
    Updates df_qas by offsetting start, end char positions (from being w.r.t. para
    to being w.r.t. page without \n's) and adds image_path column

    :param df_qas: dataframe containing the question answer pairs
    :param df_doc_paras: dataframe containing the document para pairs
    :param output_dir: directory containing all Visual SQuAD data
    :return: updated dataframe of question answer pairs
    """
    relative_output_dir_path = os.path.relpath(output_dir)

    df_qas["image_path"] = None

    grouped_by_doc_id = df_qas.groupby("doc_id")

    for doc_id in df_qas["doc_id"].unique():
        pages_dir = os.path.join(
            os.path.join(*(relative_output_dir_path.split(os.path.sep)[1:])),
            "visual_squad",
            str(doc_id),
            "pages",
        )
        page_nrs = [
            dir_name for dir_name in os.listdir(os.path.join(SRC_DIRECTORY, pages_dir))
        ]

        for page_nr in page_nrs:
            page_text_file_path = os.path.join(
                SRC_DIRECTORY, pages_dir, page_nr, "text.txt"
            )

            with open(page_text_file_path, "r", encoding="utf-8") as file:
                page_text = file.read().replace("\n", " ")

            para_ids = grouped_by_doc_id.get_group(doc_id)["idx_para_matched"].unique()

            for para_id in para_ids:
                para_text = df_doc_paras[df_doc_paras["doc_id"] == doc_id][
                    "paras"
                ].values[0][para_id]
                index_matched = page_text.find(para_text)

                if index_matched != -1:
                    df_qas.loc[
                        (df_qas["idx_para_matched"] == para_id)
                        & (df_qas["doc_id"] == doc_id),
                        "image_path",
                    ] = os.path.join(pages_dir, page_nr, "image.png")
                    df_qas.loc[
                        (df_qas["idx_para_matched"] == para_id)
                        & (df_qas["doc_id"] == doc_id),
                        "answer_start_char_idx",
                    ] += index_matched
                    df_qas.loc[
                        (df_qas["idx_para_matched"] == para_id)
                        & (df_qas["doc_id"] == doc_id),
                        "answer_end_char_idx",
                    ] += index_matched

    df_qas = df_qas[df_qas["image_path"].notna()]

    return df_qas
