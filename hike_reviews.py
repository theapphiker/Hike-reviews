import requests
import bs4
import regex as re
import spacy
import string
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from urllib.parse import urlparse
import sys

NLP = spacy.load("en_core_web_sm")

CV = joblib.load(
    "vectorizer"
) # load the vectorizer 

NB = joblib.load(
    "NB_model"
) # load the pre-trained Naive Bayes model

GOOD_COMMENTS = []

BAD_COMMENTS = []


def main():
    """
    The main function coordinates the program's execution. It enters a loop to continuously prompt the user for a valid URL until they exit.
    For each valid URL, it extracts comments from the corresponding hike page on hikingupward.com, classifies them as positive or negative using a loaded Naive Bayes model,
    and then identifies the most common positive and negative aspects mentioned in the comments. Finally, it presents these findings to the user.
    """
    while True:
        hiking_url = is_valid_url()
        hiking_comments = parse_hiking_url(hiking_url)
        parsed_comments = parse_comments(hiking_comments)
        final_good_comments = ", ".join(assigned_comments(GOOD_COMMENTS))
        final_bad_comments = ", ".join(assigned_comments(BAD_COMMENTS))
        if final_good_comments == "":
            print("Hikers did not have any common likes about this hike.")
        else:
            print("Hikers liked this stuff: " + final_good_comments)
        if final_bad_comments == "":
            print("Hikers did not have any common dislikes about this hike.")
        else:
            print("Hikers disliked this stuff: " + final_bad_comments)
        sys.exit()


def is_valid_url():
    """
    This function validates user input to ensure it points to a valid hike review page on hikingupward.com.
    It enters a loop prompting the user until a valid URL is provided. A valid URL starts with "https", points to "www.hikingupward.com",
    and has a path that begins with one of the specified hiking area codes.
    """
    while True:
        print("Please enter the url for the hikingupward.com hike.")
        url = input()
        # attempt to parse the url
        result = urlparse(url)
        # check if all components are present
        # hiking_areas will need to be updated as needed
        hiking_areas = [
            "GWNF",
            "GSMNP",
            "JNF",
            "MNF",
            "NNF",
            "PNF",
            "SNP",
            "WMNF",
            "UNF",
        ]
        if (
            result.scheme == "https"
            and result.netloc == "www.hikingupward.com"
            and result.path.split("/")[1] in hiking_areas
        ):
            return url.rstrip().lstrip()
        else:
            print("Please enter a valid url for a hike on hikingupward.com.")
            continue


def parse_hiking_url(hiking_url):
    """
    This function takes a URL for a hike on hikingupward.com and fetches the list of comments from that hike's page.
    If the page has no comments, it returns the message "There are no comments."
    """
    original_request = requests.get(hiking_url)
    original_soup = bs4.BeautifulSoup(original_request.text, "html5lib")
    comments_link = ""
    for a in original_soup.find_all("a"):
        if "all_reviews" in str(a):
            comments_link = "https://www.hikingupward.com" + str(a["href"])
    if comments_link == "":
        return "There are no comments."
    else:
        return get_comments(comments_link)


def get_comments(comments_link):
    """
    This function takes a URL for the comments page of a hike on hikingupward.com and extracts a list of comments from that page.
    """
    comments_request = requests.get(comments_link)
    comments_soup = bs4.BeautifulSoup(comments_request.text, "html5lib")
    comments = [
        comment.get_text()
        for comment in comments_soup.select("font")
        if 'font size="1"' in str(comment)
    ]
    return comments


def filter_tokens(txt):
    """
    This function preprocesses a text string by removing stop words and punctuation.
    """
    doc = NLP(txt)
    # Filter tokens based on stop word and punctuation checks
    return " ".join(
        token.text
        for token in doc
        if not token.is_stop and str(token) not in string.punctuation
    )


def parse_comments(comments):
    """
    This function analyzes a list of comments about a hike, classifying them as positive or negative
    using a pre-trained Naive Bayes model, and storing the classified comments in global variables.
    """
    for comment in comments:
        for sentence in re.split("[.!]", comment):
            token_sentence = filter_tokens(sentence)
            transformed_sentence = CV.transform([token_sentence])
            if NB.predict(transformed_sentence) == ["GOOD"]:
                GOOD_COMMENTS.append(sentence)
            elif NB.predict(transformed_sentence) == ["BAD"]:
                BAD_COMMENTS.append(sentence)


def get_noun_adj_pairs(doc, verbose=False):
    """Return tuples of noun and adjective for each document."""
    compounds = [tok for tok in doc]  # Get list of compounds in doc
    tuple_list = []
    if compounds:
        for tok in compounds:
            pair_item_1, pair_item_2 = (False, False)  # initialize false variables
            noun = doc[tok.i]
            pair_item_1 = noun
            # If noun is in the subject, we may be looking for adjective in predicate
            # In simple cases, this would mean that the noun shares a head with the adjective
            if noun.dep_ == "nsubj":
                adj_list = [r for r in noun.head.rights if r.pos_ == "ADJ"]
                if adj_list:
                    pair_item_2 = adj_list[0]
                if (
                    verbose == True
                ):  # For trying different dependency tree parsing rules
                    print("Noun: ", noun)
                    print("Noun root: ", noun.root)
                    print("Noun root head: ", noun.root.head)
                    print(
                        "Noun root head rights: ",
                        [r for r in noun.root.head.rights if r.pos_ == "ADJ"],
                    )
            if pair_item_1 and pair_item_2:
                tuple_list.append((str(pair_item_1), str(pair_item_2)))
    return tuple_list


def assigned_comments(comments):
    """
    This function analyzes a list of comments and identifies the most common positive adjectives used to describe the hike.
    """
    comments_counter = Counter()
    for pair in get_noun_adj_pairs(NLP(" ".join(comments))):
        comments_counter[pair[1]] += 1
    return [e[0] for e in comments_counter.most_common(3) if e[1] > 1] # don't add comment adjective if it only appeared in one comment


if __name__ == "__main__":
    main()
