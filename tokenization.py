
import six
import collections
import unicodedata
from model_params import params

def convert_to_unicode(text):
    if six.PY3:
        if isinstance(text,str):
            return text
        elif isinstance(text,bytes):
            return text.decode("utf-8","ignore")
        else:
            raise ValueError("Unsupported string type {}".format(type(text)))

def load_vocab(vocab_file):
    vocab=collections.OrderedDict()
    index_vocab=collections.OrderedDict()
    index=0

    with open(vocab_file,'rb') as reader:
        while True:
            tmp=reader.readline()
            token=convert_to_unicode(tmp)

            if not token:
                break

            token=token.strip()
            vocab[token]=index
            index_vocab[index]=token
            index+=1
        params.update(vocab_size=len(vocab))
    return vocab,index_vocab

def convert_tokens_to_ids(vocab, tokens):
    """Converts a sequence of tokens into ids using the vocab."""
    ids = []
    for token in tokens:
        ids.append(vocab[token])
    return ids


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a peice of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens

class FullTokenizer(object):

    def __init__(self,vocab_file,do_lower_case=True):
        self.vocab,self.index_vocab=load_vocab(vocab_file)
        self.basic_tokenizer= BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer=WordpieceTokenizer(vocab=self.vocab)

    def tokenize(self,text):
        split_tokens=[]
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
        return split_tokens

    def convert_tokens_to_ids(self,tokens):
        return convert_tokens_to_ids(self.vocab,tokens)


class BasicTokenizer(object):

    def __init__(self,do_lower_case=True):
        self.do_lower_case=do_lower_case

    def tokenize(self,text):
        text=convert_to_unicode(text)
        text=self._clean_text(text)

        text=self._tokenize_chinese_chars(text)
        orig_tokens=whitespace_tokenize(text)
        split_tokens=[]
        for token in orig_tokens:
            if self.do_lower_case:
                token=token.lower()
                token=self._run_strip_accents(token)
            split_tokens.append(self._run_split_on_punc(token)[0])

        output_token=whitespace_tokenize(" ".join(split_tokens))
        return output_token

    def _run_strip_accents(self,text):
        text=unicodedata.normalize("NFD",text)
        output=[]
        for char in text:
            cat=unicodedata.category(char)
            if cat=="Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self,text):
        chars=list(text)
        i=0
        start_new_word=True
        output=[]
        while i <len(chars):
            char=chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word=True
            else:
                if start_new_word:
                    output.append([])
                start_new_word=False
                output[-1].append(char)
            i+=1
        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self,text):
        output=[]
        for char in text:
            cp=ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self,cp):
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.
        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.
        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]
        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer.
        Returns:
          A list of wordpiece tokens.
        """

        text = convert_to_unicode(text)

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False



