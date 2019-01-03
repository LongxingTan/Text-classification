
import six

def convert_to_unicode(text):
    if six.PY3:
        if isinstance(text,str):
            return text
        elif isinstance(text,bytes):
            return text.decode("utf-8","ignore")
        else:
            raise ValueError("Unsupported string type {}".format(type(text)))


